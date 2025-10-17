#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import sys, os
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.42.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/translation/requirements.txt",
)


# === BBG/MBE imports ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from transformers.modeling_outputs import Seq2SeqLMOutput

logger = logging.getLogger(__name__)

# === Inject BBG (Boundary Gate) + MBE (Mixture-of-Braille-Experts) into model ===
def patch_bbg_mbe(model, tokenizer,
                  n_experts: int = 4,
                  router_tau: float = 1.0,
                  router_topk: int = 0,
                  alpha: float = 1.0,
                  beta: float = 1.0,
                  lambda_bg: float = 0.5,
                  lambda_ent: float = 0.0):
    """Modify the loaded Seq2Seq model *in-place*:
    - Attach BBG and MBE submodules as children of `model`
    - Monkey-patch `model.forward` so both training *and* generation use the modified logits.
    """
    # Collect Braille token ids and sp id
    vocab = tokenizer.get_vocab()
    braille_ids = sorted({tid for tok, tid in vocab.items() if (len(tok) == 1 and '\u2800' <= tok <= '\u28FF')})
    assert len(braille_ids) > 0, "No Braille tokens found in tokenizer. Please add them first."

    sp_id = tokenizer.convert_tokens_to_ids("sp")
    if sp_id == tokenizer.unk_token_id:
        sp_id = tokenizer.convert_tokens_to_ids("▁sp")
    assert sp_id != tokenizer.unk_token_id, "Cannot find token id for 'sp' (or '▁sp')."
    d_model = model.config.d_model

    # Attach submodules (become part of model state dict)
    model.bbg = nn.Linear(d_model, 1)
    nn.init.normal_(model.bbg.weight, std=1e-3); nn.init.zeros_(model.bbg.bias)

    model.router = nn.Linear(d_model, n_experts)
    nn.init.normal_(model.router.weight, std=1e-3); nn.init.zeros_(model.router.bias)

    model.experts = nn.ModuleList([nn.Linear(d_model, len(braille_ids)) for _ in range(n_experts)])
    for e in model.experts:
        nn.init.normal_(e.weight, std=1e-3); nn.init.zeros_(e.bias)

    # Hyperparameters stored on the model
    model.register_buffer("braille_idx_tensor", torch.tensor(braille_ids, dtype=torch.long))
    model._sp_id = int(sp_id)
    model._bbg_alpha = float(alpha)
    model._mbe_beta = float(beta)
    model._router_tau = float(router_tau)
    model._router_topk = int(router_topk)
    model._lambda_bg = float(lambda_bg)
    model._lambda_ent = float(lambda_ent)

    # Keep original forward
    if not hasattr(model, "_orig_forward"):
        model._orig_forward = model.forward
    return model

import types
from typing import Any, Dict

def _forward_with_bbg_mbe(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs: Dict[str, Any]
):
    # —— 1) generate() 编码阶段常直接传 encoder_outputs；遇到就原样回退 —— 
    if kwargs.get("encoder_outputs", None) is not None:
        # 不要过滤 kwargs，保持 generate() 预期
        return self._orig_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )

    # —— 2) 二选一兜底：确保有 input_ids 或 inputs_embeds —— 
    if inputs_embeds is None and input_ids is None:
        raise ValueError("[BBG/MBE] Need either input_ids or inputs_embeds, got neither.")

    # 如果没给 embeds，则用共享嵌入映射；后续我们在 logits 上做修改
    if inputs_embeds is None and input_ids is not None:
        base_embeds = self.shared(input_ids)  # T5/mT5 的共享嵌入
        inputs_embeds = base_embeds
        input_ids = None  # 传了 embeds 就不要再传 ids（T5 规则）

    # —— 3) 先跑一遍原始 forward 拿到 logits/hidden_states —— 
    base_out = self._orig_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_hidden_states=True,   # 我们要 decoder_hidden_states
        return_dict=True,
        **kwargs
    )

    # —— 4) 你原有的 BBG/MBE 逻辑（保持不变） —— 
    H = base_out.decoder_hidden_states[-1] if hasattr(base_out, 'decoder_hidden_states') else None
    logits = base_out.logits  # [B,T,V]
    V = logits.size(-1)

    cfg = {
        "alpha": getattr(self, "_bbg_alpha", 1.0),
        "beta": getattr(self, "_mbe_beta", 1.0),
        "router_tau": getattr(self, "_router_tau", 1.0),
        "router_topk": getattr(self, "_router_topk", 0),
        "lambda_bg": getattr(self, "_lambda_bg", 0.5),
        "lambda_ent": getattr(self, "_lambda_ent", 0.0),
    }

    bidx = self.braille_idx_tensor.to(logits.device) if hasattr(self, 'braille_idx_tensor') else None
    ent = H.new_tensor(0.0) if H is not None else logits.new_tensor(0.0)

    gate = None
    # ===== MBE：只覆盖盲文切片 =====
    if bidx is not None and bidx.numel() > 0 and cfg["beta"] != 0.0 and H is not None:
        w = self.router(H) / max(cfg["router_tau"], 1e-9)
        pi = torch.softmax(w, dim=-1)  # [B,T,E]
        if cfg["router_topk"] and 0 < cfg["router_topk"] < pi.size(-1):
            topk = torch.topk(pi, cfg["router_topk"], dim=-1)
            mask = torch.zeros_like(pi).scatter(-1, topk.indices, 1.0)
            pi = (pi * mask) / (mask.sum(dim=-1, keepdim=True) + 1e-9)
        ent = -(pi * (pi.clamp_min(1e-9)).log()).sum(dim=-1).mean()

        exp_stack = torch.stack([e(H) for e in self.experts], dim=-1)  # [B,T,|B|,E]
        logits_b = (exp_stack * pi.unsqueeze(-2)).sum(dim=-1)          # [B,T,|B|]

        base_slice = logits.index_select(-1, bidx)                      # [B,T,|B|]
        mixed = cfg["beta"] * logits_b + (1.0 - cfg["beta"]) * base_slice
        logits = logits.scatter(-1, bidx.unsqueeze(0).unsqueeze(0).expand_as(mixed), mixed)

    # ===== BBG：给 <sp> logit 注入偏置 =====
    sp_id = getattr(self, "_sp_id", None)
    if sp_id is not None and cfg["alpha"] != 0.0 and H is not None:
        gate = torch.sigmoid(self.bbg(H)).squeeze(-1)  # [B,T]
        logits[..., sp_id] = logits[..., sp_id] + cfg["alpha"] * gate

    # ===== 在修改后的 logits 上重算损失 =====
    loss = None
    if labels is not None:
        ce = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
        loss = ce
        if gate is not None and cfg["lambda_bg"] > 0.0:
            with torch.no_grad():
                valid = (labels != -100).float()
                target_sp = (labels == sp_id).float() * valid
            bce = F.binary_cross_entropy(gate, target_sp, reduction="none")
            loss_bg = (bce * valid).sum() / (valid.sum() + 1e-9)
            loss = loss + cfg["lambda_bg"] * loss_bg
        if cfg["lambda_ent"] > 0.0 and bidx is not None and bidx.numel() > 0 and cfg["beta"] != 0.0:
            loss = loss + cfg["lambda_ent"] * (-ent)

    return Seq2SeqLMOutput(
        loss=loss,
        logits=logits,
        past_key_values=base_out.past_key_values,
        decoder_hidden_states=None,
        decoder_attentions=getattr(base_out, "decoder_attentions", None),
        cross_attentions=getattr(base_out, "cross_attentions", None),
        encoder_last_hidden_state=base_out.encoder_last_hidden_state,
        encoder_hidden_states=None,
        encoder_attentions=getattr(base_out, "encoder_attentions", None),
    )

    # Bind new forward
    model.forward = types.MethodType(_forward_with_bbg_mbe, model)
    return model


# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer,
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/media/ubuntu/DATA/lck/mt5_small",
        metadata={
            "help": "LOCAL model directory path (必须包含 model.safetensors 或 pytorch_model.bin 和 config.json)"
                    "\n示例: /media/ubuntu/DATA/lck/model"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="/media/ubuntu/DATA/lck/mt5_small",
        metadata={
            "help": "分词器目录路径（需包含 spiece.model 和 tokenizer_config.json）"
                    "\n默认与模型目录相同"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "【重要】必须关闭！因使用 SentencePiece 分词器（spiece.model）"
                    "\n开启会导致加载失败"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": "【本地模型无需设置】仅远程私有模型需要"
                    "\n保留为 None 即可"
        }
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "【安全强制】必须关闭！本地模型无需远程代码执行"
                    "\n开启可能导致安全风险"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(
        default="Chinese", metadata={"help": "Source language id for translation."}
    )
    target_lang: str = field(
        default="Braille", metadata={"help": "Target language id for translation."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default="/media/ubuntu/DATA/lck/data/train.json",
        metadata={"help": "The input training data file (a jsonlines/json)."}
    )
    validation_file: Optional[str] = field(
        default="/media/ubuntu/DATA/lck/data/val.json",
        metadata={
            "help": "An optional input evaluation data file (json/jsonl)."
        },
    )
    test_file: Optional[str] = field(
        default="/media/ubuntu/DATA/lck/data/test.json",
        metadata={
            "help": "An optional input test data file (json/jsonl)."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=os.cpu_count() // 2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=285,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=285,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="translate Chinese to Braille：",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        # elif self.source_lang is None or self.target_lang is None:
        #     raise ValueError("Need to specify the source language and the target language.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert (
                    extension in valid_extensions
            ), "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert (
                    extension in valid_extensions
            ), "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def get_var_types(
        model_args, data_args, training_args
) -> tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments]:
    return model_args, data_args, training_args


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = get_var_types(
        model_args, data_args, training_args
    )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.


    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        filename=os.path.join(training_args.output_dir, "training_log.txt"),
        encoding='utf-8'  # 添加编码参数
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                filename=os.path.join(training_args.output_dir, "training_log.txt"),
                encoding="utf-8"
            )
        ]
    )

    if training_args.should_log:
        print("Start logging...")
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_debug()
        file_handler.setLevel("DEBUG")
        logger.addHandler(file_handler)
        transformers.utils.logging.add_handler(file_handler)
        logging.root.addHandler(file_handler)

    log_level = training_args.get_process_log_level()
    print("**** log_level: ", log_level)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "google-t5/t5-small",
        "google-t5/t5-base",
        "google-t5/t5-large",
        "google-t5/t5-3b",
        "google-t5/t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if extension == "jsonl":
            builder_name = (
                "json"  # the "json" builder reads both .json and .jsonl files
            )
        else:
            builder_name = extension  # e.g. "parquet"
        raw_datasets = load_dataset(
            builder_name,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    
    # === Inject BBG & MBE heads into the model (internal module changes) ===
    try:
        model = patch_bbg_mbe(
            model, tokenizer,
            n_experts=4,      # number of experts
            router_tau=1.0,   # router temperature
            router_topk=0,    # >0 for sparse routing
            alpha=1.0,        # BBG strength for <sp> logit
            beta=1.0,         # MBE replacement ratio over Braille slice
            lambda_bg=0.5,    # BCE weight for boundary supervision
            lambda_ent=0.0    # router entropy regularization
        )
        logger.info("[BBG+MBE] Successfully patched model with internal modules.")
    except Exception as e:
        logger.error(f"[BBG+MBE] Failed to patch model: {e}")
        raise
# Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(
            tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
                data_args.target_lang
            ]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                data_args.target_lang
            )

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
                data_args.target_lang is not None and data_args.source_lang is not None
        ), (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token]
            if data_args.forced_bos_token is not None
            else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the language codes for input/target.
    # source_lang = data_args.source_lang.split("_")[0]
    # target_lang = data_args.target_lang.split("_")[0]

    # Check the whether the source target length fits in the model, if it has absolute positional embeddings
    if (
        hasattr(model.config, "max_position_embeddings")
        and not hasattr(model.config, "relative_attention_max_distance")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        raise ValueError(
            f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
            f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
            f" `--max_source_length` to {model.config.max_position_embeddings} or using a model with larger position "
            "embeddings"
        )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
            model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enab256led but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        pref = data_args.source_prefix or ""
        if pref and not pref.endswith(" "):
            pref += " "
        inputs = [f"{pref}{x}" for x in examples["input_text"]]
        targets = examples["output_text"]

        # ——只打印一次，方便确认——
        if not hasattr(preprocess_function, "_dumped"):
            print("[PREPROCESS] prefix =", repr(pref))
            print("[PREPROCESS] raw    =", repr(examples["input_text"][0][:80]))
            print("[PREPROCESS] with   =", repr(inputs[0][:80]))
            preprocess_function._dumped = True
        # ————————————————

        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding=padding, truncation=True
        )
        labels = tokenizer(
            text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
                desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = evaluate.load("/media/ubuntu/DATA/lck/metrics/sacrebleu")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        logger.debug(
            "Example output: {}".format(
                json.dumps(
                    {
                        "pred": decoded_preds[0],
                        "label": decoded_labels[0],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        )

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.save_model()  # Saves the tokenizer too for easy upload


        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        logger.info(f"Params: {max_length=}, {num_beams=}")

        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(
                    predictions != -100, predictions, tokenizer.pad_token_id
                )
                predictions = tokenizer.batch_decode(
                    predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [
        l for l in [data_args.source_lang, data_args.target_lang] if l is not None
    ]
    if len(languages) > 0:
        kwargs["language"] = languages

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
