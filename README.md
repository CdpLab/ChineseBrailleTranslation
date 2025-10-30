<div id="top" align="center">
  
# An End-to-End Chinese-Braille Translation Method Based on mT5: Vocabulary Expansion and Structural Enhancement
  
  Dapeng Chen, Chenkai Li, Zhou Zhuang, Lina Wei, Jia Liu*
  
</div>

## Abstract
With the increasing demand for visually impaired people to access information and integrate into society, Chinese-Braille automatic translation plays an increasingly important role in accessible communication and assisted education. However, existing methods still face challenges in handling polyphonic characters, the appropriateness of word-segmentation (WS), and long-sequence generation. To this end, we propose an end-to-end Chinese-Braille translation model based on mT5-small model. The model combines Braille-character tokenization strategy, Mixture of Braille Experts (MBE), and Boundary-based Generation (BBG) mechanism to improve performance in character-level generation and WS prediction. We constructed a Chinese-Braille parallel corpus and trained the proposed model on this corpus. Ablation results show that the tokenization strategy enhances the advantage of character-level generation, MBE improves the model's ability to discriminate among different characters, and BBG effectively alleviates errors in boundary prediction. Comparative experiments show that the proposed model improves the BLEU score by more than 15% over the baseline mT5-small, reaching as high as 98.3%. Meanwhile, the average generated sequence length is reduced from 207 to 99, and the average inference time per sentence is reduced from 78 ms to 40 ms, yielding a 48.7% improvement in efficiency. Therefore, the proposed model provides an efficient solution for Chinese-Braille automatic translation.


## Setup Environment
 
 We recommend using a conda environment to manage dependencies.
 
```bash
conda create -n braille_env python=3.10
conda activate braille_env
pip install -r requirements.txt
 ```
 
The installation of `pytorch` may vary depending on your system.  
Please refer to the [official website](https://pytorch.org) for more information.

All the training and evaluation scripts use `accelerate` to speed up the training process.  
If you want to run the scripts without `accelerate`, you can remove the related code in the scripts.  
Remember to run `accelerate config` before you run our scripts, or you may encounter some errors.

###  Add Braille Characters as New Tokens

Before fine-tuning, all Braille Unicode characters must be added to the mT5 tokenizer to ensure that the model can recognize and generate Braille symbols directly.

```bash
python down_model.py
```

### Training

The fine-tuning process is based on the mT5-small model that has been extended with Braille character tokens. After downloading and preparing the tokenizer with all unique Braille symbols from the dataset, the model is fine-tuned on the Chinese–Braille parallel corpus located in `Chinese_braille_data/Parallel Corpus`. The training uses the standard sequence-to-sequence objective of mT5, where the input is a Chinese sentence and the output is the corresponding Braille transcription. The fine-tuned model is saved to `save_model/` and can later be evaluated using the provided evaluation script.

```bash
bash train.sh
```
### Evaluate on the Validation and Test Set

The model evaluation is conducted on the validation and test sets in `Chinese_braille_data/Parallel Corpus`. During evaluation, the model generates Braille outputs, which are compared with the reference Braille texts to assess translation quality. BLEU is used as the primary metric to evaluate the model’s performance on the Chinese-to-Braille translation task. The evaluation is performed using the evaluation.sh script, and all results and logs are automatically saved to the `save_model/evaluation-final/` directory for further analysis and model comparison.

```bash
bash evaluation.sh
```

### Model Architecture

![Model Architecture](./image/Model%20Architecture%20Diagram.png)


