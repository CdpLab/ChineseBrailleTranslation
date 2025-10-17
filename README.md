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
