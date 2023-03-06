# FVW
Implementation of [FVW: Finding Valuable Weight on Deep Neural Network for Model Pruning]

## Attention
Please download the weights file from [here](https://drive.google.com/file/d/1UXGUXzU83i69bJRyioAuuEUDQ0STAsgq/view?usp=share_link) and extract it to the `masks/weights` folder. The weights file contains the weights of the model before pruning.

## Setup
Run `pip install -r requirements.txt` to install the dependencies. 


```
numpy==1.23.3
torch==1.12.1+cu113
torchvision==0.13.1+cu113
tqdm==4.64.1
```

## Runing command
### get the mask
```bash
cd masks
python main.py --dataset cifar10 --thre 0.98 --alpha 0.00000001 --num_steps 2 --iters 100 --op add --labeled --sum
```
This command will generate a mask file in the `masks_{dataset}` folder. The mask file is a pickle file, which contains a dictionary with parameters and the mask. The mask is a torch tensor with the same shape as the model parameters. The mask is a binary array, where 1 means the corresponding parameter is kept and 0 means the corresponding parameter is pruned.
fine-tune the model with the mask
```bash
cd ..
cd finetune
python finetune.py --dataset cifar10 --masks {mask_path} --finetune --saved_path {saved_path}
```
This command will fine-tune the model with the mask and save the model in the `saved_path` folder.