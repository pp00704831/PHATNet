# PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing (ICCV 2025)

Pytorch Implementation of "[PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](https://arxiv.org/abs/2507.14826)" (ICCV 2025)

## Installation
```
git clone https://github.com/pp00704831/PHATNet.git
cd PHATNet
conda create -n PHATNet python=3.7
source activate PHATNet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python tqdm glog scikit-image tensorboardX thop
```

## Training (PHATNet)
Download "[NHHaze20](https://drive.google.com/drive/folders/16ykFSBYFrEfssnVxVgz-yyI4iD8cT513?usp=drive_link)" dataset. <br>
Unzip NHHaze20.zip and put it into './datasets' as: './datasets/NHHaze20'
* Run the following command for training
```
cd PHATNet 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_PHATNet_NHHaze20.py
```

## Testing (PHATNet)
Take NHHaze21 dataset for example. <br>
Download "[NHHaze21](https://drive.google.com/drive/folders/16ykFSBYFrEfssnVxVgz-yyI4iD8cT513?usp=drive_link)" dataset. <br>
Unzip NHHaze21.zip and put it into './datasets' as: './datasets/NHHaze21'
* Run the following command for testing (haze-transfer)
```
cd PHATNet 
CUDA_VISIBLE_DEVICES=0 python predict_PHATNet_NHHaze20_to_NHHaze21.py
```

**We also provide the haze-transferred results in "[Here](https://drive.google.com/drive/folders/16ykFSBYFrEfssnVxVgz-yyI4iD8cT513?usp=drive_link)", such as "PHATNet_NHHaze20_to_NHHaze21".** <br>
**Please unzip them into './dataset'**
