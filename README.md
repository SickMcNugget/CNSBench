# Semantic Segmentation for Improved Cell Nuclei Analysis

Tested with python 3.10, Ubuntu 22.04.2 LTS

# Installation
1. Some form of conda is required for this project.
    - I recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    - The larger [Anaconda](https://www.anaconda.com/download) is also an option
2. Create a conda environment.
```bash
conda create --name cnsbench python=3.10 -y
conda activate cnsbench
```
3. Install [Pytorch](https://pytorch.org/get-started/locally/) according to the instructions.
```bash
# As of 12 Jul 2023
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```
4. Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#customize-installation) with the following instructions.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
```

5. Install left-over dependencies (OpenSlide, ultralytics, scikit-image)
```bash
sudo apt -y install openslide-tools #debian/ubuntu
conda install scikit-image -c conda-forge -y
pip install -r requirements.txt
```

# Usage
Each script handles a step of the process, beginning with get_datasets.py

## Command line arguments meaning
- dataset: The name of the dataset to train on
- normalised: Whether stain normalised version should be used
- batch: The batch size
- lr: The learning rate
- epochs: The number of epochs for training
- iterations: The number of iterations (total batches) for training
- cache: Allows for disabling caching
- model: Choose either unet or deeplabv3plus
- dataset-root: The base directory for downloaded datasets
- *-interval: the number of iterations before something happens
- wandb: Allows training results to be logged to Weights and Biases

## Datasets
To download all datasets, simply run
```bash
python get_datasets.py
```
To change the root directory for downloaded datasets, use the --dataset-root flag
```bash
python get_datasets.py --dataset-root ~/my_datasets 
```

## Training

There are training scripts for both YOLOv8 and the MMSegmentation models.
To run the YOLOv8 training script on MoNuSeg:
```bash
python train_yolo.py --dataset MoNuSeg
```
To run the YOLOv8 training script on Normalised MoNuSeg images:
```bash
python train_yolo.py --dataset MoNuSeg --normalised
```
To change some hyperparameters (note that --cache DISABLES caching):
```bash
python train_yolo.py --dataset MoNuSeg --batch 16 --lr 0.001 --epochs 200 --cache
```
  
To run the MMSegmentation training script:
```bash
python train.py --dataset MoNuSeg --model deeplabv3plus --normalised
```
Some other settings
```bash
python train.py --dataset MoNuSAC --model unet --wandb --iterations 25000 --val-interval 10000 --log-interval 10 --checkpoint-interval 5000
```

## Visualising
For YOLOv8:
```bash
python visualise_yolo.py --dataset MoNuSeg --model <your_model>.pt
```
Additional options
```bash
python visualise_yolo.py --dataset MoNuSeg --model <your_model>.pt --max-det 1000 --normalised --binary --outline
```
  
For MMSegmentation
```bash
python visualise.py --dataset CryoNuSeg --config <your_config>.py --checkpoint <your ckpt>.pth --binary --outline --normalised
```

## Exporting predictions
For YOLOv8:
```bash
python export_yolo.py --model <yolo_model>.pt --dataset TNBC --normalised
```
For MMSegmentation:
```bash
python export.py --checkpoint <your_ckpt>.pth --dataset TNBC --normalised
```

## Evaluation
```bash
python evaluate.py --dataset MoNuSeg --compare-root <path_to_model_predictions>
```

## A full example for TNBC (stain normalised)
```bash
python train.py --dataset TNBC --model unet --normalised
python export.py --checkpoint "work_dirs/TNBC/unet/<date>/iter_20000*.pth" --dataset TNBC --normalised
python evaluate.py --dataset TNBC --compare-root work_dirs/export/stainnorm/TNBC/unet
```

# To see High Resolution Images from Dicta 2023 Conference Paper *Semantic Segmentation for Improved Cell Nuclei Analysis*
![Semantic Segmentation for Improved Cell Nuclei Analysis](/../images/images/dicta_figures.md)