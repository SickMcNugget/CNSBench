Tested with python 3.10.

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
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
cd ..
```

5. Install left-over dependencies (OpenSlide, ultralytics, scikit-image)
```bash
sudo apt -y install openslide-tools #debian/ubuntu
conda install scikit-image -c conda-forge -y
pip install -r requirements.txt
```

# Usage
Each script handles a step of the process, beginning with get_datasets.py

## Datasets
To download all datasets, simply run
```bash
python get_datasets.py
```
To change the root directory for downloaded datasets, use the --dataset-root flag
```bash
python get_datasets.py --dataset-root ~/my_datasets # for example
```