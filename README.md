Tested with python 3.9.15

# Installation
With pip
```bash
pip install -r requirements.txt
```

# Usage
Do everything (More explanation down below)
```bash
python get_monuseg.py -g -m -e tif -e png -e jpg -y --yolofy-det
```

Download MoNuSeg
```bash
#Default
python get_monuseg.py -g

#Change zip path storage location
python get_monuseg.py -g --zip-path "your_folder"

#Change output folder location
python get_monuseg.py -g --out-path "your_folder"
```

Generate binary masks
```bash
# Default (creates .tif files)
python get_monuseg.py -m

# as .tif, .png and .jpg
python get_monuseg.py -m -e tif -e png -e jpg

# Change the output path for mask data
python get_monuseg.py -m --mask-path "your_folder"
```

Convert the dataset into yolo segmentation format
```bash
# Default
python get_monuseg.py -y

# Change the output path for yolo data
# This will create the path <your_folder>/segment
python get_monuseg.py -y --yolofy-path "your_folder"
```

Convert the dataset into yolo detection format
```bash
# Default
python get_monuseg.py --yolofy-det

# Change the output path for yolo data
# This will create the path <your_folder>/detect
python get_monuseg.py --yolofy-det --yolofy-path "your_folder"
```

Check profiling statistics (not particularly relevant)
```bash
#Default (saves to "profile_stats")
python get_monuseg.py -p
python show_stats.py

#Change file name for saving statistics
python get_monuseg.py -p --profile-path "your_file_name"
python show_stats.py -p "your_file_name"
```