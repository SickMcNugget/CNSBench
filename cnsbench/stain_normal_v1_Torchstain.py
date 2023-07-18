import torch
import sys
from torchvision import transforms
import torchstain
import cv2
import shutil
import os
import numpy as np
print('Python version: ' + sys.version)
print('---------------------------------------------------------------------------')

STANDARDIZE_BRIGHTNESS = True # Change this for no brightness standardisation
METHOD = 'reinhard' # [ 'macenko', 'reinhard'] # default will be macenko
DIR_TO_START = 'C:/Users/chakr/Documents/Honours/deeplabv3+_Stuff/PreProcessing/NEW/torchstain/CryoNuSeg/yolo/' #     start of path the directory to normalise (path must not include masks)
DIR_TO_SAVE = 'C:/Users/chakr/Documents/Honours/deeplabv3+_Stuff/PreProcessing/NEW/torchstain/' #      where the new normalised dataset should go
DIR_NAME_SAVE = 'CryoNuSeg_norm_reinhard' #    to change the name of the normalised dataset
DIR_FILE_TO_FIT = 'C:/Users/chakr/Documents/Honours/deeplabv3+_Stuff/PreProcessing/NEW/torchstain/CryoNuSeg/yolo/train/Human_AdrenalGland_01.png' #  full filename + path for file to fit to

# --        Initial Fitting of file     -- #
image_1 = cv2.cvtColor(cv2.imread(DIR_FILE_TO_FIT), cv2.COLOR_BGR2RGB)
if (METHOD == 'reinhard'):
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='numpy')
    print("[*] Chosen Reinhard stain normaliser")
else: 
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    print("[*] Chosen Macenko stain normaliser")
normalizer.fit(image_1)
print('[*] Normalising and fiting the chosen image')
# --   -- #
SAVE_DIR = os.path.join(DIR_TO_SAVE, DIR_NAME_SAVE)
# Create new directory
if not os.path.exists(SAVE_DIR):
    print('[*] Creating Directory: ' + DIR_NAME_SAVE)
    os.makedirs(SAVE_DIR)
i = 0
for root, subdirs, files in os.walk(DIR_TO_START, topdown=True):
    # Will get the current folder
    TEMPROOT = root
    ROOTFROMCURTEMP = TEMPROOT.replace(DIR_TO_START,'')
    DIR_NEW_NAME = SAVE_DIR + "/" + ROOTFROMCURTEMP
     # Will create the new directory
    if not os.path.exists(DIR_NEW_NAME):
        print('[*] Creating Directory: '+ ROOTFROMCURTEMP)
        os.makedirs(DIR_NEW_NAME)
    for names in files:
        FILENAME = os.path.join(DIR_NEW_NAME, names)
        OLDNAME = root + "/" + names
        if(names.endswith('.png')):
            FILENAME = os.path.join(DIR_NEW_NAME, names)
            NAMEE = root + "/" + names
            print('[*] Getting image from: ' + NAMEE)
            image = cv2.cvtColor(cv2.imread(NAMEE), cv2.COLOR_BGR2RGB)
            if(NAMEE == DIR_FILE_TO_FIT):
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
                # --------------NOW SAVE IMAGE TO NEW FOLDER-------#
                cv2.imwrite(FILENAME, image_1)
            else:
                # normalize the stain by using first image to fit
                # then normalize it to the first image
                if (METHOD == 'macenko'):
                    image, H, E = normalizer.normalize(I=image, stains=True)
                else:
                    image = normalizer.normalize(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # --------------NOW SAVE IMAGE TO NEW FOLDER-------#
                cv2.imwrite(FILENAME, image)
        else:
            # Move all other files that is not an image to new place
            shutil.copyfile(OLDNAME, FILENAME)
