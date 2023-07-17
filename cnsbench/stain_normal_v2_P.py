import sys
print('Python version: ' + sys.version)
print('---------------------------------------------------------------------------')
# stain normalisation for datasets and creating mmseg approved dataset folder layout: stain normalisation from https://github.com/Peter554/StainTools 
import staintools
import os
import cv2
import shutil
# --        set up      -- #
METHOD = 'vahadane'  # Update this for different methods ['vahadane', 'macenko']
STANDARDIZE_BRIGHTNESS = True # Change this for no brightness standardisation
DIR_TO_START = 'C:/Users/chakr/Documents/Honours/deeplabv3+_Stuff/PreProcessing/NEW/CryoNuSeg/yolo/' #     start of path the directory to normalise (path must not include masks)
DIR_TO_SAVE = 'C:/Users/chakr/Documents/Honours/deeplabv3+_Stuff/PreProcessing/NEW/' #      where the new normalised dataset should go
DIR_NAME_SAVE = 'CryoNuSeg_norm' #    to change the name of the normalised dataset
DIR_FILE_TO_FIT = 'C:/Users/chakr/Documents/Honours/deeplabv3+_Stuff/PreProcessing/NEW/CryoNuSeg/yolo/train/Human_AdrenalGland_01.png' #  full filename + path for file to fit to

# --        Initial Fitting of file     -- #
image_1 = staintools.read_image(DIR_FILE_TO_FIT)
image_1 = staintools.LuminosityStandardizer.standardize(image_1)
normalizer = staintools.StainNormalizer(method=METHOD)
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
            image = staintools.read_image(NAMEE)
            # Standardize the brightness
            if(NAMEE == DIR_FILE_TO_FIT):
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
                cv2.imwrite(FILENAME, image_1)
            else:
                image = staintools.LuminosityStandardizer.standardize(image)
                # normalize the stain by using first image to fit
                # then normalize it to the first image
                image = normalizer.transform(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # STAIN AUGMENTER [TODO]

                # --------------NOW SAVE IMAGE TO NEW FOLDER-------#
                cv2.imwrite(FILENAME, image)
        else:
            # Move all other files that is not an image to new place
            shutil.copyfile(OLDNAME, FILENAME)