import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import cv2
from datetime import datetime
import argparse
import shutil

import numpy as np
import pandas as pd

dir_train_img = './input/stage_1_train_images'
dir_test_img = './input/stage_1_test_images'
dir_train_png = './input/stage_1_train_images_png'
dir_test_png = './input/stage_1_test_images_png'

print('read train and test files...')
train_total = tqdm(listdir(dir_train_img))
test_total = tqdm(listdir(dir_test_img))
train_files = [f for f in train_total if isfile(join(dir_train_img, f))]
test_files = [f for f in test_total if isfile(join(dir_test_img, f))]
print('complete read train and test files!')

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)-1
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def _read(path, desired_size):
    """Will be used in DataGenerator"""
    
    dcm = pydicom.dcmread(path)

    slope, intercept = dcm.RescaleSlope, dcm.RescaleIntercept
    
    try:
        img = (dcm.pixel_array * slope + intercept)
    except:
        img = np.zeros(desired_size[:2])-1
    
    if img.shape != desired_size[:2]:
        img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
    
    img = _normalize(img)
    
    # return np.stack((img,)*3, axis=-1)
    return img

def _imshow(filename):
    # print(filename)
    ds = _read(dir_train_img+"/"+filename+".dcm")
    plt.imshow(np.squeeze(ds))
    plt.show()

ii = 0
for img in tqdm(train_files):
    if ii is 5:
        break
    ds = _read(dir_train_img+"/"+img,(224, 224))
    img_name = img.split('.')[0]
    print(img_name)
    plt.imsave("./"+img_name+".png", ds, cmap='Greys')
    ii += 1

# for img in tqdm(test_files):
#     ds = _read(dir_test_img+"/"+img,(224, 224))
#     img_name = img.split('.')[0]
#     plt.imsave(dir_test_png+"/"+img_name+".png", ds, cmap='Greys')

