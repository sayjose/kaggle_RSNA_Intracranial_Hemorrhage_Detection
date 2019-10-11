import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

# CUDA for PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

dir_csv = './input'
dir_train_img = './input/stage_1_train_images'
dir_test_img = './input/stage_1_test_images'

# Parameters
test_nums = 1000
val_nums = 10

print('read train and test files...')
train_total = tqdm(listdir(dir_train_img))
test_total = tqdm(listdir(dir_test_img))
train_files = [f for f in train_total if isfile(join(dir_train_img, f))]
test_files = [f for f in test_total if isfile(join(dir_test_img, f))]
print('complete read train and test files!')

print('read csv...')
train_csv = pd.read_csv(dir_csv+"/stage_1_train.csv", sep='\t')
len_csv = len(train_csv)
print('complete read csv!')

def _conv_raw_to_separation(df):
    slice_df = pd.DataFrame(df["ID,Label"].str.split(',').tolist(), columns=["ID", "Label"])
    slice_df[['ID', 'Image', 'Index']] = slice_df['ID'].str.split('_', expand=True)
    slice_df = slice_df[['Image', 'Index', 'Label']]
    slice_df.drop_duplicates(inplace=True)
    slice_df = slice_df.pivot(index='Image', columns='Index', values='Label').reset_index()
    slice_df['Image'] = 'ID_' + slice_df['Image']
    # slice_df = pd.DataFrame(df["ID,Label"].str.split(',').tolist(), columns=["Image_Diagnosis", "Label"])
    # slice_df["Image"] = slice_df["Image_Diagnosis"].str.slice(stop=12)
    # slice_df["Diagnosis"] = slice_df["Image_Diagnosis"].str.slice(start=13)    
    # slice_df = slice_df.drop(["Image_Diagnosis"], axis=1)
    return slice_df

def _split_csv_to_train_nums(train_csv, i):
    return train_csv[0:i*6]

def _split_csv_to_val_nums(train_csv, i, n):
    return train_csv[i*6:(i+n)*6]


def _get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def _get_windowing(data):
    dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def _window_image(img, window_center, window_width, slope, intercept):
    img = (img * slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    return np.clip(img, img_min, img_max)

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def _read(path, desired_size=(512, 512)):
    
    dcm = pydicom.dcmread(path)

    window_params = _get_windowing(dcm) # (center, width, slope, intercept)

    try:
        # dcm.pixel_array might be corrupt (one case so far)
        img = _window_image(dcm.pixel_array, *window_params)
    except:
        img = np.zeros(desired_size)
        print('corrupt')

    img = _normalize(img)
    
    return img[:,:,np.newaxis]

def _imshow(filename):
    # print(filename)
    ds = _read(dir_train_img+"/"+filename+".dcm")
    plt.imshow(np.squeeze(ds))
    plt.show()

# img[:,:,np.newaxis]
# ds = _read(dir_train_img+"/"+train_files[10])

trn = _split_csv_to_train_nums(train_csv, test_nums)
trn = _conv_raw_to_separation(trn)
val = _split_csv_to_val_nums(train_csv, test_nums, val_nums)
val = _conv_raw_to_separation(val)


class Dataset(data_utils.Dataset):

    def __init__(self, df, typeIs):
        self.len = df.shape[0]
        self.x_data = df['Image'].values
        _y_data = df.drop(['Image'], axis=1).values.tolist()
        self.y_data = np.array(_y_data, dtype=np.float32)
        self.typeIs = typeIs


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.typeIs is 'train':
            ds = _read(dir_train_img+"/"+train_files[index])
        else:
            ds = _read(dir_train_img+"/"+train_files[index+test_nums])
        img = torch.from_numpy(np.transpose(ds, (2,0,1))).float()
        label = self.y_data[index]
        return img, label

labels = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]

#parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 10

train_dataset = Dataset(trn, 'train')
val_dataset = Dataset(val, 'val')
trainloader = data_utils.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
valloader = data_utils.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Linear(30*30*64, 6, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN().to(device)

criterion = nn.MultiLabelSoftMarginLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(trainloader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in trainloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {}] cost = {}'.format(epoch+1, avg_cost))

print('Learning Finished!')


with torch.no_grad():
    for X, Y in valloader:
        X_test = X.to(device)
        Y_test = Y.to(device)

        prediction = model(X_test)
        print(prediction)
    