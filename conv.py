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

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
from torchvision import models

from efficientnet_pytorch import EfficientNet


parser = argparse.ArgumentParser(description='RSNA Intracranial Hemorrhage Detection by PyTorch')
parser.add_argument('--resume', help='Resume Training')
parser.add_argument('--flush', type=str, default='false', help='Flush Log Files')
args = parser.parse_args()

isResume = args.resume
isFlush = args.flush

print(isResume)


if isFlush=='all':
    os.system('rm -rf rsna_log')
    os.system('mkdir rsna_log')
    print('remove all log')
elif isFlush=='false':
    print('not flush log')
else:
    # os.system('rm ./rsna_log/'+isFlush)
    print('remove {}'.format(isFlush))
    
# CUDA for PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

summary = SummaryWriter('./rsna_log')

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

now = datetime.now().timestamp()
model_path = './rsna_model/rsna_model_{}.pth'.format(now)

dir_csv = './input'
# dir_train_img = './input/stage_1_train_images'
# dir_test_img = './input/stage_1_test_images'
dir_train_png = './input/stage_1_train_images_png'
dir_test_png = './input/stage_1_test_images_png'

labels = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


print('read train and test files...')
train_total = tqdm(listdir(dir_train_png))
test_total = tqdm(listdir(dir_test_png))
train_files = [f for f in train_total if isfile(join(dir_train_png, f))]
test_files = [f for f in test_total if isfile(join(dir_test_png, f))]
print('complete read train and test files!')

print('read csv...')
train_csv = pd.read_csv(dir_csv+"/stage_1_train.csv", sep='\t')
submission_csv = pd.read_csv(dir_csv+"/stage_1_sample_submission.csv", sep='\t')
print('complete read csv!')

len_train_totl = len(train_total)
len_test_totl = len(test_total)

test_nums = len_train_totl
val_nums = len_test_totl
# test_nums = 100
# val_nums = 100

def _conv_raw_to_separation(df):
    slice_df = pd.DataFrame(df["ID,Label"].str.split(',').tolist(), columns=["ID", "Label"])
    slice_df[['ID', 'Image', 'Index']] = slice_df['ID'].str.split('_', expand=True)
    slice_df = slice_df[['Image', 'Index', 'Label']]
    slice_df.drop_duplicates(inplace=True)
    slice_df = slice_df.pivot(index='Image', columns='Index', values='Label').reset_index()
    slice_df['Image'] = 'ID_' + slice_df['Image']
    return slice_df

def _split_csv_to_train_nums(train_csv, i):
    return train_csv[0:i*6]

def _split_csv_to_val_nums(test_csv, i):
    return test_csv[0:i*6]


trn = _split_csv_to_train_nums(train_csv, test_nums)
trn = _conv_raw_to_separation(trn)
# trn = _conv_raw_to_separation(train_csv)

val = _split_csv_to_val_nums(submission_csv, val_nums)
val = _conv_raw_to_separation(val)
# val = _conv_raw_to_separation(submission_csv)

undersample_seed = 0
print(trn["any"].value_counts())

ill_num = trn[trn['any']=="1"].shape[0]
print(ill_num)

norm_num = trn[trn['any']=="0"].index.values
norm_num_selection = np.random.RandomState(undersample_seed).choice(
    norm_num, size = ill_num, replace = False
)
print(len(norm_num_selection))

sick_num = trn[trn['any']=="1"].index.values
selected_num = list(set(norm_num_selection).union(set(sick_num)))
print(len(selected_num)/2)

new_trn = trn.loc[selected_num].copy()
print(new_trn['any'].value_counts())


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
            img = plt.imread(dir_train_png+"/"+train_files[index])
        else:
            img = plt.imread(dir_test_png+"/"+train_files[index])
        # img = torch.from_numpy(np.transpose(ds, (2,0,1))).float()
        label = self.y_data[index]
        return img, label

#parameters
learning_rate = 2e-5
training_epochs = 100
batch_size = 10
n_classes = 6

train_dataset = Dataset(new_trn, 'train')
val_dataset = Dataset(val, 'val')
trainloader = data_utils.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
valloader = data_utils.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)


# class CNN(nn.Module):

#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=8, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )    

#         self.fc = nn.Linear(30*30*64, 6, bias=True)
#         torch.nn.init.xavier_uniform_(self.fc.weight)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
        
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         # out = torch.sigmoid(out)
#         return out

# use model renet18
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 6)


# use model EfficientNet-B0
model = EfficientNet.from_pretrained('efficientnet-b5')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 6)


model = model.to(device)

# criterion = nn.MSELoss().to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(trainloader)

for epoch in range(training_epochs):
    avg_cost = 0
    iter_num = 0

    for X, Y in tqdm(trainloader):
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        iter_num += batch_size
        if iter_num % 100 == 0: # 매 100 iteration마다
            summary.add_scalar('cost', cost.item(), iter_num)

    print('[Epoch: {}] cost = {}'.format(epoch+1, avg_cost))

print('Learning Finished!')

torch.save(model, model_path)
print('Save Model!')


# Inference

for param in model.parameters():
    param.requires_grad = False

model.eval()
test_pred = np.zeros((len(val_dataset) * 6, 1))


with torch.no_grad():

    i = 0
    for X, Y in tqdm(valloader):
        X_test = X.to(device)
        Y_test = Y.to(device)

        pred = model(X_test)
        # val_pred = torch.sigmoid(pred)
        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(pred).detach().cpu().reshape((len(X_test) * n_classes, 1))
        i += 1

submission = pd.read_csv(dir_csv+"/stage_1_sample_submission.csv", sep='\t')
submission_df = pd.DataFrame(submission["ID,Label"].str.split(',').tolist(), columns=["ID", "Label"])
submission_df = pd.concat([submission_df.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission_df.columns = ['ID', 'Label']
submission_df.to_csv('submission.csv', index=False)