import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

import math
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('using device : ' + device.type)

data_list = glob.glob('datas/img_align_celeba/*.jpg')

# PSNR function: 모델의 출력값과 high-resoultion의 유사도를 측정합니다.
# PSNR 값이 클수록 좋습니다.

# train 함수
def train_step(model, data_dl):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    for i, (image, label) in enumerate(data_dl):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_psnr = psnr(label, outputs)
        running_psnr += batch_psnr

    final_loss = running_loss / len(data_dl.dataset)
    final_psnr = running_psnr / int(len(train_ds) / data_dl.batch_size)
    return final_loss, final_psnr

def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: # label과 output이 완전히 일치하는 경우
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # padding_mode='replicate'는 zero padding이 아닌, 주변 값을 복사해서 padding 합니다.
        self.conv1 = nn.Conv2d(3, 64, 9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, 1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class SRdataset(Dataset):

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(40, 40), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        inp = cv2.GaussianBlur(img,(0,0),1)
        img = np.transpose(img, (2,0,1))
        inp = np.transpose(inp, (2,0,1))

        input_sample, label_sample = torch.tensor(inp, dtype=torch.float32), torch.tensor(img, dtype=torch.float32)

        return input_sample,label_sample

train_ds = SRdataset(data_list)
train_dl = DataLoader(train_ds, batch_size=32)

for image, label in train_dl:
  img = image[0]
  lab = label[0]
  break

plt.subplot(1,2,1)
plt.imshow(np.transpose(img, (1,2,0)))
plt.subplot(1,2,2)
plt.imshow(np.transpose(lab, (1,2,0)))

# 가중치 초기화
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

model = SRCNN().to(device)
model.apply(initialize_weights)

# 손실함수
loss_func = nn.MSELoss()

optimizer = optim.Adam(model.parameters())

num_epochs = 10

train_loss = []
train_psnr = []
start = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1} of {num_epochs}')
    train_epoch_loss, train_epoch_psnr = train_step(model, train_dl)

    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    end = time.time()
    print(f'Train PSNR: {train_epoch_psnr:.3f}, Time: {end-start:.2f} sec')

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# psnr plots
plt.figure(figsize=(10, 7))
plt.plot(train_psnr, color='green', label='train PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()

for img, label in train_dl:
    img = img[0]
    label = label[0]
    break

# super-resolution
model.eval()
with torch.no_grad():
    img_ = img.unsqueeze(0)
    img_ = img_.to(device)
    output = model(img_)
    output = output.squeeze(0)


plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.imshow(np.transpose(img, (1,2,0)))
plt.title('input')
plt.subplot(1,3,2)
plt.imshow(np.transpose(output.cpu(), (1,2,0)))
plt.title('output')
plt.subplot(1,3,3)
plt.imshow(np.transpose(label, (1,2,0)))
plt.title('origingal')