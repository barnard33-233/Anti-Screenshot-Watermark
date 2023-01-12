import torchvision
import torch
import numpy as np
import cv2
from collections import defaultdict
import pandas as pd

# 正确结果
Msg = 0xFF81ABA7CC6F0000
Msg = bin(Msg)[2:].zfill(64)
M = [None, Msg[:16], Msg[16:32], Msg[32:48], Msg[48:64]]
LSD = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
LSD_map = np.zeros((16, 16))
for i in range(4):
    for j in range(4):
        msg = M[LSD[i][j]]
        for x in range(4):
            for y in range(4):
                LSD_map[i * 4 + x, j * 4 + y] = msg[x * 4 + y]

model = torch.load('resnet18.pth')
model.cuda()
img_path = "./images/cut/restore_iphone/2_15.png"
print(img_path)
img = cv2.imread(img_path)
img = np.array(img / 255)
img = np.transpose(img, (2, 0, 1))
info = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        pred = model(torch.Tensor([img[:, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]]).cuda())
        y_pred = torch.argmax(pred, dim=1)
        info[i][j] = y_pred[0]
# print(info)
tmp = 0
for i in range(16):
    for j in range(16):
        if LSD_map[i][j]==info[i][j]:
            tmp += 1
acc = tmp / 256
print(img_path + "  acc: %.2f%%" % (acc * 100))
rec = [[], [], [], [], []]
for i in range(4):
    for j in range(4):
        tmp = ""
        for x in range(4):
            for y in range(4):
                tmp += str(int(info[i * 4 + x][j * 4 + y]))
        rec[LSD[i][j]].append(tmp)
all_possi = []
for l in rec[1]:
    for m in rec[2]:
        for n in rec[3]:
            for p in rec[4]:
                all_possi.append(l + m + n + p)
print(len(all_possi))
F = open(r'possi.txt','w')
for i in all_possi:
    F.write(i+'\n')
F.close()

