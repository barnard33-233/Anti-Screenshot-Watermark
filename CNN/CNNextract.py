import torchvision
import torch
import numpy as np
import cv2

model = torch.load('./pretrain/resnet18.pth')
model.cuda()
img = cv2.imread("../images/CNNAttack/200.jpg")
img = np.array(img / 255)
img = np.transpose(img, (2, 0, 1))
info = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        pred = model(torch.Tensor([img[:, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]]).cuda())
        y_pred = torch.argmax(pred, dim=1)
        info[i][j] = y_pred[0]
print(info)
