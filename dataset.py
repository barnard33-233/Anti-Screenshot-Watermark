import cv2
import numpy as np
import os


# Dataset用于SU-net训练
class Dataset():
    def __init__(self, type="plus", attack=0, nums=100):
        self.imgpath = []
        for k in range(nums):
            if attack:
                img = "./images/" + type + "Attack/" + str(k) + type + ".jpg"
            else:
                img = "./images/" + type + "CelebA/" + str(k) + type + ".jpg"
            self.imgpath.append(img)

    def __getitem__(self, index):
        if type(index)==list:
            imgs = []
            for i in index:
                path = self.imgpath[i]
                img = cv2.imread(path)
                img = np.array(img)
                imgs.append(img / 255)
            return np.array(imgs)
        elif type(index)==int:
            path = self.imgpath[index]
            img = cv2.imread(path)
            img = np.array(img / 255)
            return img


# Block32_32用于CNN二分类器训练
class Block32_32():
    def __init__(self, Msg="", path="./images/CNNtrain/"):
        assert len(Msg)==64, "嵌入信息长度错误！"
        M = [None, Msg[:16], Msg[16:32], Msg[32:48], Msg[48:64]]
        LSD = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
        LSD_map = np.zeros((16, 16))
        for i in range(4):
            for j in range(4):
                msg = M[LSD[i][j]]
                for x in range(4):
                    for y in range(4):
                        LSD_map[i * 4 + x, j * 4 + y] = msg[x * 4 + y]
        LSD_map = np.uint8(LSD_map)
        self.images = []
        self.labels = []
        files = os.listdir(path)
        for file in files:
            imgpath = os.path.join(path, file)
            img = cv2.imread(imgpath)
            img = np.array(img / 255)
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            for i in range(16):
                for j in range(16):
                    self.images.append(img[:, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)])
                    self.labels.append(LSD_map[i][j])
        print(len(self.images))

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
