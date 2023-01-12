import cv2
import numpy as np


def dis(x, y, b):
    return np.sqrt((x - b / 2) ** 2 + (y - b / 2) ** 2)


def TERAinsert(k, msg):
    imgpath = "./images/CelebA/" + str(k) + ".jpg"
    # 读取图片并转换为numpy形式
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (512, 512))
    img = np.float32(np.array(img))

    M = [None, msg[:16], msg[16:32], msg[32:48], msg[48:64]]
    LSD = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]

    # 遵循原论文参数，以32*32为例。
    b = 32
    PB = np.zeros((b, b))
    for i in range(b):
        for j in range(b):
            PB[i, j] = 1 - dis(i, j, b) / (b / 4)
            if PB[i, j] < 0:
                PB[i, j] = 0

    P_plus = np.ones((b, b, 3))
    P_minus = np.ones((b, b, 3))
    P_plus[:, :, 0] += PB
    P_minus[:, :, 0] -= PB
    P_plus[:, :, 2] -= PB
    P_minus[:, :, 2] += PB
    B_plus = img.copy()
    B_minus = img.copy()
    # 按照消息嵌入
    for i in range(4):
        for j in range(4):
            msg = M[LSD[i][j]]
            for x in range(4):
                for y in range(4):
                    X = i * b * 4 + x * b
                    Y = j * b * 4 + y * b
                    if msg[x * 4 + y]=="0":
                        B_plus[X:X + b, Y:Y + b, :] = np.multiply(B_plus[X:X + b, Y:Y + b, :], P_minus)
                        B_minus[X:X + b, Y:Y + b, :] = np.multiply(B_minus[X:X + b, Y:Y + b, :], P_plus)
                    else:
                        B_plus[X:X + b, Y:Y + b, :] = np.multiply(B_plus[X:X + b, Y:Y + b, :], P_plus)
                        B_minus[X:X + b, Y:Y + b, :] = np.multiply(B_minus[X:X + b, Y:Y + b, :], P_minus)
    alpha = 0.1  # 嵌入强度参数
    B_plus = np.uint8(np.clip(B_plus * alpha + img * (1 - alpha), 0, 255))
    B_minus = np.uint8(np.clip(B_minus * alpha + img * (1 - alpha), 0, 255))
    cv2.imwrite("./images/CNNplus/" + str(k) + ".jpg", B_plus)
    cv2.imwrite("./images/CNNminus/" + str(k) + ".jpg", B_minus)


# msg = ""
# for i in range(64):
#     msg += "0" if np.random.rand() < 0.5 else "1"
Msg = "1100000001101000111110100000101000111101000110010000010101100001"
for k in range(200, 300):
    TERAinsert(k, Msg)
    print(str(k) + " finished!")
