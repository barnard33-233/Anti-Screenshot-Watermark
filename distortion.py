'''
    @file: distortion.py
    @brief: Noise Layer for the DAE
    @author: Mohan Liu
    @date: 2023/01/12
    
    @Based on MM12_PIMoG: https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw
'''

import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import image
import cv2

def perspective(embeded_image):
    pass

def Light_Distortion(c, embed_image: np.ndarray) -> np.ndarray:
    '''
    c:              type (line or point)
    embed_image:    (512, 512, 3) RGB image
    '''
    shape = embed_image.shape
    mask = np.zeros((shape))
    mask_2d = np.zeros((shape[0], shape[1])) # (512, 512) Matrix
    lmin = 0.8 + np.random.rand(1) * 0.2
    lmax = 1.2 + np.random.rand(1) * 0.2
    # info:
    # print(f'lmin={lmin}, lmax={lmax}, c={c}, shape={shape}')
    if c == 0:
        # line light source
        direction = np.random.randint(1,5)
        for i in range(shape[2]):
            mask_2d[i,:] = -((lmax-lmin) * (i - shape[0])) / (shape[0] - 1) + lmin
        # plt.imshow(mask_2d)
        # plt.show()
        if direction == 1:
            result = mask_2d
        elif direction == 2:
            result = np.rot90(mask_2d, 1)
        elif direction == 3:
            result = np.rot90(mask_2d, 1)
        elif direction == 4:
            result = np.rot90(mask_2d, 1)
        for i in range(shape[2]):
            mask[:, :, i] = result
        result = mask
    else:
        # point light source
        x = np.random.randint(0, shape[0])
        y = np.random.randint(0, shape[1])
        # print(f'info: x, y = {x}, {y}')
        max_len = np.max([
            np.sqrt(x**2 + y**2),
            np.sqrt((x - shape[0] + 1)**2 + y**2),
            np.sqrt(x**2 + (y - shape[1] + 1)**2),
            np.sqrt((x - shape[0] + 1)**2 + (y - shape[1] + 1)**2)
        ])
        for i in range(shape[0]):
            for j in range(shape[1]):
                mask[i, j, :] = np.sqrt((i - x)**2 + (j - y)**2) / max_len * (lmin - lmax) + lmax
        result = mask
    return result

def MoireGen(p_size, theta, center_x, center_y) -> np.ndarray:
    z = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            z1 = 0.5 + 0.5 * math.cos(2 * math.pi * np.sqrt((i + 1 - center_x)**2 + (j + 1 - center_y)**2))
            z2 = 0.5 + 0.5 * math.cos(math.cos(theta / 180 * math.pi) * (j + 1) + math.sin(theta / 180 * math.pi) * (i + 1))
            z[i,j] = np.min([z1,z2])
    M = (z+1)/2
    return M

def Moire_Distortion(embed_image: np.ndarray) -> np.ndarray:
    result = np.zeros((embed_image.shape))
    for i in range(3):
        theta = np.random.randint(0,180)
        center_x = np.random.rand(1)*embed_image.shape[0]
        center_y = np.random.rand(1)*embed_image.shape[1]
        moire = MoireGen(embed_image.shape[0], theta, center_x, center_y)
        result[:, :, i] = moire
    return result

def Gaussian_Distortion(embed_image: np.ndarray, sigma=1) -> np.ndarray:
    # N(0, 1)
    shape = embed_image.shape
    mean = 0
    gaussian = np.random.normal(mean, sigma, shape)
    # plt.imshow(gaussian.astype(np.uint8))
    # plt.show()
    return gaussian

def rgba2rgb(rgba: np.ndarray, background=(255,255,255)) -> np.ndarray:
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert(ch == 4)

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def distortion(img: np.ndarray, show_process = False):
    c = np.random.randint(0, 2)
    light = Light_Distortion(c, img)
    moire = Moire_Distortion(img)
    gaussian = Gaussian_Distortion(img)
    img_l = np.clip(img * light, 0, 255).astype(np.uint8)
    img_ml = (img_l * 0.85 + moire * 255 * 0.15).astype(np.uint8)
    img_mlg = (img_ml + gaussian).astype(np.uint8)
    if show_process:
        res = [img, img_l, img_ml, img_mlg]
        i = 0
        for img in res:
            i += 1
            plt.subplot(1, 4, i)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.show()
    return img_mlg

if __name__ == '__main__':
    # '''this is a demo'''
    # lena = image.imread("lena.bmp")
    # lena = rgba2rgb(lena)
    # # print(lena.shape)

    # noised_lena = distortion(lena, True)
    # plt.imshow(noised_lena)
    # plt.show()
    for k in range(200, 300):
        img = cv2.imread("./images/CNNplus/" + str(k) + ".jpg")
        noised_img = distortion(img)
        cv2.imwrite("./images/CNNAttack/" + str(k) + ".jpg", noised_img)
