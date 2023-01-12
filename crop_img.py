#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import argparse


def intersect(a0, a1, b0, b1):
    a0 = np.array(a0)
    a1 = np.array(a1)
    b0 = np.array(b0)
    b1 = np.array(b1)

    left = np.array([a0 - a1, b1 - b0]).T
    right = b1 - a1
    t, s = np.linalg.solve(left, right)

    return t * a0 + (1 - t) * a1


# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_AREA)

    return warped


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+',
        help='Path of a file or a folder of files.')
    parser.add_argument('--corner', nargs=8, type=int,
        help='sum the integers (default: find the max)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    for p in args.path:
        if not os.path.isfile(p):
            raise OSError("File not found: %s" % p)

    return args


def main():
    img_path = './Lenna.jpg'
    corner = None
    quiet = False
    img = cv2.imread(img_path)

    save_pts = []

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        if event.dblclick:
            save_pts.append((event.xdata, event.ydata))
            if len(save_pts)==4:
                plt.close()

    if corner is not None:
        save_pts = np.array(corner).reshape(-1, 2)
    else:
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    print(save_pts)
    pts = order_points(np.array(save_pts))
    x1, y1 = intersect(pts[0, :], pts[1, :], pts[2, :], pts[3, :])
    x2, y2 = intersect(pts[0, :], pts[3, :], pts[1, :], pts[2, :])
    # print(x1, y1)
    # print(x2, y2)
    diag_len = np.hypot(img.shape[0], img.shape[1])
    angle_x = np.arctan(diag_len / (x1 - img.shape[1] / 2))
    angle_y = np.arctan(diag_len / (y2 - img.shape[0] / 2))

    select_area = PolyArea(pts[:, 0], pts[:, 1])
    raw_area = img.shape[0] * img.shape[1]

    if not quiet:
        print("H:", np.rad2deg(angle_x))
        print("V:", np.rad2deg(angle_y))
        print("RATIO:", select_area / raw_area)
        for i in range(4):
            print(int(pts[i, 0]), end=" ", file=sys.stderr)
            print(int(pts[i, 1]), end=" ", file=sys.stderr)
        print(file=sys.stderr)

    fname, ext = os.path.splitext(img_path)
    new_path = fname + "_correct" + ".png"
    corrected = four_point_transform(img, pts)
    corrected = cv2.resize(corrected, (512, 512))
    cv2.imwrite(new_path, corrected)
    if not quiet:
        plt.imshow(corrected)
        plt.show()


if __name__=="__main__":
    main()
