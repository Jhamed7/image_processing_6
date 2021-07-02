import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def convolution_n(img, scale):
    mask = np.ones((scale, scale), dtype=float) * (1 / scale ** 2)
    rows, cols = img.shape
    w, h = mask.shape
    output = np.ones((rows - (w - 1), cols - (h - 1)), dtype=np.uint8)

    w_lim = int((w - 1) / 2)
    h_lim = int((h - 1) / 2)

    for i in range(w_lim, rows - h_lim):
        # print(i)
        for j in range(h_lim, cols - h_lim):
            roi = img[i - w_lim:i + w_lim + 1, j - h_lim:j + h_lim + 1]
            if (np.sum(np.multiply(roi, mask))) < 100:
                output[i - w_lim, j - h_lim] = (np.sum(np.multiply(roi, mask)))
            else:
                output[i - w_lim, j - h_lim] = img[i, j]  # img[i - w_lim, j - h_lim]

    return output


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--source', default='flower_input.jpg', type=str)
    my_parser.add_argument('--save_dir', default='', type=str)
    my_parser.add_argument('--scale', default=15, type=int)
    args = my_parser.parse_args()

    image = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)

    result = convolution_n(image, args.scale)
    cv2.imwrite(os.path.join(args.save_dir, f'{args.source[:-4]}_blurred.jpg'), result)
    cv2.imshow('out', result)
    cv2.waitKey()
