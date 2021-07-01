import cv2
import numpy as np
import argparse


def show_pic(img):
    cv2.imshow('pic', img)
    cv2.waitKey()


def median(img, scale):
    rows, cols = img.shape
    output = np.ones((rows - (scale - 1), cols - (scale - 1)), dtype=np.uint8)

    w_lim = int((scale - 1) / 2)
    h_lim = int((scale - 1) / 2)

    for i in range(w_lim, rows - w_lim):
        for j in range(h_lim, cols - h_lim):
            roi = img[i - w_lim:i + w_lim + 1, j - h_lim:j + h_lim + 1]
            output[i - w_lim, j - h_lim] = np.median(roi)

    return output


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--source', default='board.tif', type=str)
    my_parser.add_argument('--kernel', default=3, type=int)
    args = my_parser.parse_args()

    image = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)

    # image_filter = cv2.medianBlur(image, 3)
    image_filter = median(image, args.kernel)

    show_pic(image_filter)
