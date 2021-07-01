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
            # print(np.sum(np.multiply(roi, mask)))
            output[i - w_lim, j - h_lim] = np.abs(np.sum(np.multiply(roi, mask)))
            # print(f'roi: {roi} - in : {np.sum(np.multiply(roi, mask))} - out : {output[i-1, j-1]}')

    return output


def convolution(img, mask):
    rows, cols = img.shape
    output = np.ones((rows - 2, cols - 2), dtype=np.uint8)

    for i in range(1, rows - 1):  # range(1,2):
        print(i)
        for j in range(1, cols - 1):
            roi = img[i - 1:i + 2, j - 1:j + 2]
            # print(np.sum(np.multiply(roi, mask)))
            output[i - 1, j - 1] = np.abs(np.sum(np.multiply(roi, mask)))
            # print(f'roi: {roi} - in : {np.sum(np.multiply(roi, mask))} - out : {output[i-1, j-1]}')

    return output


image = cv2.imread('flower_input.jpg', cv2.IMREAD_GRAYSCALE)
mask_ = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
image = convolution(image, mask_)
cv2.imshow('out', image)
cv2.waitKey()

# hist = cv2.calcHist([image], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

# pic = np.copy(image)
# thresh = 150
# pic[pic < thresh] = 0  # use histogram to estimate threshold 100
# roi = pic
#
# image[image >= thresh] = 0
# background = image

# background = cv2.blur(background, (15, 15))
# background = convolution_n(background, 15)
# print(roi.shape, background.shape)

# cv2.imshow('out', cv2.add(background, roi))
# cv2.waitKey()
