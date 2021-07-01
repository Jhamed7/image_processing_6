import cv2
import numpy as np
import os
import argparse


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--source', default='mnist.png', type=str)
    my_parser.add_argument('--save_dir', default='mnist', type=str)
    args = my_parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    image = cv2.imread(
        args.source,
        cv2.IMREAD_GRAYSCALE
    )

    rows, cols = image.shape
    buffer = []
    temp = []
    for i in range(rows):
        if np.sum(image[i,:]) > 20:
            temp.append(image[i,:])
            if i == rows-1:
                buffer.append(temp)
        elif len(temp):
                buffer.append(temp)
                temp = []

    # for k in range(len(buffer[1])):
    #     print(len(buffer[1][k]))
    # print(len(buffer))

    for j in range(len(buffer)):

        num_rows = buffer[j]
        box = np.zeros((len(num_rows), cols), dtype=np.uint8)
        # print(box.shape, len(num_rows[0]))
        for idx, row in enumerate(num_rows):
            box[idx, :] = row
        temp = []
        buffer2 = []
        for i in range(cols):
            if np.sum(box[:, i]) > 0:
                temp.append(box[:, i])
            else:
                if len(temp):
                    buffer2.append(temp)
                    temp = []

        for z in range(len(buffer2)):
            num_cols = buffer2[z]
            number = np.zeros((len(num_cols[0]), len(num_cols)), dtype=np.uint8)
            for idx, col in enumerate(num_cols):
                number[:, idx] = col
            if np.sum(number/255) > 5:
                if not os.path.exists(os.path.join(args.save_dir, f'{j//5}')):
                    path = os.path.join(args.save_dir, f'{j//5}')
                    os.mkdir(path)
                cv2.imwrite(os.path.join(args.save_dir, f'{j//5}/num_{j}_{z}.jpg'), number)
