import cv2
import os
import numpy as np
import shutil
import imageio
import re

from glob import glob
from tqdm import tqdm


def _apply_random_brightness(img_rgb, brightness_range):
    factor = np.random.uniform(brightness_range[0], brightness_range[1])

    if abs(factor - 1.0) < 0.09:
        return img_rgb

    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)

    y = img_yuv[:, :, 0]
    y = np.where(y * factor <= 255, y * factor, 255)

    adjusted_img = np.copy(img_yuv)
    adjusted_img[:, :, 0] = y
    adjusted_img = cv2.cvtColor(adjusted_img, cv2.COLOR_YUV2RGB)

    return adjusted_img


def augment_dataset(dataset_folder, output_folder, random_brightness=(0.7, 1.3)):

    print('Augmenting Dataset in {}, result will be saved into: {}'.format(dataset_folder, output_folder))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    output_folder_img = os.path.join(output_folder, 'image_2')
    output_folder_img_labels = os.path.join(output_folder, 'gt_image_2')

    os.makedirs(output_folder_img)
    os.makedirs(output_folder_img_labels)

    image_paths = glob(os.path.join(dataset_folder, 'image_2', '*.png'))

    image_labels = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(dataset_folder, 'gt_image_2', '*_road_*.png'))
    }

    for image_file in tqdm(image_paths, desc='Augmenting Dataset', unit='images'):
        img_file_name = os.path.basename(image_file)
        img_label_file = image_labels[img_file_name]
        img_label_file_name = os.path.basename(img_label_file)

        img = imageio.imread(image_file)
        img_label = imageio.imread(img_label_file)

        img_flipped = np.fliplr(img)
        img_label_flipped = np.fliplr(img_label)

        if random_brightness:
            img = _apply_random_brightness(img, random_brightness)
            img_flipped = _apply_random_brightness(img_flipped, random_brightness)

        imageio.imsave(os.path.join(output_folder_img, img_file_name), img)
        imageio.imsave(os.path.join(output_folder_img_labels, img_label_file_name), img_label)

        imageio.imsave(os.path.join(output_folder_img, 'flipped_' + img_file_name), img_flipped)
        imageio.imsave(os.path.join(output_folder_img_labels, 'flipped_' + img_label_file_name), img_label_flipped)