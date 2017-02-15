import numpy as np
import os
import glob
import cv2
import csv
from skimage.feature import hog


def one_channel_to_gray(image):
    """
    Convert single channel grayscale image to 3 channel color image
    :param image: Grayscale image with one channel
    :return: Grayscale image with three channels
    """
    return np.dstack((image, image, image))


def get_image_in_bbox(img, bbox):
    """
    Return the
    :param img:
    :param bbox:
    :return:
    """
    el1, el2 = bbox
    return img[el1[1]:el2[1], el1[0]:el2[0]]


def read_project_data(folder, label):
    """

    :param folder:
    :param label:
    :return: tuple consisting of path to image and label
    """
    data = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            all_images = glob.glob(os.path.join(root, d, "*.png"))
            for img in all_images:
                data.append((img, label))

    return data


def read_udacity_data(folder):
    """

    :param folder:
    :return:
    """
    non_vehicle = []
    vehicle = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("labels.csv"):

                with open(os.path.join(root, file), 'r') as f:

                    # Check which folder is processed and adapt data processing to this
                    crowdai_structure = ("object-detection-crowdai" in root)

                    # Skip header for object-detection-crowdai data
                    if crowdai_structure:
                        reader = csv.reader(f)
                        all_data = list(reader)
                        all_data = all_data[1:]
                    else:
                        reader = csv.reader(f, delimiter=' ')
                        all_data = list(reader)

                    for ind in range(0, len(all_data)):
                        data = all_data[ind]

                        if crowdai_structure:
                            bbox = data[0:4]
                            filename = data[4]
                            label = data[5]
                        else:
                            bbox = data[1:5]
                            filename = data[0]
                            label = data[6]

                        bbox = list(map(int, bbox))
                        bbox = (tuple(bbox[0:2]), tuple(bbox[2:4]))

                        if ("car" in label) or ("Car" in label):
                            vehicle.append((os.path.join(root, filename), 1, bbox))
                        else:
                            non_vehicle.append((os.path.join(root, filename), 0, bbox))

    return non_vehicle, vehicle


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """

    :param img:
    :param bboxes:
    :param color:
    :param thick:
    :return:
    """
    # make a copy of the image
    draw_img = np.copy(img)

    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for el1, el2 in bboxes:
        cv2.rectangle(draw_img, el1, el2, color, thick)

    return draw_img


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """

    :param img: Grayscale image
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def color_hist(img, n_bins=32, bins_range=(0, 256)):
    """

    :param img: Image with 3 color channels
    :param n_bins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=n_bins, range=(0, 360))
    channel2_hist = np.histogram(img[:, :, 1], bins=n_bins, range=(0, 1))
    channel3_hist = np.histogram(img[:, :, 2], bins=n_bins, range=(0, 1))

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def cvt_color_string_to_cv2(input_string):
    """
    Convert input color string to cv2 conversion method
    :param input_string:
    :return: cv2 conversion method
    """
    if input_string == "RGB":
        return cv2.COLOR_RGB2BGR
    elif input_string == "HLS":
        return cv2.COLOR_RGB2HLS
    else:
        return None
