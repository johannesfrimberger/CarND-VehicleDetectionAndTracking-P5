# Import static methods in Utils.py file
from lib.Utils import *

# Image processing
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# SciKit
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Import everything needed to edit/save/watch video clips
from moviepy.editor import *

# Import common python modules
import os
import glob
import random as rng
from tqdm import tqdm
import time
import pickle

class VehicleDetection:
    """

    """

    def __init__(self, settings):
        """

        :param settings:
        """
        # Store common settings
        self.visualization = settings["Visualization"]

        # Define HOG parameters
        self.orient = None
        self.pix_per_cell = None
        self.cell_per_block = None

        # Define color hist parameters
        self.n_bins = 32
        self.bins_range = (0, 256)

        # Internal storage of prediction method and scaling
        self.prediction_class = None
        self.scaler = None

    def get_features(self, img):
        """

        :param img:
        :return:
        """

        # Cvt image to grayscale and calculate HOG features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog_features = get_hog_features(gray, orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block)

        #
        color_features = color_hist(cv2.cvtColor(img, cv2.COLOR_RGB2HLS), n_bins=self.n_bins, bins_range=self.bins_range)

        # Normalize hog and color features separately and return feature vector containing both
        return np.concatenate((hog_features, color_features))

    def pre_process_inputs(self, all_data):
        """

        :return:
        """
        features = []
        labels = []

        for el in tqdm(all_data):
            file_name = el[0]
            features.append(self.get_features(mpimg.imread(file_name)))
            labels.append(el[1])

        return np.asarray(features), np.asarray(labels)

    def __visualize_classifier(self, folder):
        """
        Visualize training set and features used for classification algorithm
        :param folder: Folder to store visualization
        """

        # Use these two images for visualization
        img_vehicle = mpimg.imread("data/ProjectData/vehicles/GTI_Left/image0034.png")
        img_non_vehicle = mpimg.imread("data/ProjectData/non-vehicles/GTI/image5.png")

        # Visualize both images
        filename = os.path.join(folder, "overview_training_images.png")
        plt.subplot(1, 2, 1)
        fig = plt.imshow(img_vehicle)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle", fontsize=12)
        plt.subplot(1, 2, 2)
        fig = plt.imshow(img_non_vehicle)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("No Vehicle", fontsize=12)
        plt.savefig(filename, bbox_inches='tight', dpi=200)

        # Plot features
        img_vehicle_gray = cv2.cvtColor(img_vehicle, cv2.COLOR_RGB2GRAY)
        img_non_vehicle_gray = cv2.cvtColor(img_non_vehicle, cv2.COLOR_RGB2GRAY)
        img_vehicle_hls = cv2.cvtColor(img_vehicle, cv2.COLOR_RGB2HLS)
        img_non_vehicle_hls = cv2.cvtColor(img_non_vehicle, cv2.COLOR_RGB2HLS)

        filename = os.path.join(folder, "overview_features.png")
        font_size = 6

        hog_features, hog_image_vehicle = get_hog_features(img_vehicle_gray, orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                           cell_per_block=self.cell_per_block, vis=True)

        fig = plt.subplot(4, 4, 1)
        plt.imshow(img_vehicle_gray, cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle Grayscale", fontsize=font_size)

        fig = plt.subplot(4, 4, 2)
        plt.imshow(hog_image_vehicle, cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle Grayscale HOG", fontsize=font_size)

        hog_features, hog_image_non_vehicle = get_hog_features(img_non_vehicle_gray, orient=self.orient,
                                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                                   vis=True)

        fig = plt.subplot(4, 4, 3)
        plt.imshow(img_non_vehicle_gray, cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle Grayscale", fontsize=font_size)

        fig = plt.subplot(4, 4, 4)
        plt.imshow(hog_image_non_vehicle, cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle Grayscale HOG", fontsize=font_size)

        fig = plt.subplot(4, 4, 5)
        plt.imshow(img_vehicle_hls[:, :, 0], cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle H-Channel", fontsize=font_size)

        fig = plt.subplot(4, 4, 6)
        plt.hist(img_vehicle_hls[:, :, 0], bins=self.n_bins)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle H-Channel Histogram", fontsize=font_size)

        fig = plt.subplot(4, 4, 7)
        plt.imshow(img_non_vehicle_hls[:, :, 0], cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle H-Channel", fontsize=font_size)

        fig = plt.subplot(4, 4, 8)
        plt.hist(img_non_vehicle_hls[:, :, 0], bins=self.n_bins)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle H-Channel Histogram", fontsize=font_size)

        fig = plt.subplot(4, 4, 9)
        plt.imshow(img_vehicle_hls[:, :, 1], cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle L-Channel", fontsize=font_size)

        fig = plt.subplot(4, 4, 10)
        plt.hist(img_vehicle_hls[:, :, 1], bins=self.n_bins)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle L-Channel Histogram", fontsize=font_size)

        fig = plt.subplot(4, 4, 11)
        plt.imshow(img_non_vehicle_hls[:, :, 1], cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle L-Channel", fontsize=font_size)

        fig = plt.subplot(4, 4, 12)
        plt.hist(img_non_vehicle_hls[:, :, 1], bins=self.n_bins)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle L-Channel Histogram", fontsize=font_size)

        fig = plt.subplot(4, 4, 13)
        plt.imshow(img_vehicle_hls[:, :, 2], cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle S-Channel", fontsize=font_size)

        fig = plt.subplot(4, 4, 14)
        plt.hist(img_vehicle_hls[:, :, 2], bins=self.n_bins)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Vehicle S-Channel Histogram", fontsize=font_size)

        fig = plt.subplot(4, 4, 15)
        plt.imshow(img_non_vehicle_hls[:, :, 2], cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle S-Channel", fontsize=font_size)

        fig = plt.subplot(4, 4, 16)
        plt.hist(img_non_vehicle_hls[:, :, 2], bins=self.n_bins)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title("Non-Vehicle S-Channel Histogram", fontsize=font_size)

        plt.savefig(filename, bbox_inches='tight', dpi=200)

    def __train_classifier(self, use_udacity_data=False):
        """

        :return:
        """

        # Read labeled dataset
        if use_udacity_data:
            pd_non_vehicle, pd_vehicle = self.read_udacity_data("data/Udacity")
        else:
            pd_non_vehicle = read_project_data(folder="data/ProjectData/non-vehicles", label=0)
            pd_vehicle = read_project_data(folder="data/ProjectData/vehicles", label=1)

        # Explore initial data_set
        n_elements_vehicle = len(pd_vehicle)
        print("Total number of vehicle samples: {}".format(n_elements_vehicle))
        n_elements_non_vehicle = len(pd_non_vehicle)
        print("Total number of non-vehicle samples: {}".format(n_elements_non_vehicle))

        # Shuffle elements
        rng.seed(10)
        rng.shuffle(pd_non_vehicle)
        rng.shuffle(pd_vehicle)

        # Determine number of elements
        n_elements = min(n_elements_vehicle, n_elements_non_vehicle)

        print("Number of samples used for training: {}".format(2*n_elements))
        all_data = np.concatenate((pd_vehicle[0:n_elements], pd_non_vehicle[0:n_elements]))
        rng.shuffle(all_data)

        # Pre-process data
        all_features, all_labels = self.pre_process_inputs(all_data)

        # Normalize features
        scaler = StandardScaler().fit(all_features)
        all_features = scaler.transform(all_features)

        # Split data into training and test set
        train_features, test_features, train_labels, test_labels = \
            train_test_split(all_features, all_labels, test_size=0.20, random_state=42)

        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(train_features, train_labels)
        t2 = time.time()
        print(t2 - t, 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Train Accuracy of SVC = ', svc.score(train_features, train_labels))
        print('Test Accuracy of SVC = ', svc.score(test_features, test_labels))

        # Check the prediction time for a single sample
        t = time.time()
        prediction = svc.predict(test_features[0].reshape(1, -1))
        t2 = time.time()
        print(t2 - t, 'Seconds to predict with SVC')

        self.prediction_class = svc
        self.scaler = scaler

    def find_vehicles(self, img):
        s = 64

        window_img = np.copy(img)
        while s >= 64:
            #print(img.shape)
            window_img = self.slide_window(img, window_img, xy_window=(s, s), y_start_stop=[300, None])
            #print(windows)
            s = np.int(s/2)
            #window_img = draw_boxes(window_img, windows, color=(0, 0, 255), thick=6)

        return window_img

    def slide_window(self, img, window_img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]

        step_x = np.int(xy_window[0] * xy_overlap[0])
        step_y = np.int(xy_window[1] * xy_overlap[1])

        # Initialize a list to append window positions to
        window_list = []

        for y_left in range(y_start_stop[0], y_start_stop[1] - xy_window[1], step_y):
            for x_left in range(x_start_stop[0], x_start_stop[1] - xy_window[0], step_x):
                # Append window position to list
                window_list.append(((x_left, y_left), (x_left + xy_window[0], y_left + xy_window[1])))
                data = img[y_left:(y_left + xy_window[1]), x_left:(x_left + xy_window[0])]

                features = np.asarray(self.get_features(cv2.resize(data, (64, 64))))
                features = self.scaler.transform(features.reshape(1, -1))
                #print(features.shape)
                pred = self.prediction_class.predict(features.reshape(1, -1))

                if int(pred[0]) > 0:
                    window_img = draw_boxes(window_img, [((x_left, y_left), (x_left + xy_window[0], y_left + xy_window[1]))], color=(0, 0, 255), thick=6)

        return window_img

    def init_classifier(self, settings):
        """

        :param settings:
        """
        # Set filename to store camera calibration information
        storage_file = os.path.join(settings["Folder"], "classifier.p")
        file_exists = os.path.isfile(storage_file)

        # Update internal settings for classifier
        self.orient = settings["Orientation"]
        self.pix_per_cell = settings["PixelPerCell"]
        self.cell_per_block = settings["CellPerBlock"]

        if self.visualization:
            print("Storing Visualization for Training Classifier")
            self.__visualize_classifier(settings["Folder"])

        # Either load existing classifier or train classifier
        if settings["UseStoredFile"] and file_exists:
            print("Using trained classifer at {}".format(storage_file))
            data = pickle.load(open(storage_file, "rb"))
            self.prediction_class = data["SVM"]
            self.scaler = data["Scaler"]
        else:
            print("Start training classifier")
            self.__train_classifier()
            storage = {
                "SVM": self.prediction_class,
                "Scaler": self.scaler
            }
            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            pickle.dump(storage, open(storage_file, "wb"))

    def process_image_folder(self, settings):
        """
        :param settings:
        :return:
        """

        # Read settings
        input_folder = "data/images"
        storage_folder = "results/images"
        pattern = "test"

        # Find all images in given folder
        allImages = glob.glob(os.path.join(input_folder, "{}*.jpg".format(pattern)))

        print("Start processing images {} in folder {} with pattern {}".format(len(allImages), input_folder, pattern))

        # Iterate over all images
        for file_name in tqdm(allImages, unit="Image"):
            output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            img = mpimg.imread(file_name)
            mpimg.imsave(output_file, self.find_vehicles(img))

    def process_videos(self):
        """

        :param settings:
        :return:
        """

        file_names = ["data/videos/short_project_video.mp4"]
        storage_folder = "results/videos"

        for file_name in file_names:
            output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            print("Start processing video {} and save it as {}".format(file_name, output_file))

            input = VideoFileClip(file_name)
            output = input.fl_image(self.find_vehicles)
            output.write_videofile(output_file, audio=False)


