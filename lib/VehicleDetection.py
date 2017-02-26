# Import static methods in Utils.py file
from lib.Utils import *
from lib.Features import get_spatial_features, get_hog_features, get_color_hist_features

# Image processing
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# SciKit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from scipy.ndimage.measurements import label

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

    ##############################################################################################################
    # Init Methods
    ##############################################################################################################

    def __init__(self, settings, settings_sliding_window):
        """

        :param settings:
        """
        # Store common settings
        self.visualization = settings["Visualization"]
        self.color_space = settings["ColorSpace"]

        # Define HOG parameters
        self.orient = None
        self.pix_per_cell = None
        self.cell_per_block = None
        self.hog_channel = None
        self.spatial_size = None

        # Define color hist parameters
        self.n_bins = 16

        # Internal storage of prediction method and scaling
        self.prediction_class = None
        self.scaler = None
        self.heatmap = None

        # Update sliding window settings
        self.__update_sliding_window_settings(settings_sliding_window)

    def __update_sliding_window_settings(self, settings):
        """

        :param settings:
        """
        data = eval(settings["xyWindow"])
        if isinstance(data[0], int):
            data = [data]
        self.xy_window = data
        self.xy_overlap = eval(settings["xyOverlap"])

    ##############################################################################################################
    # Wrapper Methods for feature extraction
    ##############################################################################################################

    def __get_hog_features(self, img, feature_vec=True):

        if self.hog_channel < 0:
            ch0 = get_hog_features(img[:, :, 0], orient=self.orient,
                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                   feature_vec=feature_vec)
            ch1 = get_hog_features(img[:, :, 1], orient=self.orient,
                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                   feature_vec=feature_vec)
            ch2 = get_hog_features(img[:, :, 2], orient=self.orient,
                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                   feature_vec=feature_vec)

            return np.concatenate((ch0, ch1, ch2))
        else:
            ch = get_hog_features(img[:, :, self.hog_channel], orient=self.orient,
                                  pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                  feature_vec=feature_vec)
            return ch

    def __get_color_hist_features(self, img):
        """

        :param img:
        :return:
        """
        return get_color_hist_features(img, n_bins=self.n_bins)

    ##############################################################################################################
    # Private Methods
    ##############################################################################################################

    def __get_features(self, img):
        """

        :param img:
        :return:
        """
        file_features = []

        # Convert image to requested color space
        converted_img = cv2.cvtColor(img, cvt_color_string_to_cv2(self.color_space))

        # Extract spatial features
        file_features.append(get_spatial_features(converted_img, self.spatial_size))

        # Extract HOG features
        file_features.append(self.__get_hog_features(converted_img))

        # Get histogram of gradients color features
        file_features.append(self.__get_color_hist_features(converted_img))

        # Return feature vector containing both
        return np.concatenate(file_features)

    def __pre_process_inputs(self, all_data):
        """
        Read all input data
        :param all_data: Input tuple of
        :return: Tuple of features and corresponding labels
        """
        features = []

        for el in tqdm(all_data):
            file_name = el[0]
            features.append(self.__get_features(mpimg.imread(file_name)))

        return np.asarray(features)

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
        img_vehicle_hls = cv2.cvtColor(img_vehicle, cvt_color_string_to_cv2(self.color_space))
        img_non_vehicle_hls = cv2.cvtColor(img_non_vehicle, cvt_color_string_to_cv2(self.color_space))

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

    def __train_classifier(self):
        """

        """

        # Read labeled dataset
        if False:
            pd_non_vehicle, pd_vehicle = read_udacity_data("data/Udacity")
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
        n_elements = 1000
        random_idxs = np.random.randint(0, n_elements_vehicle, n_elements)

        cars = np.array(pd_vehicle)#[random_idxs]
        non_cars = np.array(pd_non_vehicle)#[random_idxs]

        #
        car_features = self.pre_process_inputs(cars)
        non_car_features = self.pre_process_inputs(non_cars)

        #
        X = np.vstack((car_features, non_car_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)

        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

        train_features, test_features, train_labels, test_labels = \
            train_test_split(scaled_X, y, test_size=0.1, random_state=42)

        svc = LinearSVC()

        # Check the training time for the SVC
        t = time.time()
        svc.fit(train_features, train_labels)
        t2 = time.time()
        print(t2 - t, 'Seconds to train SVC...')

        print('Test Accuracy of SVC = ', svc.score(test_features, test_labels))

        self.prediction_class = svc
        self.scaler = X_scaler

    def __detect_vehicles_frame(self, img, draw_img):
        """
        Run vehicle detection algorithm on single input image/video frame
        :param img: Image that should be processed in float
        :param draw_img: Image used to annotate
        :return: Annotated image and heatmap
        """
        y_start = 400
        y_stop = 656
        scale = 1.5

        img_to_search = img[y_start:y_stop, :, :]

        heatmap = np.zeros_like(img[:, :, 0])

        converted_img = cv2.cvtColor(img_to_search, cvt_color_string_to_cv2(self.color_space))

        if scale != 1:
            imshape = converted_img.shape
            converted_img = cv2.resize(converted_img, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch0 = converted_img[:, :, 0]
        ch1 = converted_img[:, :, 1]
        ch2 = converted_img[:, :, 2]

        # Define blocks
        nxblocks = (ch0.shape[1] // self.pix_per_cell) - 1
        nyblocks = (ch0.shape[0] // self.pix_per_cell) - 1

        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - 1
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog0 = get_hog_features(ch0, orient=self.orient,
                                pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                feature_vec=False)
        hog1 = get_hog_features(ch1, orient=self.orient,
                                pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                feature_vec=False)
        hog2 = get_hog_features(ch2, orient=self.orient,
                                pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                hog_feat0 = hog0[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                sub_img = cv2.resize(converted_img[ytop:ytop + window, xleft:xleft + window], (64, 64))

                spatial_features = get_spatial_features(sub_img, self.spatial_size)
                hist_features = self.__get_color_hist_features(sub_img)

                features = self.scaler.transform(
                    np.hstack((spatial_features, hog_features, hist_features)).reshape(1, -1))
                prediction = self.prediction_class.predict(features)

                if prediction == 1:
                    x_box_left = np.int(xleft * scale)
                    y_top_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    heatmap[y_top_draw + y_start:y_top_draw + y_start + win_draw, x_box_left:x_box_left + win_draw] += 1

        heatmap = apply_threshold(heatmap, 0)
        labels = label(heatmap)

        result_img = draw_labeled_box(draw_img, labels)

        return result_img, heatmap

    def __detect_vehicles_video(self, img):
        """
        Wrapper class for vehicle detection and tracking in video frames
        :param img: Input image as uint8
        :return: Annotated image
        """
        img_float = img.astype(np.float32) / 255
        result_img, heatmap = self.__detect_vehicles_frame(img_float, img)
        return result_img

    ##############################################################################################################
    # Public Methods
    ##############################################################################################################

    def init_classifier(self, settings):
        """

        :param settings:
        """
        # Set filename to store camera calibration information
        storage_file = os.path.join(settings["Folder"], "classifier_{}.p".format(self.color_space))
        file_exists = os.path.isfile(storage_file)

        # Update internal settings for classifier
        self.orient = settings["Orientation"]
        self.pix_per_cell = settings["PixelPerCell"]
        self.cell_per_block = settings["CellPerBlock"]
        self.hog_channel = settings["HogChannel"]
        self.spatial_size = eval(settings["SpatialSize"])

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
                "ColorSpace": self.color_space,
                "SVM": self.prediction_class,
                "Scaler": self.scaler
            }
            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            pickle.dump(storage, open(storage_file, "wb"))

            if self.visualization:
                print("Storing Visualization for Training Classifier")
                self.__visualize_classifier(settings["Folder"])

    def process_image_folder(self, settings):
        """
        Read video settings and detect vehicles in the images specified in the config file
        :param settings: Dictionary containing settings from "Image" section in yaml file
        """
        # Check if all images in a folder should be processed
        if settings["Process"]:
            # Read settings for image processing
            input_folder = settings["InputFolder"]
            storage_folder = settings["StorageFolder"]
            pattern = settings["Pattern"]

            # Find all images in given folder
            all_images = glob.glob(os.path.join(input_folder, "{}*.jpg".format(pattern)))
            print("Start processing images {} in folder {} with pattern {}".format(len(all_images), input_folder, pattern))

            result_imgs = []
            titles = []

            # Iterate over all images and detect vehicles
            for file_name in tqdm(all_images, unit="Image"):

                # Read image and detect vehicles for this still image
                img = mpimg.imread(file_name)
                img_float = img.astype(np.float32) / 255
                result_img, heatmap = self.__detect_vehicles_frame(img_float, img)

                # Determine output filename and store results as an image
                output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                mpimg.imsave(output_file, result_img)

                # Store data for visualization
                result_imgs.append(result_img)
                result_imgs.append(heatmap)
                titles.append('Annotated image for {}'.format(os.path.basename(file_name)))
                titles.append('Heatmap for {}'.format(os.path.basename(file_name)))

            if self.visualization:
                file_name = os.path.join(storage_folder, "overview_still_images.png")
                plot_images(file_name, result_imgs, titles, cols=2)

    def process_videos(self, settings):
        """
        Read video settings and detect vehicles in the video files specified in the config file
        :param settings: Dictionary containing settings from "Video" section in yaml file
        """
        # Check if a video should be processed
        if settings["Process"]:
            # Read settings for video processing
            file_names = settings["InputFile"]
            storage_folder = settings["StorageFolder"]

            for file_name in file_names:
                # Determine output filename
                output_file = os.path.join(storage_folder, "proc_" + os.path.basename(file_name))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Process video
                print("Start processing video {} and save it as {}".format(file_name, output_file))
                input = VideoFileClip(file_name)
                output = input.fl_image(self.__detect_vehicles_video)
                output.write_videofile(output_file, audio=False)
