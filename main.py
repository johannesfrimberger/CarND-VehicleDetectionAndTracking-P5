import yaml
import argparse

import matplotlib.image as mpimg

from lib.VehicleDetection import VehicleDetection

def main():
    """
    Read settings file and
    """

    # Set parser for inputs
    parser = argparse.ArgumentParser(description="Processing input arguments")
    parser.add_argument("-s", "--settings_file", help="Set yaml settings file", required=True)
    args = parser.parse_args()

    # All config parameters are written down in separate yaml file
    with open(args.settings_file) as fi:
        settings = yaml.load(fi)

    vd = VehicleDetection(settings["Common"])

    # Load or train classifier
    vd.init_classifier(settings["Classifier"])

    #vd.load_svm_classifier()
    #vd.processImageFolder()
    #vd.processVideo()

if __name__ == "__main__":
    main()
