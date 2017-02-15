import yaml
import argparse
from lib.VehicleDetection import VehicleDetection


def main():
    """
    Read settings file and run vehicle detection as configured in yaml file
    """

    # Set parser for inputs
    parser = argparse.ArgumentParser(description="Processing input arguments")
    parser.add_argument("-s", "--settings_file", help="Set yaml settings file", required=True)
    args = parser.parse_args()

    # All config parameters are written down in separate yaml file
    with open(args.settings_file) as fi:
        settings = yaml.load(fi)

    # Init vehicle detection class
    vd = VehicleDetection(settings["Common"])

    # Load or train classifier
    vd.init_classifier(settings["Classifier"])

    # Check if all images in a folder should be processed
    if settings["Image"]["Process"]:
        vd.process_image_folder(settings["Image"], settings["SlidingWindow"])

    # Check if a video should be processed
    if settings["Video"]["Process"]:
        vd.process_videos(settings["Video"], settings["SlidingWindow"])


if __name__ == "__main__":
    main()
