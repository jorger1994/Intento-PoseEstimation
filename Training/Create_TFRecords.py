import os
import TFRecord_Functions

label_file = 'C:/Users/jorge/Documents/Python/Intento PoseEstimation2/Test Pose Labels_17Keypoints_json.json'
image_folder = 'C:/Users/jorge/Documents/Python/Intento PoseEstimation2/Images/'
record_name = 'two_images_17keypoints_trial3'
destination_dir = 'C:/Users/jorge/Documents/Python/Intento PoseEstimation2/Datasets/'

TFRecord_Functions.create_records(label_file=label_file, image_folder = image_folder, record_name = record_name, destination_dir = destination_dir)