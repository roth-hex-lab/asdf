import os
import json
import random
import shutil
import sys
import cv2
import numpy as np

from tqdm import tqdm

INPUT_PATH = "../bproc_output/Nanovise_from_shiyu"
OUTPUT_PATH = "../datasets/rgbd_pose/NanoVise"
TEST_PATH = "../datasets/rgbd_pose-test/NanoVise"

# Index defined in the yolo yaml-file of the basepart which is already a state.
INDEX_OF_BASEPART = 1
# Index defined in the yolo yaml-file of the start state, which is represented by one part.
INDEX_OF_STATE_0 = 9

# Whether the YOLO labels should include the states
INCLUDE_STATE_BOUNDING_BOXES = False

# Whether the images should include the depth
INCLUDE_DEPTH = True

# Number of keypoint on each object
NUMBER_OF_KEYPOINTS = 17

# Since the values get rounded down later the test set tends to be bigger than its actual percentage so higher values
# here can be chosen
TRAINING_SHARE = 0.7
VALIDATION_SHARE = 0.3


# Requirement: https://docs.ultralytics.com/datasets/classify/
def get_images_and_keypoints(path):
    """
    Traverse all folders and files in the specified path and saves png-files in a dictionary under the correct label.
    """

    images = []
    keypoints = []
    depths = []

    for item in os.listdir(path):
        full_path = os.path.join(path, item)

        images_path = os.path.join(full_path, "rgb")
        keypoints_path = os.path.join(full_path, "kps")
        depths_path = os.path.join(full_path, "depth")

        for file in os.listdir(images_path):
            image_path = os.path.join(images_path, file)
            images.append(image_path)

        for file in os.listdir(keypoints_path):
            keypoint_path = os.path.join(keypoints_path, file)
            keypoints.append(keypoint_path)

        for file in os.listdir(depths_path):
            depth_path = os.path.join(depths_path, file)
            depths.append(depth_path)

    return {"images": images, "keypoints": keypoints, "depths": depths}


def save_elements(elements, outputpath, foldername):
    print("Saving " + foldername + "-folder ...")

    out_path_images = os.path.join(outputpath, foldername, "images")
    out_path_labels = os.path.join(outputpath, foldername, "labels")

    # Check if path exists and if not create it
    if not os.path.exists(out_path_images):
        os.makedirs(out_path_images)
    if not os.path.exists(out_path_labels):
        os.makedirs(out_path_labels)

    offset = 0
    # Calculate offset to continue index of files
    if len(os.listdir(out_path_images)) > 0:
        offset = int(os.path.basename(os.listdir(out_path_images)[-1]).split(".")[0]) + 1

    # Copy all the selected elements to folder
    for idx, element in enumerate(tqdm(elements)):

        input_image_path = element[0]
        depth_image_path = element[2]

        if INCLUDE_DEPTH:
            img = cv2.imread(input_image_path)  # The channels order is BGR due to OpenCV conventions.

            # Convert the image to from 8 bits per color channel to 16 bits per color channel
            # Notes:
            # 1. We may choose not to scale by 256, the scaling is used only for viewers that expects [0, 65535] range.
            # 2. Consider that most image viewers refers the alpha (transparency) channel, so image is going to look strange.
            img = img.astype(np.uint16) * 256

            # load depth and resize
            depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Assume depth_image.png is 16 bits grayscale.

            if depth.ndim == 3:
                depth = depth[:, :, 0]  # Keep one channel if depth has 3 channels?  depth = depth[:, :, np.newaxis]

            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)  # The depth supposed to be uint16, so code should not reach here.

            # Use the depth channel as alpha channel (the channel order is BGRA - applies OpenCV conventions).
            bgrd = np.dstack((img, depth))

            # Save the data to PNG file (the pixel format of the PNG file is 16 bits RGBA).
            cv2.imwrite(os.path.join(out_path_images, str(idx + offset).zfill(6) + ".png"), bgrd)
        else:
            # Copy rgb image to output path
            shutil.copyfile(input_image_path, os.path.join(out_path_images, str(idx + offset).zfill(6) + ".png"))

        input_keypoint_path = element[1]

        # Read the labels and keypoint coordinates from the keypoint file
        with open(os.path.join(input_keypoint_path), "r") as f:
            keypoint_labels = f.read()

        # If the states should be included it is important to add the first state, which is already a part, to the labels.
        if INCLUDE_STATE_BOUNDING_BOXES:
            # Check all lines if they contain the label for the first part and if so append the start state to the labels
            keypoint_labels_lines = keypoint_labels.splitlines()

            for line in keypoint_labels_lines:
                if line[0] == str(INDEX_OF_BASEPART):
                    line_split = line.split(" ")
                    label_id = INDEX_OF_STATE_0
                    middle_x = line_split[1]
                    middle_y = line_split[2]
                    width = line_split[3]
                    height = line_split[4]
                    keypoint_labels = keypoint_labels + str(label_id) + " " + str(middle_x) + " " + str(middle_y) + " " + str(width) + " " + str(height) + NUMBER_OF_KEYPOINTS * " 0 0 0"
                    break

        with open(os.path.join(out_path_labels, str(idx + offset).zfill(6) + ".txt"), "w") as f:
            f.write(keypoint_labels)


elements = get_images_and_keypoints(INPUT_PATH)

print(elements)

if os.path.isabs(OUTPUT_PATH):
    print("Saving in: " + OUTPUT_PATH)
else:
    print("Saving in: " + os.getcwd() + "/" + OUTPUT_PATH)


# Select the right share of training and validation elements
num_elements = len(elements["images"])
num_training_elements = int(num_elements * TRAINING_SHARE)
num_validation_elements = int(num_elements * VALIDATION_SHARE)


indices = range(num_elements)
# Get random indexes from the files based on the training elements share
random_training_elements_indices = random.sample(indices, num_training_elements)

# Collect elements based on the chosen indices
train_elements = []
for elem_idx in random_training_elements_indices:
    train_elements.append([elements["images"][elem_idx], elements["keypoints"][elem_idx], elements["depths"][elem_idx]])

# Remove used indices
indices = [i for i in indices if i not in random_training_elements_indices]

# Save train elements
save_elements(train_elements, OUTPUT_PATH, "train")

# Get random indexes from the remaining files
random_val_elements_indices = random.sample(indices, num_validation_elements)

# Collect elements based on the chosen indices
val_elements = []
for elem_idx in random_val_elements_indices:
    val_elements.append([elements["images"][elem_idx], elements["keypoints"][elem_idx], elements["depths"][elem_idx]])

# Remove used indices
indices = [i for i in indices if i not in random_val_elements_indices]

# Save validation elements
save_elements(val_elements, OUTPUT_PATH, "val")

# The remaining elements are put into the test folder
test_elements = []
for elem_idx in indices:
    test_elements.append([elements["images"][elem_idx], elements["keypoints"][elem_idx], elements["depths"][elem_idx]])

# Save the remaining files in the test folder
save_elements(test_elements, TEST_PATH, "test")
