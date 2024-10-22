import os
import json
import random
import shutil
import sys
import cv2
import numpy as np

from tqdm import tqdm

INPUT_PATH = "../bproc_output/NanoVise_new"
OUTPUT_PATH = "../datasets/rgbd_pose/NanoVise"
TEST_PATH = "../datasets/rgbd_pose-test/NanoVise"
INCLUSION_LIST = ["scene_", "Camera_", "Images", ".png"]

# Number of keypoint on each object
NUMBER_OF_KEYPOINTS = 17

# Whether the YOLO labels should include the states
INCLUDE_STATE_BOUNDING_BOXES = True

# Whether the images should include the depth
INCLUDE_DEPTH = True

# Since the values get rounded down later the test set tends to be bigger than its actual percentage so higher values
# here can be chosen
TRAINING_SHARE = 0.7
VALIDATION_SHARE = 0.3

# Set start state id to -1 to be ignored.
label_name_to_id = {
          "NanoVise_State_0": -1,
          "NanoVise_State_1": 9,
          "NanoVise_State_2": 10,
          "NanoVise_State_3": 11,
          "NanoVise_State_4": 12,
          "NanoVise_State_5": 13,
          "NanoVise_State_6": 14,
          "NanoVise_State_7": 15,
          }


def get_keypoint_labels_from_bop(path):
    key_point_labels = []

    data_path = os.path.join(path, "3Dprint", "train_pbr")

    for item in os.listdir(data_path):
        full_path = os.path.join(data_path, item, "kps")
        for file in os.listdir(full_path):
            file_path = os.path.join(full_path, file)
            key_point_labels.append(file_path)

    return key_point_labels


# Requirement: https://docs.ultralytics.com/datasets/classify/
def traverse_folders(path, elements, state_bounding_boxes):
    """
    Traverse all folders and files in the specified path and saves png-files in a dictionary under the correct label.
    """
    for item in os.listdir(path):
        # Check whether the item is in the INCLUSION_LIST and if not continue with the next item
        in_inclusion_list = False
        for inclusion in INCLUSION_LIST:
            if inclusion in item:
                in_inclusion_list = True
                break

        if not in_inclusion_list:
            continue

        full_path = os.path.join(path, item)

        # Read state_bounding_boxes before going deeper in the folder structure
        if "Camera_" in item:
            with open(os.path.join(full_path, "state_seg_maps", "state_bounding_boxes.json"), 'r') as openfile:
                # Reading from json file
                state_bounding_boxes = json.load(openfile)

        if os.path.isdir(full_path):
            #print("Folder:", full_path)
            traverse_folders(full_path, elements, state_bounding_boxes)  # Recursive call for subfolders
        else:
            #print("File:", full_path)
            if ".png" in full_path:
                filename = os.path.basename(full_path)
                filename_without_extension = os.path.splitext(filename)[0]

                # Get the to the image corresponding depth image
                depth_path = full_path.replace("Images", "depth_images")

                # Get the to the image corresponding segmentations
                state_bounding_box = state_bounding_boxes[int(filename_without_extension)]

                # If no state bounding box is saved set the values to None
                if len(state_bounding_box) == 0:
                    print("One image has no state bounding box!")
                    label_id = None
                    middle_x = None
                    middle_y = None
                    width = None
                    height = None
                else:
                    label = state_bounding_box["label"]
                    label_id = label_name_to_id[label]
                    middle_x = state_bounding_box["middle_x_norm"]
                    middle_y = state_bounding_box["middle_y_norm"]
                    width = state_bounding_box["width_norm"]
                    height = state_bounding_box["height_norm"]

                elements.append({
                    "rgb_img_path": full_path,
                    "depth_img_path": depth_path,
                    "label_id": label_id,
                    "middle_x": middle_x,
                    "middle_y": middle_y,
                    "width": width,
                    "height": height
                })

    return elements


def save_elements(elements, outputpath, foldername):
    print("Saving " + foldername + "-folder ...")

    out_path_images = os.path.join(outputpath, foldername, "images")
    out_path_labels = os.path.join(outputpath, foldername, "labels")
    # Check if path exists and if not create it
    if not os.path.exists(out_path_images):
        os.makedirs(out_path_images)
    if not os.path.exists(out_path_labels):
        os.makedirs(out_path_labels)

    # Copy all the selected elements to folder
    for idx, element in enumerate(tqdm(elements)):

        element_base = element[0]

        img_path = element_base["rgb_img_path"]
        depth_img_path = element_base["depth_img_path"]

        if INCLUDE_DEPTH:
            img = cv2.imread(img_path)  # The channels order is BGR due to OpenCV conventions.

            # Convert the image to from 8 bits per color channel to 16 bits per color channel
            # Notes:
            # 1. We may choose not to scale by 256, the scaling is used only for viewers that expects [0, 65535] range.
            # 2. Consider that most image viewers refers the alpha (transparency) channel, so image is going to look strange.
            img = img.astype(np.uint16) * 256

            # load depth and resize
            depth = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)  # Assume depth_image.png is 16 bits grayscale.

            if depth.ndim == 3:
                depth = depth[:, :, 0]  # Keep one channel if depth has 3 channels?  depth = depth[:, :, np.newaxis]

            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)  # The depth supposed to be uint16, so code should not reach here.

            # Use the depth channel as alpha channel (the channel order is BGRA - applies OpenCV conventions).
            bgrd = np.dstack((img, depth))

            # Save the data to PNG file (the pixel format of the PNG file is 16 bits RGBA).
            cv2.imwrite(os.path.join(out_path_images, str(idx).zfill(6) + ".png"), bgrd)
        else:
            # Copy rgb image to output path
            shutil.copyfile(img_path, os.path.join(out_path_images, str(idx).zfill(6) + ".png"))

        keypoint = element[1]

        label_id = element_base["label_id"]
        middle_x = element_base["middle_x"]
        middle_y = element_base["middle_y"]
        width = element_base["width"]
        height = element_base["height"]

        # Read the labels and keypoint coordinates from the keypoint file
        with open(keypoint, "r") as f:
            keypoint_labels = f.read()

        if INCLUDE_STATE_BOUNDING_BOXES and label_id != -1 and label_id is not None:
            label_complete = keypoint_labels + str(label_id) + " " + str(middle_x) + " " + str(middle_y) + " " + str(width) + " " + str(height) + NUMBER_OF_KEYPOINTS * " 0 0 0"
        else:
            label_complete = keypoint_labels

        with open(os.path.join(out_path_labels, str(idx).zfill(6) + ".txt"), "w") as f:
            f.write(label_complete)


keypoints = get_keypoint_labels_from_bop(INPUT_PATH + "/bop_data")

elements = []

traverse_folders(INPUT_PATH, elements, [])

if len(keypoints) != len(elements):
    print("There is a different number of keypoints and images! Check data!")
    sys.exit()


if os.path.isabs(OUTPUT_PATH):
    print("Saving in: " + OUTPUT_PATH)
else:
    print("Saving in: " + os.getcwd() + "/" + OUTPUT_PATH)


# Select the right share of training and validation elements
num_elements = len(elements)
num_training_elements = int(num_elements * TRAINING_SHARE)
num_validation_elements = int(num_elements * VALIDATION_SHARE)


indices = range(num_elements)
# Get random indexes from the files based on the training elements share
random_training_elements_indices = random.sample(indices, num_training_elements)

# Collect elements based on the chosen indices
train_elements = []
for elem_idx in random_training_elements_indices:
    train_elements.append([elements[elem_idx], keypoints[elem_idx]])

# Remove used indices
indices = [i for i in indices if i not in random_training_elements_indices]

# Save train elements
save_elements(train_elements, OUTPUT_PATH, "train")

# Get random indexes from the remaining files
random_val_elements_indices = random.sample(indices, num_validation_elements)

# Collect elements based on the chosen indices
val_elements = []
for elem_idx in random_val_elements_indices:
    val_elements.append([elements[elem_idx], keypoints[elem_idx]])

# Remove used indices
indices = [i for i in indices if i not in random_val_elements_indices]

# Save validation elements
save_elements(val_elements, OUTPUT_PATH, "val")

# The remaining elements are put into the test folder
test_elements = []
for elem_idx in indices:
    test_elements.append([elements[elem_idx], keypoints[elem_idx]])

# Save the remaining files in the test folder
save_elements(test_elements, TEST_PATH, "test")
