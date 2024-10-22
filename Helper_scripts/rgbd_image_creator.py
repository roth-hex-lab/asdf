import numpy as np
import cv2

scale = (720, 1280)

# load image and resize
img = cv2.imread('../images/real_images/rgb/000150.png')  # The channels order is BGR due to OpenCV conventions.
#img = cv2.resize(img, scale, interpolation=cv2.INTER_LINEAR)

# Convert the image to from 8 bits per color channel to 16 bits per color channel
# Notes:
# 1. We may choose not to scale by 256, the scaling is used only for viewers that expects [0, 65535] range.
# 2. Consider that most image viewers refers the alpha (transparency) channel, so image is going to look strange.
img = img.astype(np.uint16)*256

# load depth and resize
depth = cv2.imread('../images/real_images/depth/000150.png', cv2.IMREAD_UNCHANGED)  # Assume depth_image.png is 16 bits grayscale.
#depth = cv2.resize(depth, scale, interpolation=cv2.INTER_NEAREST)

if depth.ndim == 3:
    depth = depth[:, :, 0]  # Keep one channel if depth has 3 channels?  depth = depth[:, :, np.newaxis]

if depth.dtype != np.uint16:
    depth = depth.astype(np.uint16)  # The depth supposed to be uint16, so code should not reach here.

# Use the depth channel as alpha channel (the channel order is BGRA - applies OpenCV conventions).
bgrd = np.dstack((img, depth))

print("\n\nRGBD shape")
print(bgrd.shape)  # (1216, 64, 4)

# Save the data to PNG file (the pixel format of the PNG file is 16 bits RGBA).
cv2.imwrite('../images/real_images/rgbd/rgbd.png', bgrd)