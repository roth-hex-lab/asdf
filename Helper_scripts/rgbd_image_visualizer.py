import numpy as np
import cv2


bgrd = cv2.imread('../datasets/rgbd_pose/NanoVise/train/images/011409.png', cv2.IMREAD_UNCHANGED)

#bgrd2 = cv2.imread('../datasets/rgbd_pose/val/images/000010.png', cv2.IMREAD_UNCHANGED)

(B, G, R, D) = cv2.split(bgrd)
image = cv2.merge([B,G,R])

cv2.imshow("Test", image)
cv2.waitKey(0)

cv2.imshow("Depth", D*10)
cv2.waitKey(0)
