import time
import numpy as np
import cv2

from ultralytics import YOLO

if __name__ == '__main__':
   #model = YOLO("yolov8x-pose-p6.pt")  # load an official model
   model = YOLO("../runs/pose/yolov8m-rgbd-pose-CornerClamp7/weights/best.pt", task="pose") # Continue Training, set resume in train method to True

   # https://docs.ultralytics.com/modes/predict/
   results = model("../datasets/rgbd_pose/val/images/000005.png", imgsz=(1280, 1280), stream=False, conf=0.3, show=False)#, classes=[9,10,11,12,13,14,15])  # predict on an image

   class_names = results[0].names
   box_classes = np.array(results[0].boxes.cls.cpu())
   box_coordinates = np.array(results[0].boxes.xyxy.cpu())

   rgbd_img = cv2.imread("../datasets/rgbd_pose/val/images/000005.png", cv2.IMREAD_UNCHANGED)

   (B, G, R, D) = cv2.split(rgbd_img)
   image = cv2.merge([B, G, R])

   for idx, box_class in enumerate(box_classes):

      color = np.random.rand(3)*65535
      box_name = class_names[box_class]

      image = cv2.rectangle(image, pt1=(int(box_coordinates[idx][0]), int(box_coordinates[idx][1])), pt2=(int(box_coordinates[idx][2]), int(box_coordinates[idx][3])), color=color, thickness=2)
      cv2.putText(image, box_name, (int(box_coordinates[idx][0]), int(box_coordinates[idx][1])-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.8, color=color, thickness=2)


   cv2.imshow("Test", image)
   cv2.waitKey()

