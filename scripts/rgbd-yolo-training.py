from ultralytics import YOLO

if __name__ == '__main__':
   # Load a model
   #model = YOLO("yolov8x-pose-p6.yaml")  # build a new model from scratch
   #model = YOLO("yolov8x-pose-p6.pt")  # load a pretrained model (recommended for training)

   model = YOLO("../runs/pose/yolov8x-p6-rgbd-pose-NanoVise10/weights/last.pt", task="pose") # Continue Training, set resume in train method to True

   # Use the model
   #model.train(data="state_detection/YOLO_v8_classification/coco128.yaml", epochs=3)  # train the model
   #model.train(data="coco128.yaml", epochs=3)  # train the model
   #  https://docs.ultralytics.com/usage/cfg/#modes
   results = model.train(
      data='../datasets/rgbd_pose/NanoVise_rgbd_pose.yaml',
      imgsz=1280,# 720),
      epochs=300,
      batch=6,
      name='yolov8x-p6-rgbd-pose-NanoVise',
      device=0,
      resume=True
   )
   #metrics = model.val()  # evaluate model performance on the validation set
   # https://docs.ultralytics.com/modes/predict/
   #results = model("../test_sets/instance_segmentation/test/images/", stream=True, conf=0.3, show=True)  # predict on an image
   #for result in results:
   #   print(result.probs)
   #path = model.export(format="onnx")  # export the model to ONNX format