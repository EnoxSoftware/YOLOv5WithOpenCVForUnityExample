# YOLOv5 classification models

https://github.com/ultralytics/yolov5

Export YOLOv5_-cls model to ONNX:

```
# @title Export to ONNX models (classification)
!python export.py --weights yolov5n-cls.pt --include onnx --imgsz 224 224 --simplify
#!python export.py --weights yolov5s-cls.pt --include onnx --imgsz 224 224 --simplify
#!python export.py --weights yolov5m-cls.pt --include onnx --imgsz 224 224 --simplify
```
