# YOLOv5 models

https://github.com/ultralytics/yolov5

Export YOLOv5 model to ONNX:

```
# @title Export to ONNX models (object detection)
!python export.py --weights yolov5n.pt --include onnx --simplify
#!python export.py --weights yolov5s.pt --include onnx --simplify
#!python export.py --weights yolov5m.pt --include onnx --simplify
#!python export.py --weights yolov5l.pt --include onnx --simplify
#!python export.py --weights yolov5x.pt --include onnx --simplify

#!python export.py --weights yolov5n.pt --include onnx --imgsz 320 320 --simplify

#!python export.py --weights yolov5n.pt --include onnx # OpenCVDNN NG
#!python export.py --weights yolov5n.pt --include onnx --opset 12 # OpenCVDNN NG
#!python export.py --weights yolov5n.pt --include onnx --simplify # OpenCVDNN OK
#!python export.py --weights yolov5n.pt --include onnx --opset 12 --simplify # OpenCVDNN OK
```
