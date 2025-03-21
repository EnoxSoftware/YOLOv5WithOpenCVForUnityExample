# YOLOv5 instance segmentation models

https://github.com/ultralytics/yolov5

Export YOLOv5_-seg model to ONNX:

```
# @title Export to ONNX models (instance segmentation)
!python export.py --weights yolov5n-seg.pt --include onnx --simplify
#!python export.py --weights yolov5s-seg.pt --include onnx --simplify
#!python export.py --weights yolov5m-seg.pt --include onnx  --simplify
```
