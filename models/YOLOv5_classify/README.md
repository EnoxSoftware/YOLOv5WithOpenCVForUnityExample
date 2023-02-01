# YOLOv5 v7.0 classification models

https://github.com/ultralytics/yolov5

Export YOLOv5_-cls model to ONNX:

```
!python export.py --weights yolov5n-cls.pt --include onnx --imgsz 224 224 --opset 12
!python export.py --weights yolov5s-cls.pt --include onnx --imgsz 224 224 --opset 12
!python export.py --weights yolov5m-cls.pt --include onnx --imgsz 224 224 --opset 12
```

```
export: data=data/coco128.yaml, weights=['yolov5n-cls.pt'], imgsz=[224, 224], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt to yolov5n-cls.pt...
100% 4.87M/4.87M [00:00<00:00, 5.72MB/s]

Fusing layers... 
Model summary: 117 layers, 2489464 parameters, 0 gradients, 3.9 GFLOPs

PyTorch: starting from yolov5n-cls.pt with output shape (1, 1000) (4.9 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 11.7s, saved as yolov5n-cls.onnx (9.5 MB)

Export complete (14.3s)
Results saved to /content/yolov5
Detect:          python classify/predict.py --weights yolov5n-cls.onnx 
Validate:        python classify/val.py --weights yolov5n-cls.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n-cls.onnx')  # WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference
Visualize:       https://netron.app
export: data=data/coco128.yaml, weights=['yolov5s-cls.pt'], imgsz=[224, 224], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt to yolov5s-cls.pt...
100% 10.5M/10.5M [00:01<00:00, 9.64MB/s]

Fusing layers... 
Model summary: 117 layers, 5447688 parameters, 0 gradients, 11.4 GFLOPs

PyTorch: starting from yolov5s-cls.pt with output shape (1, 1000) (10.5 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 0.4s, saved as yolov5s-cls.onnx (20.8 MB)

Export complete (3.1s)
Results saved to /content/yolov5
Detect:          python classify/predict.py --weights yolov5s-cls.onnx 
Validate:        python classify/val.py --weights yolov5s-cls.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.onnx')  # WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference
Visualize:       https://netron.app
export: data=data/coco128.yaml, weights=['yolov5m-cls.pt'], imgsz=[224, 224], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-cls.pt to yolov5m-cls.pt...
100% 24.9M/24.9M [00:02<00:00, 9.38MB/s]

Fusing layers... 
Model summary: 166 layers, 12947192 parameters, 0 gradients, 31.7 GFLOPs

PyTorch: starting from yolov5m-cls.pt with output shape (1, 1000) (24.9 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 0.7s, saved as yolov5m-cls.onnx (49.4 MB)

Export complete (5.3s)
Results saved to /content/yolov5
Detect:          python classify/predict.py --weights yolov5m-cls.onnx 
Validate:        python classify/val.py --weights yolov5m-cls.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m-cls.onnx')  # WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference
Visualize:       https://netron.app
```