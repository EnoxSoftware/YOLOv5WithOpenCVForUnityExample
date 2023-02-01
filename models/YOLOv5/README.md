# YOLOv5 models

https://github.com/ultralytics/yolov5

Export YOLOv5 model to ONNX:

```
!python export.py --weights yolov5n.pt --include onnx --opset 12
!python export.py --weights yolov5s.pt --include onnx --opset 12
!python export.py --weights yolov5m.pt --include onnx --opset 12
#!python export.py --weights yolov5l.pt --include onnx --opset 12
#!python export.py --weights yolov5x.pt --include onnx --opset 12

#!python export.py --weights yolov5n.pt --include onnx --imgsz 320 320 --opset 12
```

```
export: data=data/coco128.yaml, weights=['yolov5n.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt to yolov5n.pt...
100% 3.87M/3.87M [00:00<00:00, 282MB/s]

Fusing layers... 
YOLOv5n summary: 213 layers, 1867405 parameters, 0 gradients

PyTorch: starting from yolov5n.pt with output shape (1, 25200, 85) (3.9 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 0.7s, saved as yolov5n.onnx (7.6 MB)

Export complete (2.1s)
Results saved to /content/yolov5
Detect:          python detect.py --weights yolov5n.onnx 
Validate:        python val.py --weights yolov5n.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.onnx')  
Visualize:       https://netron.app
export: data=data/coco128.yaml, weights=['yolov5s.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 207MB/s]

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients

PyTorch: starting from yolov5s.pt with output shape (1, 25200, 85) (14.1 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 1.0s, saved as yolov5s.onnx (28.0 MB)

Export complete (2.5s)
Results saved to /content/yolov5
Detect:          python detect.py --weights yolov5s.onnx 
Validate:        python val.py --weights yolov5s.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.onnx')  
Visualize:       https://netron.app
export: data=data/coco128.yaml, weights=['yolov5m.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt to yolov5m.pt...
100% 40.8M/40.8M [00:00<00:00, 347MB/s]

Fusing layers... 
YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients

PyTorch: starting from yolov5m.pt with output shape (1, 25200, 85) (40.8 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 1.9s, saved as yolov5m.onnx (81.2 MB)

Export complete (4.5s)
Results saved to /content/yolov5
Detect:          python detect.py --weights yolov5m.onnx 
Validate:        python val.py --weights yolov5m.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m.onnx')  
Visualize:       https://netron.app
```
