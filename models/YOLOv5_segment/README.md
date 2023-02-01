# YOLOv5 v7.0 instance segmentation models

https://github.com/ultralytics/yolov5

Export YOLOv5_-seg model to ONNX:

```
!python export.py --weights yolov5n-seg.pt --include onnx --opset 12
!python export.py --weights yolov5s-seg.pt --include onnx --opset 12
!python export.py --weights yolov5m-seg.pt --include onnx --opset 12
```

```
export: data=data/coco128.yaml, weights=['yolov5n-seg.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt to yolov5n-seg.pt...
100% 4.11M/4.11M [00:00<00:00, 19.3MB/s]

Fusing layers... 
YOLOv5n-seg summary: 224 layers, 1986637 parameters, 0 gradients, 7.1 GFLOPs

PyTorch: starting from yolov5n-seg.pt with output shape (1, 25200, 117) (4.1 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 12.2s, saved as yolov5n-seg.onnx (8.0 MB)

Export complete (14.8s)
Results saved to /content/yolov5
Detect:          python segment/predict.py --weights yolov5n-seg.onnx 
Validate:        python segment/val.py --weights yolov5n-seg.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n-seg.onnx')  # WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference
Visualize:       https://netron.app
export: data=data/coco128.yaml, weights=['yolov5s-seg.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt to yolov5s-seg.pt...
100% 14.9M/14.9M [00:02<00:00, 6.80MB/s]

Fusing layers... 
YOLOv5s-seg summary: 224 layers, 7611485 parameters, 0 gradients, 26.4 GFLOPs

PyTorch: starting from yolov5s-seg.pt with output shape (1, 25200, 117) (14.9 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 1.2s, saved as yolov5s-seg.onnx (29.5 MB)

Export complete (5.3s)
Results saved to /content/yolov5
Detect:          python segment/predict.py --weights yolov5s-seg.onnx 
Validate:        python segment/val.py --weights yolov5s-seg.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-seg.onnx')  # WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference
Visualize:       https://netron.app
export: data=data/coco128.yaml, weights=['yolov5m-seg.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-72-g064365d Python-3.8.10 torch-1.13.1+cu116 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt to yolov5m-seg.pt...
100% 42.4M/42.4M [00:03<00:00, 12.7MB/s]

Fusing layers... 
YOLOv5m-seg summary: 301 layers, 21971597 parameters, 0 gradients, 70.8 GFLOPs

PyTorch: starting from yolov5m-seg.pt with output shape (1, 25200, 117) (42.4 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 2.0s, saved as yolov5m-seg.onnx (84.3 MB)

Export complete (9.2s)
Results saved to /content/yolov5
Detect:          python segment/predict.py --weights yolov5m-seg.onnx 
Validate:        python segment/val.py --weights yolov5m-seg.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m-seg.onnx')  # WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference
Visualize:       https://netron.app
```