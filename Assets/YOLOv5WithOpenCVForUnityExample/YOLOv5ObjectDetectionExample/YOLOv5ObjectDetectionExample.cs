#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

#if !UNITY_WSA_10_0

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVRect = OpenCVForUnity.CoreModule.Rect;
using OpenCVRange = OpenCVForUnity.CoreModule.Range;

namespace YOLOv5WithOpenCVForUnityExample
{
    /// <summary>
    /// YOLOv5 Object Detection Example
    /// Referring to https://github.com/ultralytics/yolov5.
    /// https://github.com/ultralytics/yolov5/blob/master/detect.py
    /// </summary>
    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class YOLOv5ObjectDetectionExample : MonoBehaviour
    {

        [TooltipAttribute("Path to a binary file of model contains trained weights.")]
        public string model = "yolov5n.onnx";

        [TooltipAttribute("Optional path to a text file with names of classes to label detected objects.")]
        public string classes = "coco.names";

        [TooltipAttribute("Confidence threshold.")]
        public float confThreshold = 0.25f;

        [TooltipAttribute("Non-maximum suppression threshold.")]
        public float nmsThreshold = 0.45f;

        [TooltipAttribute("Maximum detections per image.")]
        public int topK = 1000;

        [TooltipAttribute("Preprocess input image by resizing to a specific width.")]
        public int inpWidth = 640;

        [TooltipAttribute("Preprocess input image by resizing to a specific height.")]
        public int inpHeight = 640;

        [Header("TEST")]

        [TooltipAttribute("Path to test input image.")]
        public string testInputImage;

        /// <summary>
        /// The texture.
        /// </summary>
        protected Texture2D texture;

        /// <summary>
        /// The webcam texture to mat helper.
        /// </summary>
        protected WebCamTextureToMatHelper webCamTextureToMatHelper;

        /// <summary>
        /// The bgr mat.
        /// </summary>
        protected Mat bgrMat;

        /// <summary>
        /// The YOLOv5 ObjectDetector.
        /// </summary>
        YOLOv5ObjectDetector objectDetector;

        /// <summary>
        /// The FPS monitor.
        /// </summary>
        protected FpsMonitor fpsMonitor;

        protected string classes_filepath;
        protected string model_filepath;

#if UNITY_WEBGL
        protected IEnumerator getFilePath_Coroutine;
#endif

        // Use this for initialization
        protected virtual void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();

            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();

#if UNITY_WEBGL
            getFilePath_Coroutine = GetFilePath();
            StartCoroutine(getFilePath_Coroutine);
#else
            if (!string.IsNullOrEmpty(classes))
            {
                classes_filepath = Utils.getFilePath("YOLOv5WithOpenCVForUnityExample/" + classes);
                if (string.IsNullOrEmpty(classes_filepath)) Debug.Log("The file:" + classes + " did not exist in the folder “Assets/StreamingAssets/YOLOv5WithOpenCVForUnityExample”.");
            }
            if (!string.IsNullOrEmpty(model))
            {
                model_filepath = Utils.getFilePath("YOLOv5WithOpenCVForUnityExample/" + model);
                if (string.IsNullOrEmpty(model_filepath)) Debug.Log("The file:" + model + " did not exist in the folder “Assets/StreamingAssets/YOLOv5WithOpenCVForUnityExample”.");
            }
            Run();
#endif
        }

#if UNITY_WEBGL
        protected virtual IEnumerator GetFilePath()
        {
            if (!string.IsNullOrEmpty(classes))
            {
                var getFilePathAsync_0_Coroutine = Utils.getFilePathAsync("YOLOv5WithOpenCVForUnityExample/" + classes, (result) =>
                {
                    classes_filepath = result;
                });
                yield return getFilePathAsync_0_Coroutine;

                if (string.IsNullOrEmpty(classes_filepath)) Debug.Log("The file:" + classes + " did not exist in the folder “Assets/StreamingAssets/YOLOv5WithOpenCVForUnityExample”.");
            }

            if (!string.IsNullOrEmpty(model))
            {
                var getFilePathAsync_1_Coroutine = Utils.getFilePathAsync("YOLOv5WithOpenCVForUnityExample/" + model, (result) =>
                {
                    model_filepath = result;
                });
                yield return getFilePathAsync_1_Coroutine;

                if (string.IsNullOrEmpty(model_filepath)) Debug.Log("The file:" + model + " did not exist in the folder “Assets/StreamingAssets/YOLOv5WithOpenCVForUnityExample”.");
            }

            getFilePath_Coroutine = null;

            Run();
        }
#endif

        // Use this for initialization
        protected virtual void Run()
        {
            //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            Utils.setDebugMode(true);

            if (string.IsNullOrEmpty(model_filepath) || string.IsNullOrEmpty(classes))
            {
                Debug.LogError("model: " + model + " or " + "classes: " + classes + " is not loaded.");
            }
            else
            {
                objectDetector = new YOLOv5ObjectDetector(model_filepath, classes_filepath, new Size(inpWidth, inpHeight), confThreshold, nmsThreshold, topK);
            }


            if (string.IsNullOrEmpty(testInputImage))
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                // Avoids the front camera low light issue that occurs in only some Android devices (e.g. Google Pixel, Pixel2).
                webCamTextureToMatHelper.avoidAndroidFrontCameraLowLightIssue = true;
#endif
                webCamTextureToMatHelper.Initialize();
            }
            else
            {
                /////////////////////
                // TEST

                var getFilePathAsync_0_Coroutine = Utils.getFilePathAsync("YOLOv5WithOpenCVForUnityExample/" + testInputImage, (result) =>
                {
                    string test_input_image_filepath = result;
                    if (string.IsNullOrEmpty(test_input_image_filepath)) Debug.Log("The file:" + testInputImage + " did not exist in the folder “Assets/StreamingAssets/YOLOv5WithOpenCVForUnityExample”.");

                    Mat img = Imgcodecs.imread(test_input_image_filepath);
                    if (img.empty())
                    {
                        img = new Mat(424, 640, CvType.CV_8UC3, new Scalar(0, 0, 0));
                        Imgproc.putText(img, testInputImage + " is not loaded.", new Point(5, img.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                        Imgproc.putText(img, "Please read console message.", new Point(5, img.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    }
                    else
                    {
                        TickMeter tm = new TickMeter();
                        tm.start();

                        Mat results = objectDetector.infer(img);

                        tm.stop();
                        Debug.Log("YOLOv5ObjectDetector Inference time (preprocess + infer + postprocess), ms: " + tm.getTimeMilli());

                        objectDetector.visualize(img, results, true, false);
                    }

                    gameObject.transform.localScale = new Vector3(img.width(), img.height(), 1);
                    float imageWidth = img.width();
                    float imageHeight = img.height();
                    float widthScale = (float)Screen.width / imageWidth;
                    float heightScale = (float)Screen.height / imageHeight;
                    if (widthScale < heightScale)
                    {
                        Camera.main.orthographicSize = (imageWidth * (float)Screen.height / (float)Screen.width) / 2;
                    }
                    else
                    {
                        Camera.main.orthographicSize = imageHeight / 2;
                    }

                    Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);
                    Texture2D texture = new Texture2D(img.cols(), img.rows(), TextureFormat.RGB24, false);
                    Utils.matToTexture2D(img, texture);
                    gameObject.GetComponent<Renderer>().material.mainTexture = texture;

                });
                StartCoroutine(getFilePathAsync_0_Coroutine);

                /////////////////////
            }
        }

        /// <summary>
        /// Raises the webcam texture to mat helper initialized event.
        /// </summary>
        public virtual void OnWebCamTextureToMatHelperInitialized()
        {
            Debug.Log("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();


            texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);

            gameObject.GetComponent<Renderer>().material.mainTexture = texture;

            gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);
            Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", webCamTextureMat.width().ToString());
                fpsMonitor.Add("height", webCamTextureMat.height().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }


            float width = webCamTextureMat.width();
            float height = webCamTextureMat.height();

            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }


            bgrMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC3);
        }

        /// <summary>
        /// Raises the webcam texture to mat helper disposed event.
        /// </summary>
        public virtual void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");

            if (bgrMat != null)
                bgrMat.Dispose();

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }
        }

        /// <summary>
        /// Raises the webcam texture to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        public virtual void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

        // Update is called once per frame
        protected virtual void Update()
        {
            if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
            {

                Mat rgbaMat = webCamTextureToMatHelper.GetMat();

                if (objectDetector == null)
                {
                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
                else
                {

                    Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR);

                    //TickMeter tm = new TickMeter();
                    //tm.start();

                    Mat results = objectDetector.infer(bgrMat);

                    //tm.stop();
                    //Debug.Log("YOLOv5ObjectDetector Inference time (preprocess + infer + postprocess), ms: " + tm.getTimeMilli());

                    Imgproc.cvtColor(bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);

                    objectDetector.visualize(rgbaMat, results, false, true);

                }

                Utils.matToTexture2D(rgbaMat, texture);
            }
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        protected virtual void OnDestroy()
        {
            webCamTextureToMatHelper.Dispose();

            if (objectDetector != null)
                objectDetector.dispose();

            Utils.setDebugMode(false);

#if UNITY_WEBGL
            if (getFilePath_Coroutine != null)
            {
                StopCoroutine(getFilePath_Coroutine);
                ((IDisposable)getFilePath_Coroutine).Dispose();
            }
#endif
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public virtual void OnBackButtonClick()
        {
            SceneManager.LoadScene("YOLOv5WithOpenCVForUnityExample");
        }

        /// <summary>
        /// Raises the play button click event.
        /// </summary>
        public virtual void OnPlayButtonClick()
        {
            webCamTextureToMatHelper.Play();
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public virtual void OnPauseButtonClick()
        {
            webCamTextureToMatHelper.Pause();
        }

        /// <summary>
        /// Raises the stop button click event.
        /// </summary>
        public virtual void OnStopButtonClick()
        {
            webCamTextureToMatHelper.Stop();
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public virtual void OnChangeCameraButtonClick()
        {
            webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.requestedIsFrontFacing;
        }


        private class YOLOv5ObjectDetector
        {
            Size input_size;
            float conf_threshold;
            float nms_threshold;
            int topK;
            int backend;
            int target;

            Net object_detection_net;
            List<string> classNames;

            List<Scalar> palette;

            Mat maxSizeImg;

            Mat boxesMat;
            Mat boxes_m_c4;
            Mat confidences_m;
            MatOfRect2d boxes;
            MatOfFloat confidences;

            public YOLOv5ObjectDetector(string modelFilepath, string classesFilepath, Size inputSize, float confThreshold = 0.25f, float nmsThreshold = 0.45f, int topK = 1000, int backend = Dnn.DNN_BACKEND_OPENCV, int target = Dnn.DNN_TARGET_CPU)
            {
                // initialize
                if (!string.IsNullOrEmpty(modelFilepath))
                {
                    object_detection_net = Dnn.readNet(modelFilepath);
                }

                if (!string.IsNullOrEmpty(classesFilepath))
                {
                    classNames = readClassNames(classesFilepath);
                }

                input_size = inputSize;
                conf_threshold = Mathf.Clamp01(confThreshold);
                nms_threshold = Mathf.Clamp01(nmsThreshold);
                this.topK = topK;
                this.backend = backend;
                this.target = target;

                object_detection_net.setPreferableBackend(this.backend);
                object_detection_net.setPreferableTarget(this.target);

                palette = new List<Scalar>();
                palette.Add(new Scalar(255, 56, 56, 255));
                palette.Add(new Scalar(255, 157, 151, 255));
                palette.Add(new Scalar(255, 112, 31, 255));
                palette.Add(new Scalar(255, 178, 29, 255));
                palette.Add(new Scalar(207, 210, 49, 255));
                palette.Add(new Scalar(72, 249, 10, 255));
                palette.Add(new Scalar(146, 204, 23, 255));
                palette.Add(new Scalar(61, 219, 134, 255));
                palette.Add(new Scalar(26, 147, 52, 255));
                palette.Add(new Scalar(0, 212, 187, 255));
                palette.Add(new Scalar(44, 153, 168, 255));
                palette.Add(new Scalar(0, 194, 255, 255));
                palette.Add(new Scalar(52, 69, 147, 255));
                palette.Add(new Scalar(100, 115, 255, 255));
                palette.Add(new Scalar(0, 24, 236, 255));
                palette.Add(new Scalar(132, 56, 255, 255));
                palette.Add(new Scalar(82, 0, 133, 255));
                palette.Add(new Scalar(203, 56, 255, 255));
                palette.Add(new Scalar(255, 149, 200, 255));
                palette.Add(new Scalar(255, 55, 199, 255));
            }

            protected virtual Mat preprocess(Mat image)
            {
                // Add padding to make it square.
                int max = Mathf.Max(image.cols(), image.rows());

                if (maxSizeImg == null)
                    maxSizeImg = new Mat(max, max, image.type());
                if (maxSizeImg.width() != max || maxSizeImg.height() != max)
                    maxSizeImg.create(max, max, image.type());

                Imgproc.rectangle(maxSizeImg, new OpenCVRect(0, 0, maxSizeImg.width(), maxSizeImg.height()), Scalar.all(114), -1);
                Mat _maxSizeImg_roi = new Mat(maxSizeImg, new OpenCVRect((max - image.cols()) / 2, (max - image.rows()) / 2, image.cols(), image.rows()));
                image.copyTo(_maxSizeImg_roi);

                // Create a 4D blob from a frame.
                Size inpSize = new Size(input_size.width > 0 ? input_size.width : 640,
                    input_size.height > 0 ? input_size.height : 640);
                Mat blob = Dnn.blobFromImage(maxSizeImg, 1.0 / 255.0, inpSize, Scalar.all(0), true, false, CvType.CV_32F); // HWC to NCHW, BGR to RGB

                return blob;// [1, 3, h, w]
            }

            public virtual Mat infer(Mat image)
            {
                // cheack
                if (image.channels() != 3)
                {
                    Debug.Log("The input image must be in BGR format.");
                    return new Mat();
                }

                // Preprocess
                Mat input_blob = preprocess(image);

                // Forward
                object_detection_net.setInput(input_blob);

                List<Mat> output_blob = new List<Mat>();
                object_detection_net.forward(output_blob, object_detection_net.getUnconnectedOutLayersNames());

                // Postprocess
                Mat results = postprocess(output_blob[0], image.size());

                // scale_boxes
                float maxSize = Mathf.Max((float)image.size().width, (float)image.size().height);
                float x_factor = maxSize / (float)input_size.width;
                float y_factor = maxSize / (float)input_size.height;
                float x_shift = (maxSize - (float)image.size().width) / 2f;
                float y_shift = (maxSize - (float)image.size().height) / 2f;

                for (int i = 0; i < results.rows(); ++i)
                {
                    float[] results_arr = new float[4];
                    results.get(i, 0, results_arr);
                    float x = results_arr[0];
                    float y = results_arr[1];
                    float w = results_arr[2];
                    float h = results_arr[3];
                    float x1 = Mathf.Round(x * x_factor - x_shift);
                    float y1 = Mathf.Round(y * y_factor - y_shift);
                    float x2 = Mathf.Round(w * x_factor - x_shift);
                    float y2 = Mathf.Round(h * y_factor - y_shift);

                    results.put(i, 0, new float[] { x1, y1, x2, y2 });
                }


                input_blob.Dispose();
                for (int i = 0; i < output_blob.Count; i++)
                {
                    output_blob[i].Dispose();
                }

                return results;
            }

            protected virtual Mat postprocess(Mat output_blob, Size original_shape)
            {
                Mat output_blob_0 = output_blob;

                if (output_blob_0.size(2) < 85)
                    return new Mat();

                int num = output_blob_0.size(1);
                Mat output_blob_numx85 = output_blob_0.reshape(1, num);
                Mat box_delta = output_blob_numx85.colRange(new OpenCVRange(0, 4));
                Mat confidence = output_blob_numx85.colRange(new OpenCVRange(4, 5));
                Mat classes_scores_delta = output_blob_numx85.colRange(new OpenCVRange(5, 85));

                // Convert boxes from [cx, cy, w, h] to [x, y, w, h] where Rect2d data style.
                if (boxesMat == null)
                    boxesMat = new Mat(num, 4, CvType.CV_32FC1);
                Mat cxy_delta = box_delta.colRange(new OpenCVRange(0, 2));
                Mat wh_delta = box_delta.colRange(new OpenCVRange(2, 4));
                Mat xy1 = boxesMat.colRange(new OpenCVRange(0, 2));
                Mat xy2 = boxesMat.colRange(new OpenCVRange(2, 4));
                wh_delta.copyTo(xy2);
                Core.divide(wh_delta, new Scalar(2.0), wh_delta);
                Core.subtract(cxy_delta, wh_delta, xy1);

                // NMS
                if (boxes_m_c4 == null)
                    boxes_m_c4 = new Mat(num, 1, CvType.CV_64FC4);
                if (confidences_m == null)
                    confidences_m = new Mat(num, 1, CvType.CV_32FC1);

                if (boxes == null)
                    boxes = new MatOfRect2d(boxes_m_c4);
                if (confidences == null)
                    confidences = new MatOfFloat(confidences_m);

                Mat boxes_m_c1 = boxes_m_c4.reshape(1, num);
                boxesMat.convertTo(boxes_m_c1, CvType.CV_64F);
                confidence.copyTo(confidences_m);
                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices, 1f, topK);

                int indicesNum = (int)indices.total();
                List<int> selectedIndices = new List<int>(indicesNum);
                List<int> selectedClassIds = new List<int>(indicesNum);
                List<float> selectedConfidences = new List<float>(indicesNum);

                for (int i = 0; i < indicesNum; ++i)
                {
                    int idx = (int)indices.get(i, 0)[0];

                    float[] classes_scores_arr = new float[80];
                    classes_scores_delta.get(idx, 0, classes_scores_arr);
                    // Get the index of max class score.
                    int class_id = classes_scores_arr.Select((val, _idx) => new { V = val, I = _idx }).Aggregate((max, working) => (max.V > working.V) ? max : working).I;

                    float[] confidence_arr = new float[1];
                    confidences.get(idx, 0, confidence_arr);

                    float newConfidence = classes_scores_arr[class_id] * confidence_arr[0];
                    if (newConfidence > conf_threshold)
                    {
                        selectedIndices.Add(idx);
                        selectedClassIds.Add(class_id);
                        selectedConfidences.Add(newConfidence);
                    }
                }

                Mat results = new Mat(selectedIndices.Count, 6, CvType.CV_32FC1);

                for (int i = 0; i < selectedIndices.Count; ++i)
                {
                    int idx = selectedIndices[i];

                    float[] bbox_arr = new float[4];
                    boxesMat.get(idx, 0, bbox_arr);
                    float x = bbox_arr[0];
                    float y = bbox_arr[1];
                    float w = bbox_arr[2];
                    float h = bbox_arr[3];
                    results.put(i, 0, new float[] { x, y, x + w, y + h, selectedConfidences[i], selectedClassIds[i] });
                }

                indices.Dispose();

                // [
                //   [xyxy, conf, cls, mask]
                //   ...
                //   [xyxy, conf, cls, mask]
                // ]
                return results;
            }

            public virtual void visualize(Mat image, Mat results, bool print_results = false, bool isRGB = false)
            {
                if (image.IsDisposed)
                    return;

                if (results.empty() || results.cols() < 6)
                    return;

                StringBuilder sb = null;

                if (print_results)
                    sb = new StringBuilder();

                for (int i = 0; i < results.rows(); ++i)
                {
                    float[] box = new float[4];
                    results.get(i, 0, box);
                    float[] conf = new float[1];
                    results.get(i, 4, conf);
                    float[] cls = new float[1];
                    results.get(i, 5, cls);

                    float left = box[0];
                    float top = box[1];
                    float right = box[2];
                    float bottom = box[3];
                    int classId = (int)cls[0];

                    Scalar c = palette[classId % palette.Count];
                    Scalar color = isRGB ? c : new Scalar(c.val[2], c.val[1], c.val[0], c.val[3]);

                    Imgproc.rectangle(image, new Point(left, top), new Point(right, bottom), color, 2);

                    string label = String.Format("{0:0.00}", conf[0]);
                    if (classNames != null && classNames.Count != 0)
                    {
                        if (classId < (int)classNames.Count)
                        {
                            label = classNames[classId] + " " + label;
                        }
                    }

                    int[] baseLine = new int[1];
                    Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

                    top = Mathf.Max((float)top, (float)labelSize.height);
                    Imgproc.rectangle(image, new Point(left, top - labelSize.height),
                        new Point(left + labelSize.width, top + baseLine[0]), color, Core.FILLED);
                    Imgproc.putText(image, label, new Point(left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(255), 1, Imgproc.LINE_AA);

                    // Print results
                    if (print_results)
                    {
                        sb.AppendLine(String.Format("-----------object {0}-----------", i + 1));
                        sb.AppendLine(String.Format("conf: {0:0.0000}", conf[0]));
                        sb.AppendLine(String.Format("cls: {0:0}", cls[0]));
                        sb.AppendLine(String.Format("box: {0:0} {1:0} {2:0} {3:0}", box[0], box[1], box[2], box[3]));
                    }
                }

                if (print_results)
                    Debug.Log(sb);
            }

            public virtual void dispose()
            {
                if (object_detection_net != null)
                    object_detection_net.Dispose();

                if (maxSizeImg != null)
                    maxSizeImg.Dispose();

                maxSizeImg = null;

                if (boxesMat != null)
                    boxesMat.Dispose();

                boxesMat = null;

                if (boxes_m_c4 != null)
                    boxes_m_c4.Dispose();
                if (confidences_m != null)
                    confidences_m.Dispose();
                if (boxes != null)
                    boxes.Dispose();
                if (confidences != null)
                    confidences.Dispose();

                boxes_m_c4 = null;
                confidences_m = null;
                boxes = null;
                confidences = null;
            }

            protected virtual List<string> readClassNames(string filename)
            {
                List<string> classNames = new List<string>();

                System.IO.StreamReader cReader = null;
                try
                {
                    cReader = new System.IO.StreamReader(filename, System.Text.Encoding.Default);

                    while (cReader.Peek() >= 0)
                    {
                        string name = cReader.ReadLine();
                        classNames.Add(name);
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError(ex.Message);
                    return null;
                }
                finally
                {
                    if (cReader != null)
                        cReader.Close();
                }

                return classNames;
            }
        }
    }
}
#endif

#endif