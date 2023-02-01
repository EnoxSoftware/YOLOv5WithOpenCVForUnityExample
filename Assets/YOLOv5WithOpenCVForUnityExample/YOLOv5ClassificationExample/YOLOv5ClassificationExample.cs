#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

#if !UNITY_WSA_10_0

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
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
    /// YOLOv5 Classification Example
    /// Referring to https://github.com/ultralytics/yolov5.
    /// https://github.com/ultralytics/yolov5/blob/master/classify/predict.py
    /// </summary>
    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class YOLOv5ClassificationExample : MonoBehaviour
    {

        [TooltipAttribute("Path to a binary file of model contains trained weights.")]
        public string model = "yolov5n-cls.onnx";

        [TooltipAttribute("Optional path to a text file with names of classes to label detected objects.")]
        public string classes = "imagenet_labels.txt";

        [TooltipAttribute("Preprocess input image by resizing to a specific width.")]
        public int inpWidth = 224;

        [TooltipAttribute("Preprocess input image by resizing to a specific height.")]
        public int inpHeight = 224;

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
        /// The YOLOv5 class predictor.
        /// </summary>
        YOLOv5ClassPredictor classPredictor;

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
                classPredictor = new YOLOv5ClassPredictor(model_filepath, classes_filepath, new Size(inpWidth, inpHeight));
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

                        Mat results = classPredictor.infer(img);

                        tm.stop();
                        Debug.Log("YOLOv5ClassPredictor Inference time (preprocess + infer + postprocess), ms: " + tm.getTimeMilli());

                        classPredictor.visualize(img, results, true, false);

                        if (fpsMonitor != null)
                        {
                            fpsMonitor.consoleText = "Best match class " + classPredictor.getBestMatchLabel(results);
                        }
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

                if (classPredictor == null)
                {
                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
                else
                {

                    Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR);

                    //TickMeter tm = new TickMeter();
                    //tm.start();

                    Mat results = classPredictor.infer(bgrMat);

                    //tm.stop();
                    //Debug.Log("YOLOv5ClassPredictor Inference time (preprocess + infer + postprocess), ms: " + tm.getTimeMilli());

                    Imgproc.cvtColor(bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);

                    classPredictor.visualize(rgbaMat, results, false, true);

                    if (fpsMonitor != null)
                    {
                        fpsMonitor.consoleText = "Best match class " + classPredictor.getBestMatchLabel(results);
                    }

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

            if (classPredictor != null)
                classPredictor.dispose();

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


        private class YOLOv5ClassPredictor
        {
            Size input_size;
            int backend;
            int target;

            Scalar IMAGENET_MEAN = new Scalar(00.485, 0.456, 0.406);// RGB mean
            Scalar IMAGENET_STD = new Scalar(0.229, 0.224, 0.225);// RGB standard deviation

            Net classification_net;
            List<string> classNames;

            List<Scalar> palette;

            Mat input_sizeMat;

            public YOLOv5ClassPredictor(string modelFilepath, string classesFilepath, Size inputSize, int backend = Dnn.DNN_BACKEND_OPENCV, int target = Dnn.DNN_TARGET_CPU)
            {
                // initialize
                if (!string.IsNullOrEmpty(modelFilepath))
                {
                    classification_net = Dnn.readNet(modelFilepath);
                }

                if (!string.IsNullOrEmpty(classesFilepath))
                {
                    classNames = readClassNames(classesFilepath);
                }

                input_size = inputSize;
                this.backend = backend;
                this.target = target;

                classification_net.setPreferableBackend(this.backend);
                classification_net.setPreferableTarget(this.target);

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
                // Create a 4D blob from a frame.
                Size inpSize = new Size(input_size.width > 0 ? input_size.width : 224,
                    input_size.height > 0 ? input_size.height : 224);

                int c = image.channels();
                int h = (int)inpSize.height;
                int w = (int)inpSize.width;

                if (input_sizeMat == null)
                    input_sizeMat = new Mat(h, w, CvType.CV_8UC3);// [h, w]

                int imh = image.height();
                int imw = image.width();
                int m = Mathf.Min(imh, imw);
                int top = (int)((imh - m) / 2f);
                int left = (int)((imw - m) / 2f);
                Mat image_crop = new Mat(image, new OpenCVRect(0, 0, image.width(), image.height()).intersect(new OpenCVRect(left, top, m, m)));
                Imgproc.resize(image_crop, input_sizeMat, new Size(w, h));

                Mat blob = Dnn.blobFromImage(input_sizeMat, 1.0 / 255.0, inpSize, Scalar.all(0), true, false, CvType.CV_32F); // HWC to NCHW, BGR to RGB

                Mat blob_cxhxw = blob.reshape(1, new int[] { c, h, w });// [c, h, w]

                for (int i = 0; i < c; ++i)
                {
                    Mat blob_1xhw = blob_cxhxw.row(i).reshape(1, 1);// [3, 224, 224] => [1, 50176]

                    // Subtract blob by mean.
                    Core.subtract(blob_1xhw, new Scalar(IMAGENET_MEAN.val[i]), blob_1xhw);
                    // Divide blob by std.
                    Core.divide(blob_1xhw, new Scalar(IMAGENET_STD.val[i]), blob_1xhw);
                }

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
                classification_net.setInput(input_blob);

                List<Mat> output_blob = new List<Mat>();
                classification_net.forward(output_blob, classification_net.getUnconnectedOutLayersNames());

                // Postprocess
                Mat results = postprocess(output_blob, image.size());

                input_blob.Dispose();
                for (int i = 0; i < output_blob.Count; i++)
                {
                    output_blob[i].Dispose();
                }

                return results;
            }

            protected virtual Mat postprocess(List<Mat> output_blob, Size original_shape)
            {
                Mat output_blob_0 = output_blob[0];

                Mat results = softmax(output_blob_0);

                return results;// [1, 1000]
            }

            protected virtual Mat softmax(Mat src)
            {
                Mat dst = src.clone();

                Core.MinMaxLocResult result = Core.minMaxLoc(src);
                Scalar max = new Scalar(result.maxVal);
                Core.subtract(src, max, dst);
                Core.exp(dst, dst);
                Scalar sum = Core.sumElems(dst);
                Core.divide(dst, sum, dst);

                return dst;
            }

            public virtual void visualize(Mat image, Mat results, bool print_results = false, bool isRGB = false)
            {
                if (image.IsDisposed)
                    return;

                if (results.empty() || results.cols() < classNames.Count)
                    return;

                StringBuilder sb = null;

                if (print_results)
                    sb = new StringBuilder();

                Core.MinMaxLocResult minmax = Core.minMaxLoc(results);
                int classId = (int)minmax.maxLoc.x;

                string label = String.Format("{0:0.00}", minmax.maxVal);
                if (classNames != null && classNames.Count != 0)
                {
                    if (classId < (int)classNames.Count)
                    {
                        label = classNames[classId] + " " + label;
                    }
                }

                Scalar c = palette[classId % palette.Count];
                Scalar color = isRGB ? c : new Scalar(c.val[2], c.val[1], c.val[0], c.val[3]);

                int[] baseLine = new int[1];
                Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, 1, baseLine);

                float top = 20f + (float)labelSize.height;
                float left = (float)(image.width() / 2 - labelSize.width / 2f);

                top = Mathf.Max((float)top, (float)labelSize.height);
                Imgproc.rectangle(image, new Point(left, top - labelSize.height),
                    new Point(left + labelSize.width, top + baseLine[0]), color, Core.FILLED);
                Imgproc.putText(image, label, new Point(left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar.all(255), 1, Imgproc.LINE_AA);

                // Print results
                if (print_results)
                {
                    sb.AppendLine(String.Format("Best match: " + (int)minmax.maxLoc.x + ", " + classNames[(int)minmax.maxLoc.x] +", " + String.Format("{0:0.00}", minmax.maxVal)));
                }

                if (print_results)
                    Debug.Log(sb);
            }

            public virtual String getBestMatchLabel(Mat results)
            {
                if (results.empty() || results.cols() < classNames.Count)
                    return string.Empty;

                Core.MinMaxLocResult minmax = Core.minMaxLoc(results);
                int classId = (int)minmax.maxLoc.x;
                double probability = minmax.maxVal;
                string className = "";
                if (classNames != null && classNames.Count != 0)
                {
                    if (classId < (int)classNames.Count)
                    {
                        className = classNames[classId];
                    }
                }

                return classId + "," + className + "," + String.Format("{0:0.00}", probability);
            }

            public virtual void dispose()
            {
                if (classification_net != null)
                    classification_net.Dispose();

                if (input_sizeMat != null)
                    input_sizeMat.Dispose();

                input_sizeMat = null;
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