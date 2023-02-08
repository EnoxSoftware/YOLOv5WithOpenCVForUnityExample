using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using OpenCVRange = OpenCVForUnity.CoreModule.Range;
using OpenCVRect = OpenCVForUnity.CoreModule.Rect;

namespace YOLOv5WithOpenCVForUnity
{

    public class YOLOv5ObjectDetector
    {
        Size input_size;
        float conf_threshold;
        float nms_threshold;
        int topK;
        int backend;
        int target;

        int num_classes = 80;

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
                num_classes = classNames.Count;
            }

            input_size = new Size(inputSize.width > 0 ? inputSize.width : 640, inputSize.height > 0 ? inputSize.height : 640);
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
            Mat blob = Dnn.blobFromImage(maxSizeImg, 1.0 / 255.0, input_size, Scalar.all(0), true, false, CvType.CV_32F); // HWC to NCHW, BGR to RGB

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
                float x1 = Mathf.Round(results_arr[0] * x_factor - x_shift);
                float y1 = Mathf.Round(results_arr[1] * y_factor - y_shift);
                float x2 = Mathf.Round(results_arr[2] * x_factor - x_shift);
                float y2 = Mathf.Round(results_arr[3] * y_factor - y_shift);

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

            if (output_blob_0.size(2) < 5 + num_classes)
                return new Mat();

            int num = output_blob_0.size(1);
            Mat output_blob_numx85 = output_blob_0.reshape(1, num);
            Mat box_delta = output_blob_numx85.colRange(new OpenCVRange(0, 4));
            Mat confidence = output_blob_numx85.colRange(new OpenCVRange(4, 5));
            Mat classes_scores_delta = output_blob_numx85.colRange(new OpenCVRange(5, 5 + num_classes));

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

                float[] classes_scores_arr = new float[num_classes];
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
            //   [xyxy, conf, cls]
            //   ...
            //   [xyxy, conf, cls]
            // ]
            return results;
        }

        public virtual void visualize(Mat image, Mat results, bool print_results = false, bool isRGB = false)
        {
            if (image.IsDisposed)
                return;

            if (results.empty() || results.cols() < 6)
                return;

            for (int i = results.rows() - 1; i >= 0; --i)
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
            }

            // Print results
            if (print_results)
            {
                StringBuilder sb = new StringBuilder();

                for (int i = 0; i < results.rows(); ++i)
                {
                    float[] box = new float[4];
                    results.get(i, 0, box);
                    float[] conf = new float[1];
                    results.get(i, 4, conf);
                    float[] cls = new float[1];
                    results.get(i, 5, cls);

                    int classId = (int)cls[0];
                    string label = String.Format("{0:0}", cls[0]);
                    if (classNames != null && classNames.Count != 0)
                    {
                        if (classId < (int)classNames.Count)
                        {
                            label = classNames[classId] + " " + label;
                        }
                    }

                    sb.AppendLine(String.Format("-----------object {0}-----------", i + 1));
                    sb.AppendLine(String.Format("conf: {0:0.0000}", conf[0]));
                    sb.AppendLine(String.Format("cls: {0:0}", label));
                    sb.AppendLine(String.Format("box: {0:0} {1:0} {2:0} {3:0}", box[0], box[1], box[2], box[3]));
                }

                Debug.Log(sb);
            }
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