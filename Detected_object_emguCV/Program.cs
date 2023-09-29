using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Drawing;

class Program
{
    static void Main(string[] args)
    {
        VideoCapture cap = new VideoCapture(0);

        // Инициализация переменных
        bool motion_detected = false;
        int frame_width = (int)cap.Get(Emgu.CV.CvEnum.CapProp.FrameWidth);
        int frame_height = (int)cap.Get(Emgu.CV.CvEnum.CapProp.FrameHeight);
        VideoWriter outVideo = new VideoWriter("output.avi", VideoWriter.Fourcc('M', 'J', 'P', 'G'), 10, new Size(frame_width, frame_height), true);

        Mat baseline_frame = null;

        while (true)
        {
            // Захват кадров с камеры
            Mat frame = new Mat();
            cap.Read(frame);

            // Преобразуйте кадр в оттенки серого для обнаружения движения
            Mat gray = new Mat();
            CvInvoke.CvtColor(frame, gray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

            
            CvInvoke.GaussianBlur(gray, gray, new Size(21, 21), 0);

            if (!motion_detected)
            {
                // Сохранение базового кадра
                baseline_frame = gray.Clone();
                motion_detected = true;
                continue;
            }

            // Поиск разницы между текущим и базовым кадром
            Mat frame_delta = new Mat();
            CvInvoke.AbsDiff(baseline_frame, gray, frame_delta);

            
            Mat thresh = new Mat();
            CvInvoke.Threshold(frame_delta, thresh, 30, 255, Emgu.CV.CvEnum.ThresholdType.Binary);

            
            CvInvoke.Dilate(thresh, thresh, null, new Point(-1, -1), 2, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1));

            // Поиск контуров
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(thresh.Clone(), contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

            // Прохождение по всем контурам и поиск движния
            for (int i = 0; i < contours.Size; i++)
            {
                if (CvInvoke.ContourArea(contours[i]) > 1000) 
                {
                    motion_detected = true;

                    
                    Rectangle boundingRect = CvInvoke.BoundingRectangle(contours[i]);

                    
                    CvInvoke.Rectangle(frame, boundingRect, new MCvScalar(0, 255, 0), 2);

                    
                    Mat screenshot = new Mat(frame, boundingRect);
                    DateTime currentDataTime = DateTime.Now;
                    screenshot.Save($"screenshot_{currentDataTime}.jpg");
                }
            }

            CvInvoke.Imshow("Обнаружено движение", frame);
            outVideo.Write(frame);

            
            if (CvInvoke.WaitKey(1) == 27)
                break;
        }
       
        cap.Dispose();
        outVideo.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}