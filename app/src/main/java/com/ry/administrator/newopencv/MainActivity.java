package com.ry.administrator.newopencv;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.Toast;

import com.yanzhenjie.permission.Action;
import com.yanzhenjie.permission.AndPermission;
import com.yanzhenjie.permission.Permission;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static String TAG = "MainActivity";
    private MyCvCameraView mCVCamera;
    private int mAbsoluteFaceSize = 0;
    private float mRelativeFaceSize = 0.2f;

    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    android.util.Log.i("TAG", "OpenCV loaded successfully");
//                    System.loadLibrary("detection_based_tracker");

                    AndPermission.with(MainActivity.this)
                            .runtime()
                            .permission(Permission.Group.CAMERA, Permission.Group.STORAGE)
                            .onGranted(new Action<List<String>>() {
                                @Override
                                public void onAction(List<String> data) {
                                    mCVCamera.enableView();
                                }
                            })
                            .onDenied(new Action<List<String>>() {
                                @Override
                                public void onAction(List<String> data) {

                                }
                            })
                            .start();

                    break;
                default:
                    break;
            }
            super.onManagerConnected(status);
        }
    };
    private DetectionBasedTracker mNativeDetector;
    private CascadeClassifier eyeDetector;
    private Mat leftEye_template;
    private Mat rightEye_template;
    private Mat k1;
    private Mat k2;
    private Mat gray = new Mat();
    private static final Scalar FACE_RECT_COLOR = new Scalar(255, 0, 0);//红色
    private static final Scalar EYE_RECT_COLOR = new Scalar(0, 0, 255);//蓝色
    private static final Scalar EYE_COLOR = new Scalar(0, 255, 255);//浅绿
    private Mat rgbaImage;
    private Mat Matlin;
    private Mat gMatlin;
    private int absoluteFaceSize;
    private boolean takepick;
    private int count;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mCVCamera = findViewById(R.id.camera_view);
        // 前置摄像头开启预览
        mCVCamera.setCameraIndex(1);
        mCVCamera.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
            @Override
            public void onCameraViewStarted(int width, int height) {
                rgbaImage = new Mat(width, height, CvType.CV_8UC4);
                Matlin = new Mat(width, height, CvType.CV_8UC4);
                gMatlin = new Mat(width, height, CvType.CV_8UC4);

            }

            @Override
            public void onCameraViewStopped() {

            }

            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

                android.util.Log.e("TAG", "OpenCV 调用了");
//                Mat frame = inputFrame.rgba();
//                Core.flip(frame, frame, 1);


                rgbaImage = inputFrame.rgba();
                Core.flip(rgbaImage, rgbaImage, 1);
                Core.rotate(rgbaImage, Matlin, Core.ROTATE_90_CLOCKWISE);
                process(Matlin);
                Core.rotate(Matlin, rgbaImage, Core.ROTATE_90_COUNTERCLOCKWISE);

//                if (takepick){
//                    saveImg(Matlin,count + "");
//                    count++;
//                }
                return rgbaImage;
            }
        });


        try {
            initDetectBasedTracker();
            initEyesDetector();
            // 缓存眼睛模板
            leftEye_template = new Mat();
            rightEye_template = new Mat();
            // 初始化结构元素
            k1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3), new Point(-1, -1));
            k2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10), new Point(-1, -1));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void initDetectBasedTracker() throws Exception {
        System.loadLibrary("detection_based_tracker");
        InputStream input = getResources().openRawResource(R.raw.lbpcascade_frontalface);
        File cascadeDir = this.getDir("face", Context.MODE_PRIVATE);
        File file = new File(cascadeDir.getAbsoluteFile(), "lbpcascade_frontalface.xml");
        FileOutputStream output = new FileOutputStream(file);
        byte[] buff = new byte[1024];
        int len = 0;
        while ((len = input.read(buff)) != -1) {
            output.write(buff, 0, len);
        }
        input.close();
        output.close();
        mNativeDetector = new DetectionBasedTracker(file.getAbsolutePath(), 0);
        file.delete();
        cascadeDir.delete();
    }

    private void initEyesDetector() throws Exception {
        InputStream input = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
        File cascadeDir = this.getDir("eye", Context.MODE_PRIVATE);
        File file = new File(cascadeDir.getAbsoluteFile(), "haarcascade_eye_tree_eyeglasses.xml");
        FileOutputStream output = new FileOutputStream(file);
        byte[] buff = new byte[1024];
        int len = 0;
        while ((len = input.read(buff)) != -1) {
            output.write(buff, 0, len);
        }
        input.close();
        output.close();
        eyeDetector = new CascadeClassifier(file.getAbsolutePath());
        file.delete();
        cascadeDir.delete();
    }


    public void process(Mat frame) {
        if (mAbsoluteFaceSize == 0) {
            int height = frame.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
            mNativeDetector.start();
        }
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.equalizeHist(gray, gray);
        MatOfRect faces = new MatOfRect();
        mNativeDetector.detect(gray, faces);
        Rect[] facesArray = faces.toArray();
//        Log.e(TAG, "face个数 = " + facesArray.length);
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 2);
          /*  if (takepick) {
                saveImage(frame, facesArray[0], "eye");
            }*/
            selectEyesArea(facesArray[i], frame);
        }
    }

    private void selectEyesArea(Rect faceROI, Mat frame) {
//        if(option < 2) return;
        int offy = (int) (faceROI.height * 0.35f);
        int offx = (int) (faceROI.width * 0.15f);
        int sh = (int) (faceROI.height * 0.18f);
        int sw = (int) (faceROI.width * 0.32f);
        int gap = (int) (faceROI.width * 0.025f);
        Point lp_eye = new Point(faceROI.tl().x + offx, faceROI.tl().y + offy);
        Point lp_end = new Point(lp_eye.x + sw - gap, lp_eye.y + sh);
        Imgproc.rectangle(frame, lp_eye, lp_end, EYE_RECT_COLOR, 2);


        int right_offx = (int) (faceROI.width * 0.095f);
        int rew = (int) (sw * 0.81f);
        Point rp_eye = new Point(faceROI.x + faceROI.width / 2 + right_offx, faceROI.tl().y + offy);
        Point rp_end = new Point(rp_eye.x + rew, rp_eye.y + sh);
        Imgproc.rectangle(frame, rp_eye, rp_end, EYE_RECT_COLOR, 2);

        // 使用级联分类器检测眼睛
//        if(option < 3) return;
        MatOfRect eyes = new MatOfRect();

        Rect left_eye_roi = new Rect();
        left_eye_roi.x = (int) lp_eye.x;
        left_eye_roi.y = (int) lp_eye.y;
        left_eye_roi.width = (int) (lp_end.x - lp_eye.x);
        left_eye_roi.height = (int) (lp_end.y - lp_eye.y);

        Rect right_eye_roi = new Rect();
        right_eye_roi.x = (int) rp_eye.x;
        right_eye_roi.y = (int) rp_eye.y;
        right_eye_roi.width = (int) (rp_end.x - rp_eye.x);
        right_eye_roi.height = (int) (rp_end.y - rp_eye.y);


        // 级联分类器
        Mat leftEye = frame.submat(left_eye_roi);
        if (takepick){
            saveImg(leftEye,"124");
        }

        Mat rightEye = frame.submat(right_eye_roi);
        if (takepick){
            saveImg(rightEye,"1235");
        }

        eyeDetector.detectMultiScale(gray.submat(left_eye_roi), eyes, 1.15, 2, 0, new Size(30, 30), new Size());
        Rect[] eyesArray = eyes.toArray();
//        Log.e(TAG, "eyesArray个数 = " + eyesArray.length);
        for (int i = 0; i < eyesArray.length; i++) {
            Log.i("EYE_DETECTION", "Found Left Eyes...");
            leftEye.submat(eyesArray[i]).copyTo(leftEye_template);
            detectPupil(leftEye.submat(eyesArray[i]));
            Imgproc.rectangle(leftEye, eyesArray[i].tl(), eyesArray[i].br(), EYE_COLOR, 2);
        }
        if (eyesArray.length == 0) {
            Rect left_roi = matchEyeTemplate(leftEye, true);
            if (left_roi != null) {
                detectPupil(leftEye.submat(left_roi));
                Imgproc.rectangle(leftEye, left_roi.tl(), left_roi.br(), EYE_COLOR, 2);
            } else {
                detectPupil(leftEye);
            }
        }

        eyes.release();
        eyes = new MatOfRect();
        eyeDetector.detectMultiScale(gray.submat(right_eye_roi), eyes, 1.15, 2, 0, new Size(30, 30), new Size());
        eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length; i++) {
            Log.i("EYE_DETECTION", "Found Right Eyes...");
            rightEye.submat(eyesArray[i]).copyTo(rightEye_template);
            detectPupil(rightEye.submat(eyesArray[i]));
            Imgproc.rectangle(rightEye, eyesArray[i].tl(), eyesArray[i].br(), EYE_COLOR, 2);
        }
        if (eyesArray.length == 0) {
            Rect right_roi = matchEyeTemplate(rightEye, false);
            if (right_roi != null) {
                detectPupil(rightEye.submat(right_roi));
                Imgproc.rectangle(rightEye, right_roi.tl(), right_roi.br(), EYE_COLOR, 2);
            } else {
                detectPupil(rightEye);
            }
        }


    }


    private void detectPupil(Mat eyeImage) {


//        if(option < 4) return;
        Mat gray = new Mat();
        Mat binary = new Mat();

        Imgproc.cvtColor(eyeImage, gray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, k1);
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, k2);

        renderEye(eyeImage, binary);
        //
        /*if(option > 4) {
            renderEye(eyeImage, binary);
        } else {
            // 轮廓发现
            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

            // 绘制轮廓
            for (int i = 0; i < contours.size(); i++) {
                Imgproc.drawContours(eyeImage, contours, i, new Scalar(0, 255, 0), -1);
            }
            hierarchy.release();
            contours.clear();
        }*/
        gray.release();
        binary.release();
    }

    private void renderEye(Mat eyeImage, Mat mask) {
        //Core.add(eyeImage, new Scalar(100, 30, 10), eyeImage, mask);
        Mat blur_mask = new Mat();
        Mat blur_mask_f = new Mat();

        // 高斯模糊
        Imgproc.GaussianBlur(mask, blur_mask, new Size(3, 3), 0.0);
        blur_mask.convertTo(blur_mask_f, CvType.CV_32F);
        Core.normalize(blur_mask_f, blur_mask_f, 1.0, 0, Core.NORM_MINMAX);

        // 获取数据
        int w = eyeImage.cols();
        int h = eyeImage.rows();
        int ch = eyeImage.channels();
        byte[] data1 = new byte[w * h * ch];
        byte[] data2 = new byte[w * h * ch];
        float[] mdata = new float[w * h];
        blur_mask_f.get(0, 0, mdata);
        eyeImage.get(0, 0, data1);

        // 高斯权重混合
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                int r1 = data1[row * ch * w + col * ch] & 0xff;
                int g1 = data1[row * ch * w + col * ch + 1] & 0xff;
                int b1 = data1[row * ch * w + col * ch + 2] & 0xff;

                int r2 = (data1[row * ch * w + col * ch] & 0xff) + 50;
                int g2 = (data1[row * ch * w + col * ch + 1] & 0xff) + 20;
                int b2 = (data1[row * ch * w + col * ch + 2] & 0xff) + 10;

                float w2 = mdata[row * w + col];
                float w1 = 1.0f - w2;

                r2 = (int) (r2 * w2 + w1 * r1);
                g2 = (int) (g2 * w2 + w1 * g1);
                b2 = (int) (b2 * w2 + w1 * b1);

                r2 = r2 > 255 ? 255 : r2;
                g2 = g2 > 255 ? 255 : g2;
                b2 = b2 > 255 ? 255 : b2;

                data2[row * ch * w + col * ch] = (byte) r2;
                data2[row * ch * w + col * ch + 1] = (byte) g2;
                data2[row * ch * w + col * ch + 2] = (byte) b2;
            }
        }
        eyeImage.put(0, 0, data2);

        // 释放内存
        blur_mask.release();
        blur_mask_f.release();
        data1 = null;
        data2 = null;
        mdata = null;
    }


    private Rect matchEyeTemplate(Mat src, Boolean left) {
        Mat tpl = left ? leftEye_template : rightEye_template;
        if (tpl.cols() == 0 || tpl.rows() == 0) {
            return null;
        }
        int height = src.rows() - tpl.rows() + 1;
        int width = src.cols() - tpl.cols() + 1;
        if (height < 1 || width < 1) {
            return null;
        }
        Mat result = new Mat(height, width, CvType.CV_32FC1);

        // 模板匹配
        int method = Imgproc.TM_CCOEFF_NORMED;
        Imgproc.matchTemplate(src, tpl, result, method);
        Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(result);
        Point maxloc = minMaxResult.maxLoc;

        // ROI
        Rect rect = new Rect();
        rect.x = (int) (maxloc.x);
        rect.y = (int) (maxloc.y);
        rect.width = tpl.cols();
        rect.height = tpl.rows();

        result.release();
        return rect;
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV library not found!");
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        if (mCVCamera != null) {
            mCVCamera.disableView();
        }
        super.onDestroy();
    }


    public void takepic(View view) {
        String path = Environment.getExternalStorageDirectory().getPath() + "/" + "eyenurse.jpg";
//        mCVCamera

        takepick = true;

    }

    public boolean saveImage(Mat image, Rect rect, String fileName) {
        try {
            String PATH = Environment.getExternalStorageDirectory() + "/FaceDetect/" + fileName + ".jpg";
            // 把检测到的人脸重新定义大小后保存成文件
//            Mat sub = image.submat(rect);
//            Mat mat = new Mat();
//            Size size = new Size(100, 100);
//            Imgproc.resize(sub, mat, size);
            Imgcodecs.imwrite(PATH, image);
            takepick = false;
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            takepick = false;
            return false;
        } finally {
            takepick = false;
        }
    }


    private void saveImg(Mat rgba,String name) {
        //先把mat转成bitmap
        Bitmap mBitmap = null;
        //Imgproc.cvtColor(seedsImage, rgba, Imgproc.COLOR_GRAY2RGBA, 4); //转换通道
        mBitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgba, mBitmap);
        String PATH = Environment.getExternalStorageDirectory() + "/FaceDetect/" + name + ".jpg";
        File file = new File(PATH);
        FileOutputStream fileOutputStream = null;
        try {
            if (!file.exists()) {
                // 先得到文件的上级目录，并创建上级目录，在创建文件
                file.getParentFile().mkdir();
                file.createNewFile();
            }

            fileOutputStream = new FileOutputStream(file);
            mBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream);
            fileOutputStream.flush();
            fileOutputStream.close();
            takepick = false;
            Log.d(TAG, "图片已保存至本地");
        } catch (FileNotFoundException e) {
            takepick = false;
            e.printStackTrace();
        } catch (IOException e) {
            takepick = false;
            e.printStackTrace();
        }

    }






}
