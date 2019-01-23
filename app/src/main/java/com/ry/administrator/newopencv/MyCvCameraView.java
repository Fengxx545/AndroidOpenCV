package com.ry.administrator.newopencv;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.Log;

import org.opencv.android.JavaCameraView;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class MyCvCameraView extends JavaCameraView implements Camera.PictureCallback {
    private String TAG = "MyCvCameraView";
    private String imageFileName;
    private boolean takePhotoFlag = false;

    public MyCvCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void takePhoto(String name){
        imageFileName = name;
        mCamera.takePicture(null, null, this);
        takePhotoFlag = true;
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        Log.i(TAG, "Saving a bitmap to file");
        if (takePhotoFlag){
            Camera.Size previewSize = mCamera.getParameters().getPreviewSize();
            BitmapFactory.Options newOpts = new BitmapFactory.Options();
            newOpts.inJustDecodeBounds = true;
            YuvImage yuvimage = new YuvImage(
                    data,
                    ImageFormat.NV21,
                    previewSize.width,
                    previewSize.height,
                    null);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            yuvimage.compressToJpeg(new Rect(0, 0, previewSize.width, previewSize.height), 100, baos);
            byte[] rawImage = baos.toByteArray();
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.RGB_565;
            Bitmap bmp = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length, options);
            try {
                BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(imageFileName));
                bmp.compress(Bitmap.CompressFormat.JPEG, 100, bos);
                bos.flush();
                bos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            bmp.recycle();
            takePhotoFlag = false;
        }
        synchronized (this) {
/*            mFrameChain[mChainIdx].put(0, 0, frame);
            mCameraFrameReady = true;*/
            this.notify();
        }
/*        if (mCamera != null)
            mCamera.addCallbackBuffer(mBuffer);*/
    }
}