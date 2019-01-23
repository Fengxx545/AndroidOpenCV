package com.ry.administrator.newopencv;

import android.app.Application;
import android.util.Log;

import org.opencv.android.OpenCVLoader;

public class Myapp extends Application {

    //OpenCV库静态加载并初始化
    private void staticLoadCVLibraries(){
        boolean load = OpenCVLoader.initDebug();
        if(load) {
            Log.i("CV", "Open CV Libraries loaded...");
        }
    }


    @Override
    public void onCreate() {
        super.onCreate();
        staticLoadCVLibraries();
    }
}
