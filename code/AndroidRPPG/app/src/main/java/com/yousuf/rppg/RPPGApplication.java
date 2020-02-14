package com.yousuf.rppg;

import android.app.Application;
import android.content.Context;

public abstract class RPPGApplication extends Application {

    private static Context context;

    public void onCreate(){
        super.onCreate();
        RPPGApplication.context = getApplicationContext();
    }

    public static Context getContext(){
        return context;
    }
}
