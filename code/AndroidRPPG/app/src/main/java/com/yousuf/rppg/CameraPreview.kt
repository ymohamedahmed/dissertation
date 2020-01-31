package com.yousuf.rppg

import android.content.Context
import android.graphics.Camera
import android.util.AttributeSet
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.ViewGroup
import com.google.android.gms.vision.CameraSource

class CameraPreview(context: Context?, attrs: AttributeSet?) : ViewGroup(context, attrs) {

    var cameraSource: CameraSource? = null
    var surfaceView: SurfaceView? = null
    var overlay : CameraOverlay = null

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }
    init {
        surfaceView = SurfaceView(context)
        surfaceView!!.holder.addCallback(SurfaceCallback())
        addView(surfaceView)
    }

    fun start(source: CameraSource, overlay: CameraOverlay?){
        cameraSource = source

    }
    private fun stop(){
        cameraSource!!.stop()
    }
    fun startCamera() {

    }
    class SurfaceCallback : SurfaceHolder.Callback {
        override fun surfaceDestroyed(holder: SurfaceHolder?) {
            val surfaceAvailable = true

        }

        override fun surfaceCreated(holder: SurfaceHolder?) {
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
        }

        override fun surfaceChanged(holder: SurfaceHolder?, format: Int, width: Int, height: Int) {
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
        }

    }

}