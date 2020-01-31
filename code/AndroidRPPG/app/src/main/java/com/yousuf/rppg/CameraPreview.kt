package com.yousuf.rppg

import android.content.Context
import android.util.AttributeSet
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.ViewGroup
import com.google.android.gms.vision.CameraSource

class CameraPreview(context: Context?, attrs: AttributeSet?) : ViewGroup(context, attrs), SurfaceHolder.Callback{
    override fun surfaceChanged(holder: SurfaceHolder?, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder?) {
    }

    override fun surfaceCreated(holder: SurfaceHolder?) {
    }

    var cameraSource: CameraSource? = null
    var surfaceView: SurfaceView? = null
    var cameraOverlay : CameraOverlay? = null

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        val size = cameraSource?.previewSize
        start(cameraSource, cameraOverlay)
    }
    init {
        surfaceView = SurfaceView(context)
        surfaceView!!.holder.addCallback(this@CameraPreview)
        addView(surfaceView)
    }

    fun start(source: CameraSource?, overlay: CameraOverlay?){
        cameraSource = source
        cameraOverlay = overlay
        cameraSource!!.start(surfaceView?.holder)
        val size = cameraSource!!.previewSize

    }

    private fun stop(){
        cameraSource!!.stop()
    }

}