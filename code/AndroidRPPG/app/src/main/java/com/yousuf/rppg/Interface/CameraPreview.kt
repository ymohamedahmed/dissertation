package com.yousuf.rppg.Interface

import android.content.Context
import android.util.AttributeSet
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.ViewGroup
import com.google.android.gms.vision.CameraSource
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


class CameraPreview(context: Context?, attrs: AttributeSet?) : ViewGroup(context, attrs), SurfaceHolder.Callback{
    override fun surfaceChanged(holder: SurfaceHolder?, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder?) {
    }

    override fun surfaceCreated(holder: SurfaceHolder?) {
        surfaceAvailable = true
        start(cameraSource, cameraOverlay)
    }

    var cameraSource: CameraSource? = null
    var surfaceView: SurfaceView? = null
    var cameraOverlay : CameraOverlay? = null
    var startRequested = false
    var surfaceAvailable = false
    val TAG = "CameraPreview"

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        Log.d(TAG, "OnLayout")
        var height = 320
        var width = 240
        if (cameraSource != null){
            val size = cameraSource!!.previewSize
            if(size != null) {
                height = size.height
                width = size.width
            }
        }
        val layoutWidth = r - l
        val layoutHeight = b - t

        // Computes height and width for potentially doing fit width.
        var childWidth = layoutWidth
        var childHeight = (height * layoutWidth.toFloat()/width.toFloat()).roundToInt()


        // If height is too tall using fit width, does fit height instead.
        if (childHeight > layoutHeight) {
            childHeight = layoutHeight
//            childWidth = (layoutHeight.toFloat() / height).times(width)
            childWidth = (width*layoutHeight.toFloat()/height.toFloat()).roundToInt()
        }

        for (i in 0 until childCount) {
            getChildAt(i).layout(0, 0, childWidth, childHeight)
        }
        start(cameraSource, cameraOverlay)
    }
    init {
        surfaceView = SurfaceView(context)
        surfaceView!!.holder.addCallback(this@CameraPreview)
        addView(surfaceView)
    }

    fun start(source: CameraSource?, overlay: CameraOverlay?){
        Log.d(TAG, "Start StartRequest: ${startRequested}, SurfaceAvailable: $surfaceAvailable ")
        if (overlay != null) {
            cameraOverlay = overlay

        }
        if (source == null) {
            stop()
        }

        cameraSource = source
        if(cameraSource != null){
            startRequested = true
        }

        if(startRequested && surfaceAvailable) {
            cameraSource?.start(surfaceView?.holder)
            Log.d(TAG, "Starting Camera Source")
            Log.d(TAG, "CameraOverlay is null: ${cameraOverlay == null}")
            if (cameraOverlay != null) {
                val size = cameraSource?.previewSize
                val max = max(size!!.width, size.height)
                val min = min(size.width, size.height)
                Log.d(TAG, "Setting Camera Info ${size.width}, ${size.height}")
                cameraOverlay!!.setCameraInfo(
                    min,
                    max,
                    cameraSource!!.cameraFacing
                )
                cameraOverlay?.clear()
            }
        }
        startRequested = false
    }

    private fun stop(){
        cameraSource?.stop()
    }

    private fun release(){
        cameraSource?.release()
        cameraSource = null
    }


}