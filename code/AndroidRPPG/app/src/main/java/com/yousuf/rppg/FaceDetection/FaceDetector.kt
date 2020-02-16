package com.yousuf.rppg.FaceDetection

import android.util.Log
import android.util.SparseArray
import com.google.android.gms.vision.Detector
import com.google.android.gms.vision.Frame
import com.google.android.gms.vision.face.Face
import com.google.android.gms.vision.face.FaceDetector
import androidx.core.util.valueIterator
import java.io.FileOutputStream
import java.io.IOException
import android.R.attr.top
import android.R.attr.left
import android.content.Context
import android.graphics.*
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.*
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Bitmap
import android.os.Build
import androidx.annotation.RequiresApi
import com.yousuf.rppg.HRIsolation.HRIsolator
import com.yousuf.rppg.RegionSelection.RegionSelector


class FaceDetector(context: Context, roi : RegionSelector, detector: FaceDetector) : Detector<Face>() {
    private var mRoi : RegionSelector = roi
    private var mDetector : FaceDetector = detector
    private var mHRIsolator : HRIsolator = HRIsolator()
    private val TAG = "CustomFaceDetector"
    private val mContext = context
    private val DEBUG = false

    @RequiresApi(Build.VERSION_CODES.O)
    override fun detect(frame: Frame?): SparseArray<Face> {
        val detections = mDetector.detect(frame)
        if (frame != null) {
            val width = frame.metadata.width
            val height = frame.metadata.height
            val yuvImage =
                YuvImage(
                    frame.grayscaleImageData.array(),
                    ImageFormat.NV21,
                    width,
                    height,
                    null
                )
            val byteArrayOutputStream = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, byteArrayOutputStream)
            val jpegArray = byteArrayOutputStream.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(jpegArray, 0, jpegArray.size)
            Log.d(TAG, "Bitmap size: width ${bitmap.width} and height ${bitmap.height}")
            // Rotate bitmap so that face is vertical
            // Then call the region selector on a vertical representation of the face
            // Simplifies all the code for selecting regions

            detections.valueIterator().forEach { face ->
                if (bitmap != null) {
                    Log.d(TAG, "Z angle: ${face.eulerZ}, cos value: ${cos(face.eulerZ*PI/180)}")
                    Log.d(TAG, "Face values: $left $top ${face.width} vs ${bitmap.width-left}, ${face.height} vs ${bitmap.height-top}")
                    var matrix = Matrix()
                    matrix.setRotate(-90f, bitmap.width/2f, bitmap.height/2f)
                    var bitmap = Bitmap.createBitmap(
                        bitmap,
                        0,
                        0,
                        bitmap.width,
                        bitmap.height,
                        matrix,
                        true
                    )
                    var x = (face.position.x+face.width/2f)
                    var y = (face.position.y+face.height/2f)
                    var leftR = (x-face.width/2.0f)
                    var topR = (y-face.height/2.0f)
                    var right = (x+face.width/2.0f)
                    var bottom = (y+face.height/2.0f)
                    var output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                    val canvas = Canvas(output)
                    // Initialize a new Paint instance
                    val paint = Paint()
                    paint.isAntiAlias = true
                    Log.d(TAG, "Position: ($x,$y), frame size (${bitmap.width}, ${bitmap.height})")
                    paint.color = Color.BLACK
                    canvas.drawRect(leftR, topR, right, bottom, paint)
                    canvas.rotate(face.eulerZ,x,y)
                    paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
                    canvas.drawBitmap(bitmap, 0f, 0f, paint)
                    var meanColor = mRoi.detect(face, output)
                    mHRIsolator.put(meanColor)
                    if(DEBUG) {
                        try {

                            var fileName = "rotated_bitmap.png"
                            var file = File(mContext.filesDir, fileName)
                            var out = FileOutputStream(file, false)
                            output.compress(
                                Bitmap.CompressFormat.PNG,
                                100,
                                out
                            )
                            Log.d(TAG, "WRITTEN BITMAP to ${mContext.filesDir}")
                        } catch (e: IOException) {
                            Log.e(TAG, "Error writing bitmap")
                            e.printStackTrace()
                        }
                    }


                }




            }

        }


        return detections
    }

}
