package com.yousuf.rppg

import android.util.Log
import android.util.SparseArray
import androidx.core.util.forEach
import com.google.android.gms.vision.Detector
import com.google.android.gms.vision.Frame
import com.google.android.gms.vision.face.Face
import com.google.android.gms.vision.face.FaceDetector
import android.opengl.ETC1.getHeight
import android.opengl.ETC1.getWidth
import androidx.core.util.valueIterator
import android.view.ViewGroup.LayoutParams.FILL_PARENT
import android.widget.LinearLayout
import android.widget.ImageView.ScaleType
import android.graphics.drawable.BitmapDrawable
import android.R.attr.y
import android.widget.ImageView
import java.io.FileOutputStream
import java.io.IOException
import android.R.attr.bottom
import android.R.attr.right
import android.R.attr.top
import android.R.attr.left
import android.R.array
import android.content.Context
import android.graphics.*
import android.os.Environment
import android.os.storage.StorageManager
import android.os.storage.StorageVolume
import androidx.core.app.ActivityCompat.startActivityForResult
import androidx.core.content.ContextCompat.getSystemService
import java.io.ByteArrayOutputStream
import java.io.File
import java.lang.Double.max
import java.lang.Double.min
import java.lang.Math.pow
import kotlin.math.*
import android.opengl.ETC1.getHeight
import android.opengl.ETC1.getWidth
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.graphics.Bitmap




class FaceDetector(context: Context, roi : RegionSelector, detector: FaceDetector) : Detector<Face>() {
    private var mRoi : RegionSelector = roi
    private var mDetector : FaceDetector = detector
    private val TAG = "CustomFaceDetector"
    private val mContext = context

    override fun detect(frame: Frame?): SparseArray<Face> {
//        val bitmap = frame?.bitmap
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
//                Log.d(TAG, "Looking at face, NULL ${bitmap == null}")
//                matrix.postTranslate(x, y)
                if (bitmap != null) {
//                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, bitmap.width, bitmap.height, true)
//                    val x = face.position.x
//                    val y = face.position.y
//                    var left = (x-face.width/2.0).toInt()
//                    var top = (y-face.height/2.0).toInt()
//                    var matrix = Matrix()
//                matrix.preTranslate(-x, -y)
//                    matrix.setRotate(-face.eulerZ-90, bitmap.width/2f, bitmap.height/2f)
/*
                    matrix.setRotate(face.eulerZ-90, y,x)
                    val bitmap = Bitmap.createBitmap(

                        bitmap,
*/
/*                        0,
                        0,
                        bitmap.width,
                        bitmap.height,
*//*

                        max(top,0),
                        max(left, 0),
                        min(face.height.toInt(), bitmap.height-top),
                        min(face.width.toInt(), bitmap.width-left),
                        matrix,
                        true
                    )
*/
/*
                    val rotatedBitmap = Bitmap.createBitmap(
                        bitmap,
                        max(left, 0),
                        max(top,0),
                        min(face.width.toInt(), bitmap.width-left),
                        min(face.height.toInt(), bitmap.height-top),
                        matrix,
                        true
                    )
*/
//                    val left = bitmap.width/2f-face.width
//                    val top = bitmap.height/2f-face.height
//                    val newY = y/ cos(abs(face.eulerZ*PI/180))
                    Log.d(TAG, "Z angle: ${face.eulerZ}, cos value: ${cos(face.eulerZ*PI/180)}")
//                    val newY = (x.toDouble().pow(2.0) + y.toDouble().pow(2.0)).pow(0.5)
//                    val newX = x*newY/y
//                    Log.d(TAG, "Recalculated center: (x,y) = ($x, $y), (x',y') = ($newX, $newY or ${pow(pow(x.toDouble(),2.0) + pow(y.toDouble(),2.0), 0.5)})")
//                    left = (newX - face.width/2f).toInt()
//                    top = (newY - face.height/2f).toInt()
/*
                    val rotatedBitmap = Bitmap.createBitmap(
                        bitmap,
                        max(left.toInt(), 0),
                        max(top.toInt(),0),
*/
/*
                        newX.toInt(),
                        newY.toInt(),
*//*

                        min(face.width.toInt(), bitmap.width-left.toInt()),
                        min(face.height.toInt(), bitmap.height-top.toInt())
*/
/*
                        0,
                        0,
                        bitmap.width,
                        bitmap.height,
                        matrix,
                        true
*//*

                    )
*/
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
/*
                    bitmap = Bitmap.createBitmap(
                        bitmap,
                        0,
                        0,
                        bitmap.height,
                        bitmap.width
                    )
*/
                    // Calculate the circular bitmap width with border
                    // Initialize a new instance of Bitmap
//                    val dstBitmap = Bitmap.createBitmap(
//                        face.width.toInt(), // Width
//                        face.height.toInt(), // Height
////                        bitmap.height,
////                        bitmap.width,
//                        Bitmap.Config.ARGB_8888 // Config
//                    )
//                    var x = bitmap.width - face.position.x
//                    var y = bitmap.height - face.position.y
                    var x = (face.position.x+face.width/2f)
                    var y = (face.position.y+face.height/2f)
//                    var x = face.position.x
//                    var y = face.position.y
                    var leftR = (x-face.width/2.0f)
                    var topR = (y-face.height/2.0f)
                    var right = (x+face.width/2.0f)
                    var bottom = (y+face.height/2.0f)
                    var output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                    val canvas = Canvas(output)
                    // Initialize a new Paint instance
                    val paint = Paint()
                    paint.isAntiAlias = true
//                    canvas.drawBitmap(bitmap, 0f,0f, paint)
//                    val rect = Rect(0, 0, squareBitmapWidth, squareBitmapWidth)
//                    val rectF = RectF(rect)
//                    canvas.drawOval(rectF, paint)
                    Log.d(TAG, "Position: ($x,$y), frame size (${bitmap.width}, ${bitmap.height})")
                    paint.color = Color.BLACK
//                    canvas.clipRect(leftR, topR, right, bottom)
                    canvas.drawRect(leftR, topR, right, bottom, paint)
//                    canvas.rotate(-face.eulerZ, x, y)
                    canvas.rotate(face.eulerZ,x,y)
                    paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
                    canvas.drawBitmap(bitmap, 0f, 0f, paint)
//                    canvas.drawRect(topR, leftR, bottom, right, paint)
//                    canvas.rotate(face.eulerZ-90,face.width*x/bitmap.width,face.height*y/bitmap.height)
//                    canvas.drawBitmap(bitmap, (face.width-bitmap.width)/2f, (face.height-bitmap.height)/2f, paint)
//                    canvas.drawRect(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat(), paint)

//                    val left = x - xOffset
//                    val top = y - yOffset
//                    val right = x + xOffset
//                    val bottom = y + yOffset
//                    canvas.save()
//                    canvas.rotate(face.eulerZ, x, y)
//                    Log.d(TAG, "FaceOrientation ${face.eulerZ}")
//                    canvas.drawRect(left, top, right, bottom, mBoxPaint)
//                    canvas.drawRect(left, top, right, bottom, roi)
//                    canvas.restore()

//                    canvas.rotate(-face.eulerZ, x,y)
                    // Calculate the left and top of copied bitmap
//                    val left = (squareBitmapWidth - srcBitmap.getWidth()) / 2
//                    val top = (squareBitmapWidth - srcBitmap.getHeight()) / 2
//                    canvas.drawBitmap(bitmap, leftR, topR, paint)
                    // Free the native object associated with this bitmap.
//                    bitmap.recycle()
                    // Return the circular bitmap
/*
                    val croppedBitmap = Bitmap.createBitmap(
                        rotatedBitmap,
                        max(left, 0),
                        max(top,0),
                        min(face.width.toInt(), bitmap.width-left),
                        min(face.height.toInt(), bitmap.height-top)
                    )
*/
//                val croppedBitmap = rotatedBitmap[x]

/*

                    try {

//                        Environment.getDataDirectory().
//                        val context = RPPGApplication.getContext()
                        var dir = Environment.DIRECTORY_PICTURES + "/AndroidRPPG/"
                        var fileName = "rotated_bitmap.png"
                        var file = File(mContext.filesDir, fileName)
                        var file_created = file.createNewFile()
                        Log.d(TAG, "File created: $file_created ")
                        var out = FileOutputStream(file, false)
                        output.compress(
                            Bitmap.CompressFormat.PNG,
                            100,
                            out
                        )
*/
/*
                        fileName = "cropped_bitmap.png"
                        file = File(mContext.filesDir, fileName)
                        file_created = file.createNewFile()
                        Log.d(TAG, "File created: $file_created ")
                        out = FileOutputStream(file, false)
                        dstBitmap.compress(
                            Bitmap.CompressFormat.PNG,
                            100,
                            out
                        ) // bmp is your Bitmap instance
*//*

                        // PNG is a lossless format, the compression factor (100) is ignored
                        Log.d(TAG, "WRITTEN BITMAP to ${mContext.filesDir}")
                    } catch (e: IOException) {
                        Log.e(TAG, "Error writing bitmap")
                        e.printStackTrace()
                    }

*/

                }


            }

//        detections.valueAt()
//        matrix.post
//        bitmap[]
//        mRoi.detect()
        }
        return detections
    }

}
