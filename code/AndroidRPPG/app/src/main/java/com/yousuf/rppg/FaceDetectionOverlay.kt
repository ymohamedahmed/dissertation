package com.yousuf.rppg

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.google.android.gms.vision.face.Face

internal class FaceDetectionOverlay(overlay: CameraOverlay) : CameraOverlay.Graphic(overlay) {

    private val mFacePositionPaint: Paint
    private val mIdPaint: Paint
    private val mBoxPaint: Paint
    private val roi : Paint

    @Volatile
    private var mFace: Face? = null
    private var mFaceId: Int = 0
    private val mFaceHappiness: Float = 0.toFloat()

    private final val TAG = "FaceDetectionOverlay"

    init {

        mCurrentColorIndex = (mCurrentColorIndex + 1) % COLOR_CHOICES.size
        val selectedColor = COLOR_CHOICES[mCurrentColorIndex]

        mFacePositionPaint = Paint()
        mFacePositionPaint.color = selectedColor

        mIdPaint = Paint()
        mIdPaint.color = selectedColor
        mIdPaint.textSize = ID_TEXT_SIZE

        mBoxPaint = Paint()
        mBoxPaint.color = selectedColor
        mBoxPaint.style = Paint.Style.STROKE
        mBoxPaint.strokeWidth = BOX_STROKE_WIDTH

        roi = Paint()
        roi.color = selectedColor
        roi.alpha = 60
        roi.style = Paint.Style.FILL_AND_STROKE

    }

    fun setId(id: Int) {
        mFaceId = id
    }


    /**
     * Updates the face instance from the detection of the most recent frame.  Invalidates the
     * relevant portions of the overlay to trigger a redraw.
     */
    fun updateFace(face: Face) {
        mFace = face
        postInvalidate()
    }

    /**
     * Draws the face annotations for position on the supplied canvas.
     */
    override fun draw(canvas: Canvas) {
        val face = mFace ?: return
        face.landmarks
        // Draws a circle at the position of the detected face, with the face's track id below.
        val x = translateX(face.position.x + face.width / 2)
        val y = translateY(face.position.y + face.height / 2)
        canvas.drawCircle(x, y, FACE_POSITION_RADIUS, mFacePositionPaint)
//        canvas.drawBitma
/*
        canvas.drawText("id: $mFaceId", x + ID_X_OFFSET, y + ID_Y_OFFSET, mIdPaint)
        canvas.drawText(
            "happiness: " + String.format("%.2f", face.getIsSmilingProbability()),
            x - ID_X_OFFSET,
            y - ID_Y_OFFSET,
            mIdPaint
        )
        canvas.drawText(
            "right eye: " + String.format("%.2f", face.getIsRightEyeOpenProbability()),
            x + ID_X_OFFSET * 2,
            y + ID_Y_OFFSET * 2,
            mIdPaint
        )
        canvas.drawText(
            "left eye: " + String.format("%.2f", face.getIsLeftEyeOpenProbability()),
            x - ID_X_OFFSET * 2,
            y - ID_Y_OFFSET * 2,
            mIdPaint
        )
*/


//        val positions  = face.landmarks.map{(scaleX(it.position.x), scaleY(it.position.y)}.toFloatArray()

//        Log.d("FaceDetection", "Length of positions ${positions.size}")
//        Log.d("FaceDetection", "Length of positions ${face.landmarks.size}")
//        Log.d("FaceDetection", positions.foldRight(""){acc, e -> acc.toString() + e})
//        Log.d("FaceDetection", "Positions ${positions.contentToString()}")
//        canvas.drawPoints(positions, mBoxPaint)
//        positions.forEach { canvas.drawCircle() }
        face.landmarks.forEach{canvas.drawCircle(translateX(it.position.x), translateY(it.position.y), 10.0f, mIdPaint)}
        // Draws a bounding box around the face.
        val xOffset = scaleX(face.width / 2.0f)
        val yOffset = scaleY(face.height / 2.0f)
        val left = x - xOffset
        val top = y - yOffset
        val right = x + xOffset
        val bottom = y + yOffset
        canvas.save()
        canvas.rotate(face.eulerZ, x, y)
        Log.d(TAG, "FaceOrientation ${face.eulerZ}")
        canvas.drawRect(left, top, right, bottom, mBoxPaint)
        canvas.drawRect(left, top, right, bottom, roi)
        canvas.restore()
//        bitmap = Bitmap()
//            canvas.drawBitmap()
    }

    companion object {
        private val FACE_POSITION_RADIUS = 10.0f
        private val ID_TEXT_SIZE = 40.0f
        private val ID_Y_OFFSET = 50.0f
        private val ID_X_OFFSET = -50.0f
        private val BOX_STROKE_WIDTH = 5.0f

        private val COLOR_CHOICES = intArrayOf(
            Color.BLUE,
            Color.CYAN,
            Color.GREEN,
            Color.MAGENTA,
            Color.RED,
            Color.WHITE,
            Color.YELLOW
        )
        private var mCurrentColorIndex = 0
    }
}