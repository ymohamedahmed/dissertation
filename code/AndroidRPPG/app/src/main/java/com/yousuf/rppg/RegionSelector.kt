package com.yousuf.rppg

import android.graphics.Bitmap
import com.google.android.gms.vision.face.Face
import org.ejml.simple.SimpleMatrix

interface RegionSelector{

    fun detect(face : Face, faceBitmap: Bitmap) : SimpleMatrix

}