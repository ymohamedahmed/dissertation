package com.yousuf.rppg.RegionSelection

import android.graphics.Bitmap
import com.google.android.gms.vision.face.Face
import com.yousuf.rppg.RegionSelection.RegionSelector
import org.ejml.simple.SimpleMatrix


class PrimitiveRoi : RegionSelector {
    override fun detect(face: Face, bitmap: Bitmap): SimpleMatrix {
//        face.position
//        frame.bitmap[]
//        return 1*SimpleMatrix(bitmap.height, bitmap.width).scale()
        return SimpleMatrix(bitmap.height, bitmap.width)
    }

}