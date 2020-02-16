package com.yousuf.rppg.RegionSelection

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Color
import com.google.android.gms.vision.face.Face
import com.yousuf.rppg.RegionSelection.RegionSelector
import org.ejml.simple.SimpleMatrix


class PrimitiveRoi : RegionSelector {
    @SuppressLint("NewApi")
    override fun detect(face: Face, bitmap: Bitmap): Color {
//        face.position
//        frame.bitmap[]
//        return 1*SimpleMatrix(bitmap.height, bitmap.width).scale()
//        return SimpleMatrix(bitmap.height, bitmap.width)
        val size = (bitmap.width * bitmap.height).toFloat()
        var r = 0f
        var g = 0f
        var b = 0f
        for (x in 0 until bitmap.width step 1) {
            for (y in 0 until bitmap.height step 1) {
                var color = bitmap.getColor(x, y)
                r += color.red()
                g += color.green()
                b += color.blue()
            }
        }
        return Color.valueOf(r/size, g/size, b/size)
    }

}