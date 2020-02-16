package com.yousuf.rppg.HRIsolation

import android.graphics.Color
import android.os.AsyncTask
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import jsat.DataSet
import jsat.SimpleDataSet
import jsat.classifiers.CategoricalData
import jsat.classifiers.DataPoint
import jsat.datatransform.FastICA
import jsat.linear.DenseVector
import jsat.linear.Vec
import java.util.*

class HRIsolator(){
   private var mDataSet = SimpleDataSet(null, 3)
   private val WINDOW_SIZE = 1024
   private val STRIDE_SIZE = 30

   private val TAG = "HRIsolator"
   private var mNumberOfPoints = 0

   @RequiresApi(Build.VERSION_CODES.O)
   fun put(color: Color){
/*
      mRedValues.add(color.red())
      mBlueValues.add(color.blue())
      mGreenValues.add(color.green())
*/
      var components = doubleArrayOf(color.red().toDouble(), color.green().toDouble(), color.blue().toDouble())
      mDataSet.add(DataPoint(DenseVector(components)))
      mNumberOfPoints ++
      if(mNumberOfPoints == WINDOW_SIZE){
         IsolateTask().execute(mDataSet)
      }
   }


   class IsolateTask : AsyncTask<SimpleDataSet, Void, Float>() {
      private val TAG = "IsolationTask"
      override fun doInBackground(vararg params: SimpleDataSet): Float? {
         var dataset = params[0]
         var ica = FastICA(dataset, 3)
         dataset.applyTransform(ica)
          var icaComponents = dataset.numericColumns.map{it.arrayCopy()}
/*
         var icaComponents = dataset.dataPoints.map {
            it.numericalValues[i]
         }.toDoubleArray()
*/
         // Compute FFT


         // Bandpass filter for between 40BPM and 240BPM

         // Butterworth filter of the power spectrum

         // Select maximum peak


         Log.d(TAG, "Number of data points: ${dataset.dataPoints.size}")
         return heartRate
      }

      override fun onPostExecute(result: Float?) {
          Log.d(TAG, "Heart rate is $result")
      }


   }



}