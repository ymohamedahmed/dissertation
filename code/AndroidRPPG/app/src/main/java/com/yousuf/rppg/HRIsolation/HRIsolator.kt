package com.yousuf.rppg.HRIsolation

import android.graphics.Color
import android.os.AsyncTask
import android.os.Build
import android.util.Log
import android.view.Window
import androidx.annotation.RequiresApi
import jsat.DataSet
import jsat.SimpleDataSet
import jsat.classifiers.CategoricalData
import jsat.classifiers.DataPoint
import jsat.datatransform.FastICA
import jsat.linear.DenseVector
import jsat.linear.Vec
import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.TransformType
import uk.me.berndporr.iirj.Butterworth
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
      if(mNumberOfPoints % 100 == 0){
         Log.d(TAG, "Number of frames collected: $mNumberOfPoints")
      }
      if(mNumberOfPoints == WINDOW_SIZE){
         IsolateTask(WINDOW_SIZE).execute(mDataSet)
      }
   }


   class IsolateTask(windowSize:Int) : AsyncTask<SimpleDataSet, Void, Double>() {
      private val TAG = "IsolationTask"
      private val WINDOW_SIZE = windowSize
      private val MIN_HR = 0.2
      private val MAX_HR = 4
      private val SAMPLE_FREQ = 60.0
      override fun doInBackground(vararg params: SimpleDataSet): Double? {
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
         var fft = FastFourierTransformer(DftNormalization.STANDARD)
         var powerSpectra = icaComponents.map {  fft.transform(it, TransformType.FORWARD).map { complex -> complex.abs() }.toDoubleArray()}

         // Bandpass filter for between 30BPM (0.5Hz) and 240BPM (4Hz)

         // The formula for each fft-bin is freq = (id*sampleFreq) / N
         // So id =(N*freq)/sampleFreq
         // We get a total of N bins for an N-bit input
         // Here N is 1024 (i.e. the window size)
         val lowerBin = (WINDOW_SIZE*MIN_HR/SAMPLE_FREQ).toInt()
         val upperBin = (WINDOW_SIZE*MIN_HR/SAMPLE_FREQ).toInt()

         powerSpectra = powerSpectra.map{it.mapIndexed{index,power -> if(index < lowerBin || index > upperBin) 0.0 else power }.toDoubleArray()}

         // Butterworth filter of the power spectrum
         var butterworth = Butterworth()
         butterworth.lowPass(3, SAMPLE_FREQ, 0.2)

         powerSpectra = powerSpectra.map{it.map{p -> butterworth.filter(p)}.toDoubleArray()}

         // Select maximum peak
         var maxPower = powerSpectra.map{ it.max()!! }.max()
//         var frequencyOfMaxPower = powerSpectra.map{it.filterIndexed{index, power -> }}
         var idMaxPowerSpectrum = powerSpectra.indexOfFirst { it.max() == maxPower}
         val heartRateBinId = powerSpectra[idMaxPowerSpectrum].indexOfFirst { it ==  maxPower}

//          maxOfEachSpectra.mapIndexed{index, maximum -> }
         val heartRate = 60 * heartRateBinId*SAMPLE_FREQ/WINDOW_SIZE


         Log.d(TAG, "Number of data points: ${dataset.dataPoints.size}")
         return heartRate
      }

      override fun onPostExecute(result: Double?) {
          Log.d(TAG, "Heart rate is $result")
      }


   }



}