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
   private var mMeasurementNumber = 0

   private val TAG = "HRIsolator"
   private var mNumberOfPoints = 0
   private var start = 0L


   @RequiresApi(Build.VERSION_CODES.O)
   fun put(red:Float, green:Float, blue:Float){
       if(mNumberOfPoints == 0){
          start = System.currentTimeMillis()
       }
/*
      mRedValues.add(color.red())
      mBlueValues.add(color.blue())
      mGreenValues.add(color.green())
*/
//      var components = doubleArrayOf(color.red().toDouble(), color.green().toDouble(), color.blue().toDouble())
      var components = doubleArrayOf(red.toDouble(), green.toDouble(), blue.toDouble())
      mDataSet.add(DataPoint(DenseVector(components)))
      mNumberOfPoints ++
      if(mNumberOfPoints % 100 == 0){
         Log.d(TAG, "Number of frames collected: $mNumberOfPoints")
      }
      if((mNumberOfPoints == WINDOW_SIZE && mMeasurementNumber == 0) || (mMeasurementNumber > 0 && ((mNumberOfPoints-WINDOW_SIZE)% STRIDE_SIZE) == 0)){
         var timeSeconds = (System.currentTimeMillis()-start)/1000.0
         var sampleFreq = mNumberOfPoints/timeSeconds
         IsolateTask(WINDOW_SIZE, mMeasurementNumber, STRIDE_SIZE, sampleFreq).execute(mDataSet)
         mMeasurementNumber ++
      }
   }


   class IsolateTask(windowSize:Int, measurementNumber: Int, strideSize:Int, sampleFreq : Double) : AsyncTask<SimpleDataSet, Void, Double>() {
      private val TAG = "IsolationTask"
      private val WINDOW_SIZE = windowSize
      private val MIN_HR = 0.5
      private val MAX_HR = 4
      private val SAMPLE_FREQ = sampleFreq
      private val mMeasurementNumber = measurementNumber
      private val STRIDE_SIZE = strideSize

      override fun doInBackground(vararg params: SimpleDataSet): Double? {
         val start = System.currentTimeMillis()
         val lower = mMeasurementNumber*STRIDE_SIZE
         val upper = lower+WINDOW_SIZE
         var dataset = params[0].shallowClone()
         dataset = SimpleDataSet(dataset.dataPoints.filterIndexed{i, dp -> i in lower..upper })
         var ica = FastICA(dataset, 3)
         dataset.applyTransform(ica)
          var icaComponents = dataset.numericColumns.map{it.arrayCopy()}
          Log.d(TAG, "Sample frequency: $SAMPLE_FREQ")
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
         val upperBin = (WINDOW_SIZE*MAX_HR/SAMPLE_FREQ).toInt()
//         Log.d(TAG, "Power spectra")
         Log.d(TAG, "Band pass, lower bound $lowerBin, upper bound: $upperBin")

         // Recall that the indices n/2 to n represent negative frequencies
         powerSpectra = powerSpectra.map{it.mapIndexed{index,power -> if(index < lowerBin || index > upperBin || index > WINDOW_SIZE/2) 0.0 else power }.toDoubleArray()}
         Log.d(TAG, "Power spectra: ${powerSpectra[0].joinToString(",")}")

         // Butterworth filter of the power spectrum
         var butterworth = Butterworth()
//         butterworth.lowPass(3, SAMPLE_FREQ, 0.08)
         butterworth.lowPass(3, SAMPLE_FREQ, 0.8)

         powerSpectra = powerSpectra.map{it.map{p -> butterworth.filter(p)}.toDoubleArray()}
         powerSpectra = powerSpectra.map{it.mapIndexed{index,power -> if(index < lowerBin || index > upperBin || index > WINDOW_SIZE/2) 0.0 else power }.toDoubleArray()}
         Log.d(TAG, "Butterworth filter: ${powerSpectra[0].joinToString(",")}")

         // Select maximum peak
         var maxPower = powerSpectra.map{ it.max()!! }.max()
         Log.d(TAG, "Max power: $maxPower")
//         var frequencyOfMaxPower = powerSpectra.map{it.filterIndexed{index, power -> }}
         var idMaxPowerSpectrum = powerSpectra.indexOfFirst { it.max() == maxPower}
         val heartRateBinId = powerSpectra[idMaxPowerSpectrum].indexOfFirst { it ==  maxPower}
         Log.d(TAG, "Heart rate bin $heartRateBinId")
//          maxOfEachSpectra.mapIndexed{index, maximum -> }
         val heartRate = 60 * heartRateBinId*SAMPLE_FREQ/WINDOW_SIZE
         val end = System.currentTimeMillis()
         Log.d(TAG, "Time to find HR: ${end-start}")


         Log.d(TAG, "Number of data points: ${dataset.dataPoints.size}")
         return heartRate
      }

      override fun onPostExecute(result: Double?) {
          Log.d(TAG, "Heart rate is $result")
      }


   }



}