package com.yousuf.ppglogger

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.support.wearable.activity.WearableActivity
import android.util.Log
import android.view.KeyEvent
import android.widget.TextView
import com.opencsv.CSVWriter
import java.io.File
import java.io.FileWriter
import java.sql.Timestamp
import java.util.*
import kotlin.collections.ArrayList


class MainActivity : WearableActivity() {
    val mPPGListener = PPGListener()
    val TAG = "MainActivity"
    var timestamp : Timestamp? = null
    var recording = false
    var start = System.currentTimeMillis()
    var mSensorManager : SensorManager? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        mSensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val sensorList: List<Sensor> = mSensorManager!!.getSensorList(Sensor.TYPE_ALL)
        for (currentSensor in sensorList) {
            Log.d("List sensors", "Name: " + currentSensor.name.toString() + " /Type_String: " + currentSensor.stringType.toString() + " /Type_number: " + currentSensor.type)
        }
        setAmbientEnabled()
    }
    fun writePPGData(data: ArrayList<Array<String>>, time: Long){
        val samplingFreq = data.size/(time/1000)
        val filePath = applicationContext.filesDir.absolutePath + File.separator + "$timestamp-freq-$samplingFreq.csv"
        var file = File(filePath)
        var writer : CSVWriter

        writer = if(file.exists() && !file.isDirectory) {
            var fileWriter = FileWriter(filePath, true)
            CSVWriter(fileWriter, ',', CSVWriter.NO_QUOTE_CHARACTER)
        } else {
            CSVWriter(FileWriter(filePath),',', CSVWriter.NO_QUOTE_CHARACTER)
        }

        writer.writeNext(arrayOf("Timestamp", "PPG"))

        for(row in data){
            writer.writeNext(row)
        }
        Log.d(TAG, "Written PPG data to $timestamp-freq-$samplingFreq.csv")
        writer.close();
    }
    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        val textView = findViewById<TextView>(R.id.statusTextView)
        return if (event.repeatCount == 0) {
            when (keyCode) {
                KeyEvent.KEYCODE_STEM_1 -> {
                    recording = !recording
                    if(recording){
                        Log.d(TAG, "Beginning to record PPG data")
                        timestamp = Timestamp(Date().time)
                        start = System.currentTimeMillis()
                        mSensorManager!!.registerListener(mPPGListener,
                                mSensorManager!!.getDefaultSensor(65572),
                                SensorManager.SENSOR_DELAY_FASTEST)
                        textView.text = "Recording"

                    }else{
                        Log.d(TAG, "Stopping recording PPG data and writing output")
                        // We've stopped recording so write the output to a file
                        val end = System.currentTimeMillis()
                        mSensorManager?.unregisterListener(mPPGListener)
                        writePPGData(mPPGListener.mData, end-start)
                        textView.text = "Not recording"
                    }
                    true
                }
                else -> {
                    super.onKeyDown(keyCode, event)
                }
            }
        } else {
            super.onKeyDown(keyCode, event)
        }
    }

    class PPGListener : SensorEventListener {
        val TAG = "PPGListener"
        var mData : ArrayList<Array<String>> = ArrayList()
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        }

        override fun onSensorChanged(event: SensorEvent?) {
//            Log.d(TAG, "Timestamp: ${event?.timestamp.toString()}")
            var row = arrayOf(event?.timestamp.toString(), event?.values!![0].toRawBits().toString())
            mData.add(row)
//            mData.add(arrayOf())
//            mData.add(event?.values!![0].toRawBits().toFloat())
        }

    }

}
