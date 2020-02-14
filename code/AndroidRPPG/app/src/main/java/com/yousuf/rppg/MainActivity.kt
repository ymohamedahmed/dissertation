package com.yousuf.rppg

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import com.google.android.material.snackbar.Snackbar
import androidx.appcompat.app.AppCompatActivity
import android.view.Menu
import com.google.android.gms.vision.MultiProcessor
import android.view.MenuItem
import androidx.core.app.ActivityCompat

import com.yousuf.rppg.R
import android.view.View
import com.google.android.gms.vision.CameraSource
import com.google.android.gms.vision.Detector
import com.google.android.gms.vision.face.FaceDetector
import com.google.android.gms.vision.Tracker
import com.google.android.gms.vision.face.Face
import kotlinx.android.synthetic.main.activity_main.*
import com.google.android.gms.common.GoogleApiAvailability
import com.google.android.gms.common.ConnectionResult
import android.util.Log
import java.io.IOException
import android.graphics.BitmapFactory
import android.graphics.Bitmap
import androidx.core.app.ComponentActivity
import androidx.core.app.ComponentActivity.ExtraData
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
//import android.R




class MainActivity : AppCompatActivity() {

    var preview : CameraPreview? = null
    var overlay : CameraOverlay? = null
    var cameraSource : CameraSource? = null
    val TAG = "MainActivity"


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)
        preview = this.findViewById(R.id.cameraPreview)
        overlay = this.findViewById(R.id.cameraOverlay)

        if (overlay == null){
            Log.d(TAG, "Overlay is null")
        }
        if(ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
            createFaceDetector()
        }else{
            requestPerms()
        }
    }
    fun requestPerms(){
        val permissions = Array(1) {Manifest.permission.CAMERA}
        if(!ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)){
            ActivityCompat.requestPermissions(this, permissions, 1)
            return
        }
        val activity = this
        val listener = object : View.OnClickListener {
            override fun onClick(view: View) {
                ActivityCompat.requestPermissions(
                    activity, permissions,
                    1
                )
            }
        }

        Snackbar.make(
            this!!.overlay!!, "Test",
            Snackbar.LENGTH_INDEFINITE
        ).setAction("Ok", listener).show()

    }

    private fun createFaceDetector(){
        val context = applicationContext
//        val detector = FaceDetector.Builder(context).setClassificationType(FaceDetector.ACCURATE_MODE).build()//.setLandmarkType().setLandmarkType(FaceDetector.ALL_LANDMARKS).setTrackingEnabled(true).build()
        val detector = FaceDetector(context, PrimitiveRoi(), FaceDetector.Builder(context).setClassificationType(FaceDetector.ACCURATE_MODE).setLandmarkType(FaceDetector.ALL_LANDMARKS).setTrackingEnabled(true).build())
        detector.setProcessor(MultiProcessor.Builder(GraphicFaceTrackerFactory()).build())

        cameraSource = CameraSource.Builder(context, detector)
            .setRequestedPreviewSize(640,480)
            .setAutoFocusEnabled(true)
            .setFacing(CameraSource.CAMERA_FACING_FRONT)
            .setRequestedFps(60.0f).build()
    }


    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
//        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }
    private fun startCameraSource() {

        // check that the device has play services available.
        val code = GoogleApiAvailability.getInstance().isGooglePlayServicesAvailable(
            applicationContext
        )
        if (code != ConnectionResult.SUCCESS) {
//            val dlg = GoogleApiAvailability.getInstance().getErrorDialog(this, code, RC_HANDLE_GMS)
//            dlg.show()
        }

        if (cameraSource != null) {
            try {
                Log.d(TAG, "Starting preview")
                Log.d(TAG, "Overlay is null: ${overlay==null}")
                preview?.start(cameraSource, overlay)
            } catch (e: IOException) {
                cameraSource!!.release()
                cameraSource = null
            }

        }
    }

    override fun onResume() {
        super.onResume()
        startCameraSource()
    }


    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
/*
        return when(item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
        */
        return super.onOptionsItemSelected(item)
    }
    private inner class GraphicFaceTrackerFactory : MultiProcessor.Factory<Face> {
        override fun create(face: Face): Tracker<Face> {
            return overlay?.let { GraphicFaceTracker(it) }!!
        }
    }

    /**
     * Face tracker for each detected individual. This maintains a face graphic within the app's
     * associated face overlay.
     */
    private inner class GraphicFaceTracker(cameraOverlay: CameraOverlay) : Tracker<Face>() {
        private val mFaceGraphic: FaceDetectionOverlay

        init {
            overlay = cameraOverlay
            mFaceGraphic = FaceDetectionOverlay(overlay!!)
        }

        /**
         * Start tracking the detected face instance within the face overlay.
         */
        override fun onNewItem(faceId: Int, item: Face) {
            mFaceGraphic.setId(faceId)
        }

        /**
         * Update the position/characteristics of the face within the overlay.
         */
        override fun onUpdate(p0: Detector.Detections<Face>?, detectionResults: Face?) {
            overlay?.add(mFaceGraphic)
            if (detectionResults != null) {
                mFaceGraphic.updateFace(detectionResults)
            }
//            val mBitmap = BitmapFactory.decodeResource(resources, )
        }

        /**
         * Hide the graphic when the corresponding face was not detected.  This can happen for
         * intermediate frames temporarily (e.g., if the face was momentarily blocked from
         * view).
         */
        override fun onMissing(p0: Detector.Detections<Face>?) {
            overlay?.remove(mFaceGraphic)
        }

        /**
         * Called when the face is assumed to be gone for good. Remove the graphic annotation from
         * the overlay.
         */
        override fun onDone() {
            overlay?.remove(mFaceGraphic)
        }

    }
}
