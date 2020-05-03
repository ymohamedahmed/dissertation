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
import com.yousuf.rppg.Interface.CameraOverlay
import com.yousuf.rppg.Interface.CameraPreview
import com.yousuf.rppg.Interface.FaceDetectionOverlay
import java.io.IOException
import com.yousuf.rppg.RegionSelection.PrimitiveRoi


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
        val detector = com.yousuf.rppg.FaceDetection.FaceDetector(
            context,
            PrimitiveRoi(),
            FaceDetector.Builder(context).setClassificationType(FaceDetector.FAST_MODE)//.setTrackingEnabled(false).build()
                    .setLandmarkType(
                FaceDetector.ALL_LANDMARKS
            ).setTrackingEnabled(true).build()
        )
        detector.setProcessor(MultiProcessor.Builder(GraphicFaceTrackerFactory()).build())

        cameraSource = CameraSource.Builder(context, detector)
            .setRequestedPreviewSize(640, 480)
            .setAutoFocusEnabled(true)
            .setFacing(CameraSource.CAMERA_FACING_FRONT)
            .setRequestedFps(60.0f).build()
    }


    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        return true
    }
    private fun startCameraSource() {

        val code = GoogleApiAvailability.getInstance().isGooglePlayServicesAvailable(
            applicationContext
        )

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
        return super.onOptionsItemSelected(item)
    }
    private inner class GraphicFaceTrackerFactory : MultiProcessor.Factory<Face> {
        override fun create(face: Face): Tracker<Face> {
            return overlay?.let { GraphicFaceTracker(it) }!!
        }
    }

    private inner class GraphicFaceTracker(cameraOverlay: CameraOverlay) : Tracker<Face>() {
        private val mFaceGraphic: FaceDetectionOverlay

        init {
            overlay = cameraOverlay
            mFaceGraphic = FaceDetectionOverlay(overlay!!)
        }

        override fun onNewItem(faceId: Int, item: Face) {
            mFaceGraphic.setId(faceId)
        }

        override fun onUpdate(p0: Detector.Detections<Face>?, detectionResults: Face?) {
            overlay?.add(mFaceGraphic)
            if (detectionResults != null) {
                mFaceGraphic.updateFace(detectionResults)
            }
        }

        override fun onMissing(p0: Detector.Detections<Face>?) {
            overlay?.remove(mFaceGraphic)
        }

        override fun onDone() {
            overlay?.remove(mFaceGraphic)
        }

    }
}
