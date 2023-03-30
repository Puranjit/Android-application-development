package com.example.blueberiyield

import android.graphics.Bitmap
import android.graphics.Paint
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.view.TextureView
import android.widget.EditText
import android.widget.ImageView
import com.example.blueberiyield.ml.Model
import org.tensorflow.lite.support.image.ImageProcessor

class MainActivity2 : AppCompatActivity() {


    lateinit var textInput: EditText
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main2)



        textInput = findViewById(R.id.textInput)
    }

}