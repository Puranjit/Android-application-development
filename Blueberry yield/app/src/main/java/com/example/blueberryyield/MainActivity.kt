package com.example.blueberiyield

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import com.example.blueberiyield.ml.Model
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    lateinit var imageView: ImageView
    lateinit var button1: Button

    lateinit var button2: Button
    lateinit var button3: Button

    lateinit var bitmap: Bitmap
    val paint = Paint()

    lateinit var model: Model

    lateinit var labels: List<String>

    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(512, 512, ResizeOp.ResizeMethod.BILINEAR)).build()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        model = Model.newInstance(this)

        labels = FileUtil.loadLabels(this, "labels.txt")

        imageView = findViewById(R.id.ImageV)
        button1 = findViewById(R.id.button)

        button1.setOnClickListener{
            startActivityForResult(intent, 101)
        }

        button2 = findViewById(R.id.button2)
        button2.setOnClickListener {
            val Intent1 = Intent(this, MainActivity2::class.java)
            startActivity(Intent1)
        }

        button3 = findViewById(R.id.button3)
        button3.setOnClickListener {
            val Intent2 = Intent(this, Realtime::class.java)
            startActivity(Intent2)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101) {
            var uri = data?.data

//            Uri - uniform resource indicator - location where image is located


            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            get_predictions()
        }
    }


    // perform prediction
    fun get_predictions(){

        var image = TensorImage.fromBitmap(bitmap)

        image = imageProcessor.process(image)

        val outputs = model.process(image)
//                val detectionResult = outputs.detectionResultList.get(0)

        val location = outputs.locationAsTensorBuffer.floatArray
        val category = outputs.categoryAsTensorBuffer.floatArray
        val score = outputs.scoreAsTensorBuffer.floatArray
        val numOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)

        val h = mutable.height
        val w = mutable.width

        paint.textSize = h/19f
        paint.strokeWidth = h/175f
        var x = 0

        score.forEachIndexed{index, fl ->
            x = index
            x *=4

            if(fl > 0.55){
//                        paint.setColor(colors.get(index))
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(location.get(x+1)*w, location.get(x)*h, location.get(x+3)*w, location.get(x+2)*h), paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(labels.get(category.get(index).toInt()) + fl.toString(), location.get(x+1)*w, location.get(x)*h, paint)
            }
        }

        imageView.setImageBitmap(mutable)
    }
}