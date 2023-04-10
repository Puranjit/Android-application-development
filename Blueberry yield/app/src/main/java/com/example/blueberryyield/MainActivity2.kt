package com.sunayanpradhan.imagecropper

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
import android.widget.Toast
import com.sunayanpradhan.imagecropper.ml.ColabPro
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity2 : AppCompatActivity() {

    lateinit var bitmap: Bitmap
    var count = 0

    lateinit var imageView: ImageView
    lateinit var button1: Button

    val paint = Paint()

    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)).build()

    lateinit var colabPro: ColabPro

    lateinit var labels: List<String>


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main2)

        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        colabPro = ColabPro.newInstance(this)

        imageView = findViewById(R.id.ImageV)
        button1 = findViewById(R.id.button1)

        button1.setOnClickListener{
            startActivityForResult(intent, 101)
        }

        labels = FileUtil.loadLabels(this, "labels.txt")

    }
    override fun onDestroy() {
        super.onDestroy()
        colabPro.close()
//        model.close()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101) {
            var uri = data?.data

//            Uri - uniform resource indicator - location where image is located


            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            get_predictions()

            val result = calculateResult(count) // your code that returns a result
            val toast = Toast.makeText(this, "The result is: $result", Toast.LENGTH_SHORT)
            toast.show()
            count = 0
        }
    }


    // perform prediction
    fun get_predictions(){

        var image = TensorImage.fromBitmap(bitmap)

        image = imageProcessor.process(image)

        val outputs = colabPro.process(image)
        val detectionResult = outputs.detectionResultList.get(0)

        val location = outputs.locationAsTensorBuffer.floatArray
        val category = outputs.categoryAsTensorBuffer.floatArray
        val score = outputs.scoreAsTensorBuffer.floatArray
        val numOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_4444, true)
        val canvas = Canvas(mutable)

        val h = mutable.height
        val w = mutable.width

        paint.textSize = h/19f
        paint.strokeWidth = h/275f
        var x = 0

        score.forEachIndexed{index, fl ->
            x = index
            x *=4

            if(fl > 0.01){
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(location.get(x+1)*w, location.get(x)*h, location.get(x+3)*w, location.get(x+2)*h), paint)
                paint.style = Paint.Style.FILL
//                canvas.drawText(labels.get(category.get(index).toInt()) + fl.toString(), location.get(x+1)*w, location.get(x)*h, paint)
                if(labels.get(category.get(index).toInt()) == "Ripe"){
                    count++
                }
            }
        }
        imageView.setImageBitmap(mutable)
    }

    fun calculateResult(n: Int):Int{
        val result = n
        return result
    }



}