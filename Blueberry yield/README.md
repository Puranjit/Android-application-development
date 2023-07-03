<b>Blueberry yield prediction android application</b>

* I am currently working on developing this project as a part of my research work at Auburn University.

The primary objective of this research project is to design and develop an Android application that offers a user-friendly interface and demonstrates efficient real-time blueberry yield estimation. The application leverages advanced deep learning-based object detection techniques to accurately identify blueberries within images collected from different blueberry cultivars in the fields.<br>
The proposed Android application aims to provide a seamless experience to users, allowing them to effortlessly capture images of blueberry plants in the field using their smartphone's camera. The deep learning algorithms integrated into the application perform robust object detection, precisely locating and delineating individual blueberries within the images.<br>
Once the blueberries are detected, the application utilizes predictive modeling to estimate the yield of ripe blueberries in the fields. By leveraging statistical analysis and machine learning algorithms, the application can provide reliable and accurate predictions regarding the number of ripe blueberries present in a given area.<br>
By enabling real-time analysis and estimation, this application offers a practical solution for farmers, researchers, and blueberry industry professionals. It empowers them to make informed decisions regarding harvest planning, resource allocation, and overall crop management strategies. Additionally, the user-friendly interface ensures accessibility, making the application valuable to a wide range of users, even those with limited technical expertise.<br>
Ultimately, the successful implementation of this research project will contribute to the advancement of blueberry farming practices by providing a convenient tool for yield estimation. It has the potential to optimize resource utilization, improve productivity, and enhance overall decision-making processes within the blueberry industry.<br>

1. I have pre-trained an efficient net-d3 .tflite model on my custom dataset to detect green (unripe), half-ripe, and ripe berries
2. The DL model would be deployed on an Android application to perform the prediction tasks
3. The Android application would provide 3 different user-friendly applications for the user -
* Performing real-time blueberry detection and predicting yields
* Open camera - click an image - edit the image, and then perform predictions and predict yields
* Use images from the photo gallery on an Android phone and use it to perform blueberry predictions and predict approximate yield estimates for different cultivars

This is an ongoing research project and I will continuously update this folder with the updated models and Android application development
