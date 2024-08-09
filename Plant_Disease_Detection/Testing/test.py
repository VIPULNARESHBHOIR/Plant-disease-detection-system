import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf   
from tensorflow.keras import models, layers
import numpy as np
from PIL import Image

saved_modelT1 = tf.keras.models.load_model(r'models\tomatoE5.h5',compile=False)
saved_modelP1 = tf.keras.models.load_model(r'models\potatoes21.h5',compile=False)

class Plant_DD:

    def __init__(self):

        self.tomato_classes = ['Tomato_Bacterial_spot',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'] # Excluded late and Early blight disease from dataset

        self.potato_classes = ['Potato___Early_blight', 
            'Potato___Late_blight',
            'Potato___healthy']


    def prediction(self, model, img, classes):
        self.im = img
        self.model = model 
        im_array = tf.keras.preprocessing.image.img_to_array(self.im)
        im_array = tf.expand_dims(im_array, 0)
        self.prediction = self.model.predict(im_array)
        if classes == "tomato":
            self.result(self.tomato_classes)
        else:
            self.result(self.potato_classes)

    def result(self, category):
        print("Predicted Label: ", category[np.argmax(self.prediction)])
        print("Confidence: ", round(np.max(self.prediction)*100,2) )


image1 = Image.open(r'Testing\Sample images\spider.jpg')
d1 = Plant_DD()
d1.prediction(saved_modelT1, image1, "tomato")

image2 = Image.open(r'Testing\Sample images\potato-diseases.jpg')
d2 = Plant_DD()
d2.prediction(saved_modelP1, image2, "potato")




