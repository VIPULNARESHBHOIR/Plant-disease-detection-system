import tensorflow as tf   
from tensorflow.keras import models, layers
import numpy as np
from PIL import Image 

saved_modelT1 = tf.keras.models.load_model(r'C:\Users\bhoir\Downloads\Plant_Disease_Detection\models\tomatoE5.h5',compile=False)
#saved_modelP1 = tf.keras.models.load_model(r'models\potatoes10.h5',compile=False)

class Plant_DD:

    def __init__(self):

        self.tomato_classes = ['Tomato_Bacterial_spot',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy']

        self.potato_classes = ['Potato___Early_blight', 
            'Potato___Late_blight',
            'Potato___healthy']



    def prediction(self, model, img):
        self.im = img
        self.model = model 
        im_array = tf.keras.preprocessing.image.img_to_array(self.im)
        im_array = tf.expand_dims(im_array, 0)
        self.prediction = self.model.predict(im_array)

    def result(self):
        print("Predicted Label: ", self.tomato_classes[np.argmax(self.prediction)])
        print("Confidence: ", np.max(self.prediction)*100 )

image1 = Image.open('C:\\Users\\bhoir\\Downloads\\Plant_Disease_Detection\\Testing\\Sample images\\yellow_curl_virus_tomato.JPG') 
image2 = Image.open(r'C:\Users\bhoir\Downloads\Plant_Disease_Detection\Testing\Sample images\Bacterial_l.JPG') 
d1 = Plant_DD()
d1.prediction(saved_modelT1, image1)
d1.result()

d2 = Plant_DD()
d2.prediction(saved_modelT1, image2)
d2.result()



