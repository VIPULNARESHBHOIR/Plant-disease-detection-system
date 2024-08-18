import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(r'models\tomatoE5.h5',compile=False)
tomato_classes = ['Tomato_Bacterial_spot',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy']


@app.get("/ping")
async def ping():
    return "Hello, i am vipul"

def image_into_np_array(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    return image
    
@app.post("/predict")
async def predict( file: UploadFile = File(...) ):

    image = image_into_np_array( await file.read())
    img = np.expand_dims(image, 0)
    
    predictions  = model.predict(img) 
    predicted_label = tomato_classes[np.argmax(predictions[0])]
    score = round(np.max(predictions[0]),2)

    return {
        'label' : predicted_label,
        'score' : float(score)
    }


if __name__ == '__main__':

    uvicorn.run(app, host = 'localhost', port = 8000)