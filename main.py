


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io

app = FastAPI(title="Skin Disease Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models
def load_models():
    # Define list of class names
    class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "vitiligo"]
    
    # Load the VGG16 model for feature extraction
    vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
    
    # Load saved model
    model = tf.keras.models.load_model('6claass.h5')
    
    return vgg_model, model, class_names

vgg_model, disease_model, class_names = load_models()

@app.get("/")
async def root():
    return {"message": "Welcome to the Skin Disease Classification API"}

@app.post("/predict/")
async def predict_disease(file: UploadFile = File(...)):
    # Read the file as bytes
    contents = await file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize and preprocess the image
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Extract features using VGG16
    features = vgg_model.predict(img)
    features = features.reshape(1, -1)
    
    # Make prediction using the disease classification model
    prediction = disease_model.predict(features)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    # Get confidence score
    confidence = float(prediction[predicted_class_index])
    
    # Prepare response
    response = {
        "prediction": predicted_class_name
    }
    
    return JSONResponse(content=response)

# For running the app directly
