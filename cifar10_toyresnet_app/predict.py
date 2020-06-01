import base64
import numpy as np
import io
from PIL import Image
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request, Flask

app = Flask(__name__)

global label_names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def get_model():
    global model
    model = load_model('cifa10_ToyResNetv1.h5')
    print(' * Model loaded')

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image

print(' * Loading Keras model...')
get_model()

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(32, 32))

    prediction = model.predict(processed_image)
    highest_pred = np.argmax(prediction)
    int_to_name = label_names[highest_pred]

    response = int_to_name

    return response
