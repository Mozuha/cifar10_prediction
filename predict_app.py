import base64
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request, Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

global label_names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def get_model():
    global model
    model = load_model('cifar10_tfds_woaug_224_ResNet50.h5')
    print(' * Model loaded')

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = image / 255.0
    
    return image

print(' * Loading Keras model...')
get_model()

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image)
    highest_pred = np.argmax(prediction)
    idx2name = label_names[highest_pred]
    response = idx2name

    return response
