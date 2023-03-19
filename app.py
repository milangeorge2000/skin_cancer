import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from PIL import Image


app = Flask(__name__)



# Define a flask app
app = Flask(__name__)




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



  
# Load the machine learning model
model = tf.keras.models.load_model('skin_dense.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB
    image = image.convert('RGB')
    # Resize the image to the required input shape of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Preprocess the image using the preprocessing function of the model
    image = tf.keras.applications.densenet.preprocess_input(image)
    # Expand the dimensions of the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Define a dictionary to map class indices to class names
class_names = {
    0: 'actinic keratosis',
    1: 'basal cell carcinoma',
    2: 'dermatofibroma',
    3: 'melanoma',
    4: 'nevus',
    5: 'pigmented benign keratosis',
    6: 'seborrheic keratosis',
    7: 'squamous cell carcinoma',
    8: 'vascular lesion'
}




@app.route('/diseasepredict', methods=['GET', 'POST'])
def upload():
    

    if request.method == 'POST':
        f1 = request.files['file']
        f1.save('img.jpg')
        image = Image.open('img.jpg')

        # Preprocess the test image
        image = preprocess_image(image)

        # Make a prediction using the model
        prediction = model.predict(image)

        # Get the predicted class index
        class_index = np.argmax(prediction)

        # Get the predicted class name
        class_name = class_names[class_index]

    return render_template('result.html',prediction=class_name)
            




if __name__ == '__main__':
    app.run(debug=False)
