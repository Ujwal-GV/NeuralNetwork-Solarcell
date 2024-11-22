from flask import Flask, render_template, request, jsonify, redirect, url_for
import train_model
import test
import matplotlib
import os
matplotlib.use('Agg')

app = Flask(__name__)

def save_image(image_file):
    # Define the directory where you want to save the images
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Save the image to the uploads directory
    image_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(image_path)
    
    return image_path

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'UG' and password == 'UG2002':
            return redirect(url_for('index'))
        else:
            error_message = "Incorrect username or password. Please try again."
            return render_template('login.html', error=error_message)
    else:
        return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/train')
def train_func():
    return render_template('train.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/train_model', methods=['POST'])
def train():
    train_model.process()
    return jsonify({'message': 'Training completed successfully'})

@app.route('/test', methods=['POST'])
def predict_image():
    image_file = request.files['image']
    image_path = save_image(image_file)  # Save the image and get its path
    prediction = test.predict_process(image_path)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
    app.run(threaded=False)
