from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# --- Ensure Upload Folder Exists ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model ---
model = tf.keras.models.load_model('coffee_classifier.h5')

# --- Label Mapping ---
labels = ['Arabica', 'Liberica', 'Excelsa', 'Robusta']

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=(150, 150))  # <-- Resize here
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Make it (1, 150, 150, 3)
    return img_array

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            import uuid
            import os
            raw_filename = file.filename
            # Clean and make filename unique
            cleaned_name = secure_filename(os.path.basename(raw_filename))
            filename = f"{uuid.uuid4().hex}_{cleaned_name}"
            filepath = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
            print("Saving to:", filepath)  # Optional: helpful for debugging

            file.save(filepath)


            # Preprocess image and make prediction
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

            return render_template(
                "index.html",
                filename=filename,
                label=predicted_label,
                confidence=f"{confidence:.2f}%"
            )
        else:
            flash('Allowed file types: png, jpg, jpeg')
            return redirect(request.url)

    return render_template("index.html")

@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# --- Run App ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
