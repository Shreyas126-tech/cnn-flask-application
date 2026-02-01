import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from model_utils import load_model, make_prediction

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model globally
# Note: In production, consider lazy loading or using a serving infrastructure
print("Loading model...")
model = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        try:
            predictions = make_prediction(model, filepath)
            return render_template('result.html', image_file=filename, predictions=predictions)
        except Exception as e:
            return f"An error occurred: {e}"
            
    return redirect(request.url)

if __name__ == '__main__':
    # Disable reloader to prevent double model loading and potential hangs on Windows
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
