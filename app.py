import os
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import image_processing

app = Flask(__name__)
app.secret_key = 'golden_ratio_face_analyzer_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user doesn't select a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Process the image
            output_filename = 'analyzed_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            try:
                # Call the image processing function
                result = image_processing.process_image(input_path, output_path)
                
                if result:
                    # Redirect to result page with analysis data
                    return render_template('result.html', 
                                          filename=output_filename,
                                          original=filename,
                                          score=round(result['score'], 1),
                                          measurements=result['measurements'],
                                          image_data=result['image'],
                                          golden_ratio=1.618)
                else:
                    flash('Failed to analyze the face. Please try another image.')
                    return redirect(request.url)
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)