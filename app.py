from flask import Flask, request, jsonify
import os
import tempfile
from main import predict_cat_probability

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create a temporary file in a cross-platform way
    temp_dir = tempfile.gettempdir()
    img_path = os.path.join(temp_dir, file.filename)
    file.save(img_path)

    probability = predict_cat_probability(img_path)

    return jsonify({
        'cat_probability': float(probability),
        'is_cat': probability > 0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
