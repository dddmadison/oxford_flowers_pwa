from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import json

# 🔹 Flask 앱 초기화
app = Flask(__name__)

# 🔹 베이스 디렉터리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ 경량 모델 로드
model_path = os.path.join(BASE_DIR, 'models', 'oxford_flower_tiny.keras')
model = tf.keras.models.load_model(model_path)

# ✅ Render cold start crash 방지용 warm-up
_ = model.predict(np.zeros((1, 224, 224, 3)))

# 🔹 클래스 이름 로딩
json_path = os.path.join(BASE_DIR, 'cat_to_name.json')
with open(json_path, 'r') as f:
    class_names = json.load(f)
print(f"[INFO] Loaded class names: {list(class_names.keys())[:5]} ...")

# 🔹 이미지 전처리 함수
def process_image(image: np.ndarray) -> np.ndarray:
    image_tensor = tf.convert_to_tensor(image)
    image_resized = tf.image.resize(image_tensor, (224, 224))
    image_resized /= 255
    return image_resized.numpy()

# 🔹 홈 페이지 라우팅
@app.route('/', methods=['GET'])
def index():
    return render_template('index_pwa.html')

# 🔹 예측 라우트
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    # 🔹 이미지 저장
    filename = 'uploaded_image.jpg'
    static_dir = os.path.join(BASE_DIR, 'static')
    os.makedirs(static_dir, exist_ok=True)
    filepath = os.path.join(static_dir, filename)
    file.save(filepath)

    # 🔹 이미지 전처리
    im = Image.open(filepath).convert('RGB')
    image_arr = np.asarray(im)
    processed_image = process_image(image_arr)
    processed_image = np.expand_dims(processed_image, 0)

    # 🔹 예측 수행
    try:
        predictions = model.predict(processed_image)
        top_k_index = np.argsort(predictions[0])[-1:]
        print('top_k_index: ', top_k_index)

        classes_names_list = [class_names[str(index + 1)] for index in top_k_index]
        predicted_flower = classes_names_list[0]
        predicted_probability = predictions[0][top_k_index[0]]
    except Exception as e:
        app.logger.error(f'[PREDICT ERROR] {e}')
        return jsonify({'error': str(e)}), 500

    return render_template(
        'index_pwa.html',
        predicted_flower=predicted_flower,
        predicted_probability="{:.1%}".format(predicted_probability),
        image_path=os.path.join('static', filename)
    )

# 🔹 앱 실행
if __name__ == '__main__':
    app.run(debug=True, port=5000)
