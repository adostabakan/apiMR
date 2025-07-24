from flask import Flask, request, jsonify
import os
import uuid
import traceback

from model_loader import load_model
from analyze_tumor import calculate_tumor_ratio
from analyze_nodul import calculate_nodule_count
from stage_classifier import classify_stage

app = Flask(__name__)
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔐 Max upload boyutu (50 MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# 🔁 Modeli belleğe al
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yüklenmedi."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya adı boş."}), 400

    filename = f"{uuid.uuid4().hex}.nii"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        file_size = os.path.getsize(filepath)
        print(f"📥 Dosya alındı: {filename}, Boyut: {round(file_size/1024/1024, 2)} MB")

        tumor_ratio = calculate_tumor_ratio(filepath, model)
        nodule_count = calculate_nodule_count(filepath, model)

        if nodule_count == 0 and tumor_ratio != 0:
            nodule_count = 1

        stage = classify_stage(tumor_ratio, nodule_count)

    except Exception as e:
        print("❌ Hata oluştu:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🧹 Dosya silindi: {filename}")

    return jsonify({
        "tumor_ratio": round(tumor_ratio, 2),
        "nodule_count": int(nodule_count),
        "stage": stage
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
