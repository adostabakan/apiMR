from flask import Flask, request, jsonify
import os
import uuid

from model_loader import load_model
from analyze_tumor import calculate_tumor_ratio
from analyze_nodul import calculate_nodule_count
from stage_classifier import classify_stage  # 🆕 EKLENDİ

app = Flask(__name__)
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔁 Modeli sadece bir kere belleğe al
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yüklenmedi."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya adı boş."}), 400

    # 📥 Dosyayı kaydet
    filename = f"{uuid.uuid4().hex}.nii"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        tumor_ratio = calculate_tumor_ratio(filepath, model)
        nodule_count = calculate_nodule_count(filepath, model)

        if nodule_count == 0 and tumor_ratio != 0:
            nodule_count = 1

        # 🆕 Evreyi belirle
        stage = classify_stage(tumor_ratio, nodule_count)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # 🧹 Dosya silinsin
        if os.path.exists(filepath):
            os.remove(filepath)

    # 📤 Cevap
    return jsonify({
        "tumor_ratio": round(tumor_ratio, 2),
        "nodule_count": int(nodule_count),
        "stage": stage  # 🆕 EKLENDİ
    })

if __name__ == '__main__':
    app.run(debug=True)
