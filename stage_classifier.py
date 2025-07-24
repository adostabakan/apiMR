# stage_classifier.py

def classify_stage(tumor_ratio, nodule_count):
    score = 0

    # 🔢 Tümör oranı skoru
    if tumor_ratio <= 15:
        score += 0
    elif tumor_ratio <= 35:
        score += 1
    else:
        score += 2

    # 🔢 Nodül sayısı skoru
    if nodule_count <= 2:
        score += 0
    elif 2 <= nodule_count <= 5:
        score += 1
    else:  # 4 veya daha fazla
        score += 2

    # 🎯 Evreyi belirle
    if score == 0:
        return "Erken Evre (A)"
    elif 1 <= score <= 2:
        return "Orta Evre (B)"
    else:
        return "Kritik Evre (C)"
