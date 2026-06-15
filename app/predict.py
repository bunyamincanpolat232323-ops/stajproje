"""
Tahmin Modulu
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Bu modul yeni yorumlar icin duygu tahmini yapar.
Kaydedilmis en iyi modeli ve TF-IDF vektorleyicisini kullanir.
"""

import os
import sys
import json
import joblib
import numpy as np

# Proje kok dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_text
from app.database import save_prediction


def load_model():
    """
    Kaydedilmis en iyi modeli yukler.

    Returns:
        tuple: (model, model_name)
    """
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    info_path = os.path.join(models_dir, 'best_model_info.json')
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        model_name = info['model_name']
    except FileNotFoundError:
        model_name = 'Bilinmiyor'

    model_path = os.path.join(models_dir, 'best_model.pkl')
    try:
        model = joblib.load(model_path)
        return model, model_name
    except FileNotFoundError:
        raise FileNotFoundError("Model dosyasi bulunamadi. Lutfen once egitim yapin: python src/train_models.py")


def load_vectorizer():
    """Kaydedilmis TF-IDF vektorleyicisini yukler."""
    vectorizer_path = os.path.join(PROJECT_ROOT, 'models', 'tfidf_vectorizer.pkl')
    try:
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer
    except FileNotFoundError:
        raise FileNotFoundError("Vectorizer bulunamadi. Lutfen once egitim yapin: python src/train_models.py")


def load_specific_model(model_name):
    """Belirli bir modeli dosyadan yukler."""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    filename = f"{model_name.lower().replace(' ', '_')}.pkl"
    model_path = os.path.join(models_dir, filename)

    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")


def predict_sentiment(text, model=None, vectorizer=None, model_name=None):
    """
    Verilen metin icin duygu tahmini yapar.

    Args:
        text (str): Analiz edilecek yorum metni
        model: Kullanilacak model (None ise en iyi model yuklenir)
        vectorizer: TF-IDF vektorleyici (None ise yuklenir)
        model_name (str): Model adi

    Returns:
        dict: Tahmin sonuclari
    """
    try:
        if model is None:
            model, model_name = load_model()
        if vectorizer is None:
            vectorizer = load_vectorizer()

        # Metni on isle
        cleaned_text = preprocess_text(text)

        if not cleaned_text.strip():
            return {
                'predicted_sentiment': 'notr',
                'confidence_score': 0.0,
                'model_name': model_name or 'Bilinmiyor',
                'probabilities': {}
            }

        # TF-IDF donusumu
        text_vector = vectorizer.transform([cleaned_text])

        # Tahmin
        prediction = model.predict(text_vector)[0]

        # Guven skoru
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vector)[0]
            classes = model.classes_
            probabilities = {cls: float(prob) for cls, prob in zip(classes, proba)}
            confidence = float(max(proba))
        else:
            confidence = 1.0

        return {
            'predicted_sentiment': prediction,
            'confidence_score': confidence,
            'model_name': model_name or 'Bilinmiyor',
            'probabilities': probabilities
        }

    except Exception as e:
        print(f"Tahmin hatasi: {e}")
        raise


def predict_and_save(text, model=None, vectorizer=None, model_name=None):
    """
    Tahmin yapar ve sonucu SQL veritabanina (predictions tablosuna) kaydeder.
    """
    result = predict_sentiment(text, model, vectorizer, model_name)

    try:
        record_id = save_prediction(
            review_text=text,
            predicted_sentiment=result['predicted_sentiment'],
            confidence_score=result['confidence_score'],
            model_name=result['model_name']
        )
        result['record_id'] = record_id
    except Exception as e:
        print(f"Veritabanina kayit yapilamadi: {e}")
        result['record_id'] = None

    return result


if __name__ == '__main__':
    test_reviews = [
        "This product is amazing! I love it so much!",
        "Terrible quality. Waste of money. Very disappointed.",
        "It works fine, nothing special but does the job."
    ]

    for review in test_reviews:
        print(f"\nYorum: {review}")
        result = predict_sentiment(review)
        print(f"Tahmin: {result['predicted_sentiment']}")
        print(f"Guven: {result['confidence_score']:.2f}")
