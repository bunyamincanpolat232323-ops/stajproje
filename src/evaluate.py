"""
Model Degerlendirme Modulu
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Bu modul egitilmis modellerin performansini olcer.
Accuracy, Precision, Recall, F1-Score ve Confusion Matrix hesaplar.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Modeli test seti uzerinde degerlendirir.

    Args:
        model: Egitilmis model
        X_test: Test ozellikleri
        y_test: Test etiketleri
        model_name (str): Model adi

    Returns:
        dict: Performans metrikleri
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }

    print(f"\n{model_name} Degerlendirmesi")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall   : {metrics['recall_macro']:.4f}")
    print(f"  F1-Score : {metrics['f1_macro']:.4f}")

    return metrics
