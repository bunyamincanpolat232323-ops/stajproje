"""
Model Egitim Modulu
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Bu modul 3 farkli makine ogrenmesi modelini egitir:
1. Logistic Regression
2. Naive Bayes (Multinomial)
3. Support Vector Machine (Linear SVM)

En iyi modeli otomatik secer ve kaydeder.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Proje kok dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import get_full_pipeline
from app.database import init_db, save_model_performance, save_reviews_to_db


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression modeli egitir.
    Metin siniflandirma gorevlerinde iyi performans gosteren
    dogrusal bir siniflandirma algoritmasidir.
    """
    print("\nLogistic Regression egitiliyor...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    print("Logistic Regression egitimi tamamlandi.")
    return model


def train_naive_bayes(X_train, y_train):
    """
    Multinomial Naive Bayes modeli egitir.
    Bayes teoremine dayanan, metin siniflandirma icin
    yaygin olarak kullanilan bir algoritmadir.
    """
    print("\nNaive Bayes egitiliyor...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    print("Naive Bayes egitimi tamamlandi.")
    return model


def train_svm(X_train, y_train):
    """
    Linear SVM (Support Vector Machine) modeli egitir.
    CalibratedClassifierCV ile sarmalanarak olasilik tahmini yapabilir.
    Veri noktalarini ayiran en iyi hiperduzlemi bulan bir algoritmadir.
    """
    print("\nLinear SVM egitiliyor...")
    base_svm = LinearSVC(
        random_state=42,
        max_iter=2000,
        C=1.0
    )
    # CalibratedClassifierCV ile predict_proba destegi ekle
    model = CalibratedClassifierCV(base_svm, cv=3)
    model.fit(X_train, y_train)
    print("Linear SVM egitimi tamamlandi.")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Egitilmis modeli test seti uzerinde degerlendirir.

    Hesaplanan metrikler:
    - Accuracy (Dogruluk)
    - Precision (Kesinlik)
    - Recall (Duyarlilik)
    - F1-Score
    - Confusion Matrix

    Args:
        model: Egitilmis model
        X_test: Test ozellikleri
        y_test: Test etiketleri
        model_name (str): Model adi

    Returns:
        dict: Performans metrikleri
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} Degerlendirmesi")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Siniflandirma Raporu:")
    print(report)

    return {
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
        'f1_weighted': f1_w,
        'confusion_matrix': conf_mat,
        'classification_report': report
    }


def select_best_model(results):
    """
    En iyi modeli F1-score (weighted) metrigine gore secer.
    """
    best_model_name = max(results, key=lambda x: results[x]['f1_weighted'])
    best_f1 = results[best_model_name]['f1_weighted']
    print(f"\n{'=' * 60}")
    print(f"  EN IYI MODEL: {best_model_name}")
    print(f"  F1-Score (Weighted): {best_f1:.4f}")
    print(f"{'=' * 60}")
    return best_model_name


def save_model(model, model_name, is_best=False):
    """
    Egitilmis modeli dosyaya kaydeder.
    """
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    filename = f"{model_name.lower().replace(' ', '_')}.pkl"
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model kaydedildi: {filepath}")

    if is_best:
        best_path = os.path.join(models_dir, 'best_model.pkl')
        joblib.dump(model, best_path)

        info = {
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'file': filename
        }
        info_path = os.path.join(models_dir, 'best_model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"En iyi model kaydedildi: {best_path}")


def save_reports(results, best_model_name):
    """
    Model metriklerini CSV, JSON ve Markdown formatlarinda reports/ klasorune kaydeder.
    """
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # JSON rapor
    json_path = os.path.join(reports_dir, 'model_metrics.json')
    report_data = {
        'best_model': best_model_name,
        'trained_at': datetime.now().isoformat(),
        'results': results
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # CSV rapor
    csv_path = os.path.join(reports_dir, 'model_metrics.csv')
    rows = []
    for name, metrics in results.items():
        rows.append({
            'Model': name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision_macro'], 4),
            'Recall': round(metrics['recall_macro'], 4),
            'F1-Score': round(metrics['f1_macro'], 4),
            'F1-Weighted': round(metrics['f1_weighted'], 4),
            'En Iyi': 'Evet' if name == best_model_name else 'Hayir'
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Markdown rapor
    md_path = os.path.join(reports_dir, 'model_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Model Degerlendirme Raporu\n\n")
        f.write(f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**En Iyi Model:** {best_model_name}\n\n")
        f.write("## Performans Metrikleri\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        for row in rows:
            f.write(f"| {row['Model']} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} | {row['F1-Score']} |\n")
        f.write("\n## Siniflandirma Raporlari\n\n")
        for name, metrics in results.items():
            f.write(f"### {name}\n\n```\n{metrics['classification_report']}\n```\n\n")

    # Confusion matrix JSON
    conf_path = os.path.join(reports_dir, 'confusion_matrices.json')
    conf_data = {name: metrics['confusion_matrix'] for name, metrics in results.items()}
    with open(conf_path, 'w') as f:
        json.dump(conf_data, f, indent=2)

    print(f"\nRaporlar kaydedildi: {reports_dir}")


def main():
    """
    Ana egitim pipeline'i.
    """
    print("\n" + "=" * 60)
    print("  DUYGU ANALIZI - MODEL EGITIM PIPELINE'I")
    print("  Bunyamin Canpolat - 230206063")
    print("  Ostim Teknik Universitesi")
    print("=" * 60)

    # 1. Veri on isleme
    train_df, test_df, X_train, X_test, vectorizer = get_full_pipeline(sample_size=10000)

    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    print(f"\nEgitim seti: {X_train.shape[0]} ornek")
    print(f"Test seti: {X_test.shape[0]} ornek")

    # Etiket bilgisini kaydet
    labels = sorted(train_df['sentiment'].unique().tolist())
    labels_path = os.path.join(PROJECT_ROOT, 'models', 'labels.json')
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, 'w') as f:
        json.dump(labels, f)

    # 2. Modelleri egit
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Naive Bayes': train_naive_bayes(X_train, y_train),
        'Linear SVM': train_svm(X_train, y_train)
    }

    # 3. Modelleri degerlendir
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    # 4. En iyi modeli sec
    best_model_name = select_best_model(results)

    # 5. Modelleri kaydet
    for name, model in models.items():
        save_model(model, name, is_best=(name == best_model_name))

    # 6. Raporlari kaydet
    save_reports(results, best_model_name)

    # 7. Veritabani islemleri
    init_db()

    # Model performanslarini veritabanina kaydet
    for name, metrics in results.items():
        save_model_performance(
            model_name=name,
            accuracy=metrics['accuracy'],
            precision=metrics['precision_macro'],
            recall=metrics['recall_macro'],
            f1_score_val=metrics['f1_macro'],
            confusion_mat=metrics['confusion_matrix']
        )

    # Islenmis verileri reviews tablosuna kaydet
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    save_reviews_to_db(combined_df)

    print(f"\n{'=' * 60}")
    print(f"  EGITIM TAMAMLANDI!")
    print(f"  En iyi model: {best_model_name}")
    print(f"  Dashboard'u baslatmak icin:")
    print(f"    streamlit run app/dashboard.py")
    print(f"{'=' * 60}")

    return models, results, best_model_name


if __name__ == '__main__':
    models, results, best = main()
