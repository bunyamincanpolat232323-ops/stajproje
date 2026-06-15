"""
Yardimci Fonksiyonlar
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi
"""

import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_processed_data():
    """Islenmis veri setini yukler."""
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def load_evaluation_results():
    """Model degerlendirme sonuclarini yukler."""
    json_path = os.path.join(PROJECT_ROOT, 'reports', 'model_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
