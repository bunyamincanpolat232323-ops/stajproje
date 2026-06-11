"""
NLP On Isleme Modulu
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Bu modul Amazon Reviews veri setini yukler, temizler ve
NLP on isleme adimlarini uygular. Veri seti FastText formatinda
train.ft.txt ve test.ft.txt dosyalarindan okunur.
"""

import pandas as pd
import numpy as np
import re
import string
import os
import ssl

# SSL sertifika dogrulama sorununu coz (macOS)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
warnings.filterwarnings('ignore')

# NLTK verilerini indir
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Proje kok dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_fasttext_line(line):
    """
    FastText formatindaki bir satiri parse eder.
    Format: __label__X metin...
    __label__1 = negatif, __label__2 = pozitif

    Args:
        line (str): FastText formatinda satir

    Returns:
        tuple: (sentiment, text) veya (None, None) hata durumunda
    """
    line = line.strip()
    if not line:
        return None, None

    if line.startswith('__label__2 '):
        return 'pozitif', line[len('__label__2 '):]
    elif line.startswith('__label__1 '):
        return 'negatif', line[len('__label__1 '):]
    else:
        return None, None


def load_dataset(train_path=None, test_path=None, sample_size=10000):
    """
    Amazon Reviews veri setini FastText formatindan yukler.
    Buyuk dosyalardan rastgele ornekleme yapar.

    Args:
        train_path (str): Egitim dosyasi yolu
        test_path (str): Test dosyasi yolu
        sample_size (int): Her dosyadan alinacak ornek sayisi

    Returns:
        tuple: (train_df, test_df) DataFrame'leri
    """
    if train_path is None:
        train_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.ft.txt')
    if test_path is None:
        test_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test.ft.txt')

    print("Veri seti yukleniyor...")

    # Egitim verisini oku
    train_df = _load_fasttext_file(train_path, sample_size, 'egitim')

    # Test verisini oku
    test_df = _load_fasttext_file(test_path, sample_size // 4, 'test')

    return train_df, test_df


def _load_fasttext_file(filepath, sample_size, label):
    """
    Tek bir FastText dosyasini okur ve ornekler.

    Args:
        filepath (str): Dosya yolu
        sample_size (int): Alinacak ornek sayisi
        label (str): Log icin etiket

    Returns:
        pd.DataFrame: Yuklenen veri
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadi: {filepath}")

    # Dosya boyutunu kontrol et ve satirlari oku
    print(f"  {label} dosyasi okunuyor: {filepath}")

    # Toplam satir sayisini bul (hizli yontem)
    total_lines = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1

    print(f"  Toplam satir: {total_lines:,}")

    # Rastgele satirlari sec
    if total_lines <= sample_size:
        selected_indices = set(range(total_lines))
    else:
        rng = np.random.RandomState(42)
        selected_indices = set(rng.choice(total_lines, sample_size, replace=False))

    # Secilen satirlari oku
    sentiments = []
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in selected_indices:
                sentiment, text = parse_fasttext_line(line)
                if sentiment and text:
                    sentiments.append(sentiment)
                    texts.append(text)

    df = pd.DataFrame({'review_text': texts, 'sentiment': sentiments})
    print(f"  {label} seti: {len(df)} kayit yuklendi")
    return df


def check_missing_values(df):
    """
    Eksik ve null degerleri kontrol eder ve raporlar.

    Args:
        df (pd.DataFrame): Kontrol edilecek DataFrame

    Returns:
        pd.DataFrame: Temizlenmis DataFrame
    """
    print("\n--- Eksik Deger Kontrolu ---")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100

    missing_report = pd.DataFrame({
        'Eksik Deger Sayisi': missing,
        'Yuzde (%)': missing_pct.round(2)
    })
    print(missing_report)

    # Eksik degerleri temizle
    initial_len = len(df)
    df = df.dropna(subset=['review_text']).copy()
    removed = initial_len - len(df)

    if removed > 0:
        print(f"\n{removed} satir eksik yorum nedeniyle kaldirildi.")
    else:
        print("\nEksik deger bulunamadi.")

    return df


def clean_text(text):
    """
    Tek bir metin uzerinde temizleme islemi uygular.

    Adimlar:
        1. Kucuk harfe cevirme
        2. HTML etiketlerini kaldirma
        3. URL'leri kaldirma
        4. Sayilari kaldirma
        5. Noktalama isaretlerini kaldirma
        6. Fazla bosluklari temizleme

    Args:
        text (str): Temizlenecek metin

    Returns:
        str: Temizlenmis metin
    """
    if not isinstance(text, str):
        return ''

    # Kucuk harfe cevir
    text = text.lower()

    # HTML etiketlerini kaldir
    text = re.sub(r'<[^>]+>', '', text)

    # URL'leri kaldir
    text = re.sub(r'http\S+|www\S+', '', text)

    # Sayilari kaldir
    text = re.sub(r'\d+', '', text)

    # Noktalama isaretlerini kaldir
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Fazla bosluklari temizle
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):
    """
    Ingilizce stopword'leri metinden kaldirir.

    Args:
        text (str): Islenecek metin

    Returns:
        str: Stopword'ler kaldirilmis metin
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(filtered_tokens)


def preprocess_text(text):
    """
    Tam NLP on isleme pipeline'i uygular.

    Adimlar:
        1. Metin temizleme (kucuk harf, HTML, URL, sayi, noktalama)
        2. Stopword kaldirma
        3. Tokenization (kelime ayirma)

    Args:
        text (str): Islenecek ham metin

    Returns:
        str: On islemeden gecmis metin
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def apply_preprocessing(df, text_column='review_text'):
    """
    DataFrame'deki tum metinlere on isleme uygular.

    Args:
        df (pd.DataFrame): Islenecek DataFrame
        text_column (str): Metin sutununun adi

    Returns:
        pd.DataFrame: Islenmis DataFrame
    """
    print("\nNLP on isleme uygulaniyor...")
    df['cleaned_text'] = df[text_column].apply(preprocess_text)

    # Bos metinleri kaldir
    initial_len = len(df)
    df = df[df['cleaned_text'].str.strip().str.len() > 0].copy()
    removed = initial_len - len(df)
    if removed > 0:
        print(f"{removed} satir bos metin nedeniyle kaldirildi.")

    print(f"NLP on isleme tamamlandi. {len(df)} kayit islendi.")
    return df


def extract_features(texts, max_features=5000, vectorizer=None):
    """
    TF-IDF ile ozellik cikarimi yapar.

    TF-IDF (Term Frequency-Inverse Document Frequency):
        Bir kelimenin bir dokumandaki onemini olcen istatistiksel bir yontemdir.

    Args:
        texts (pd.Series): Metin serisi
        max_features (int): Maksimum ozellik sayisi
        vectorizer: Onceden egitilmis vectorizer (None ise yeni olusturulur)

    Returns:
        tuple: (TF-IDF matrisi, TfidfVectorizer nesnesi)
    """
    if vectorizer is None:
        print(f"\nTF-IDF ozellik cikarimi yapiliyor (max_features={max_features})...")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        tfidf_matrix = vectorizer.fit_transform(texts)

        # Vectorizer'i kaydet
        vectorizer_path = os.path.join(PROJECT_ROOT, 'models', 'tfidf_vectorizer.pkl')
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"TF-IDF Vectorizer kaydedildi: {vectorizer_path}")
    else:
        tfidf_matrix = vectorizer.transform(texts)

    print(f"TF-IDF matrisi olusturuldu: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def get_full_pipeline(sample_size=10000):
    """
    Tam veri on isleme pipeline'ini calistirir.

    Args:
        sample_size (int): Egitim verisinden alinacak ornek sayisi

    Returns:
        tuple: (train_df, test_df, X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print("=" * 60)
    print("  NLP ON ISLEME PIPELINE'I")
    print("  Bunyamin Canpolat - 230206063")
    print("=" * 60)

    # 1. Veri setini yukle
    train_df, test_df = load_dataset(sample_size=sample_size)

    # 2. Eksik degerleri kontrol et
    train_df = check_missing_values(train_df)
    test_df = check_missing_values(test_df)

    # 3. Duygu dagilimlari
    print("\n--- Egitim Seti Duygu Dagilimi ---")
    for sentiment, count in train_df['sentiment'].value_counts().items():
        pct = (count / len(train_df)) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")

    # 4. NLP on isleme uygula
    train_df = apply_preprocessing(train_df)
    test_df = apply_preprocessing(test_df)

    # 5. TF-IDF ozellik cikarimi
    X_train_tfidf, vectorizer = extract_features(train_df['cleaned_text'])
    X_test_tfidf, _ = extract_features(test_df['cleaned_text'], vectorizer=vectorizer)

    # Islenmis veriyi kaydet
    processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(processed_path, index=False)
    print(f"\nIslenmis veri kaydedildi: {processed_path}")

    return train_df, test_df, X_train_tfidf, X_test_tfidf, vectorizer


if __name__ == '__main__':
    train_df, test_df, X_train, X_test, vec = get_full_pipeline()
    print("\nOn isleme pipeline'i basariyla tamamlandi!")
