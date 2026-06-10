# Duygu Analizi Gosterge Paneli

**Proje 2 - NLP Projesi: Duygu Analizi Gosterge Paneli**

Bunyamin Canpolat - 230206063  
Ostim Teknik Universitesi - Bilgisayar Muhendisligi  
IYD 328 - Is Yeri Deneyimi - 2025-2026 Bahar Donemi

---

## Proje Aciklamasi

Bu proje, Amazon urun yorumlarini otomatik olarak Pozitif veya Negatif olarak siniflandiran uctan uca bir duygu analizi sistemidir. Sistem, makine ogrenmesi modelleri ile tahmin yapar, sonuclari SQL veritabaninda saklar ve kullanici dostu bir gosterge paneli uzerinden sonuclari gorsellestirir.

## Sistem Tasarimi

Proje uc katmanli bir mimariye sahiptir:

1. **Veri Katmani**: Amazon Reviews veri seti islenip SQL veritabaninda saklanir.
2. **Model Katmani**: 3 farkli makine ogrenmesi modeli egitilir, karsilastirilir ve en iyisi secilir.
3. **Sunum Katmani**: Streamlit dashboard ile kullaniciya gosterge paneli sunulur.

```
Kullanici --> Streamlit Dashboard --> ML Model --> SQL Veritabani
                                        |
                              TF-IDF Vectorizer
                                        |
                              NLP Preprocessing
```

## Veri Seti Aciklamasi

- **Kaynak**: Amazon Reviews (FastText formati)
- **Dosyalar**: `train.ft.txt` (3,600,000 yorum) ve `test.ft.txt` (400,000 yorum)
- **Format**: Her satir `__label__X metin` formatinda
  - `__label__1`: Negatif yorum
  - `__label__2`: Pozitif yorum
- **Ornekleme**: Buyuk veri setinden rastgele 10,000 egitim + 2,500 test ornegi alinir

## Kullanilan Teknolojiler

| Teknoloji | Kullanim Alani |
|-----------|---------------|
| Python 3 | Ana programlama dili |
| Streamlit | Gosterge paneli (dashboard) |
| scikit-learn | Makine ogrenmesi modelleri |
| pandas / numpy | Veri isleme |
| Plotly | Grafik ve gorsellestirme |
| NLTK | NLP on isleme |
| SQLAlchemy | Veritabani ORM |
| SQLite | SQL veritabani |
| Docker | Konteynerizasyon |
| joblib | Model kaydetme/yukleme |

## Klasor Yapisi

```
sentiment-dashboard/
├── app/
│   ├── __init__.py
│   ├── dashboard.py        # Streamlit gosterge paneli
│   ├── database.py         # SQLAlchemy veritabani islemleri
│   ├── predict.py          # Tahmin motoru
│   └── utils.py            # Yardimci fonksiyonlar
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # NLP on isleme pipeline
│   ├── train_models.py     # Model egitim pipeline
│   ├── evaluate.py         # Model degerlendirme
│   └── visualizations.py   # Plotly gorsellestirme
├── data/
│   ├── raw/                # Ham veri dosyalari
│   │   ├── train.ft.txt
│   │   └── test.ft.txt
│   ├── processed_data.csv  # Islenmis veri
│   └── sentiment_analysis.db  # SQLite veritabani
├── models/
│   ├── best_model.pkl      # En iyi model
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── linear_svm.pkl
├── reports/
│   ├── model_metrics.json
│   ├── model_metrics.csv
│   ├── model_report.md
│   └── confusion_matrices.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## NLP On Isleme Adimlari

Metin verileri modele verilmeden once su adimlardan gecirilir:

1. **Kucuk harfe donusturme**: Tum metin lowercase yapilir
2. **HTML etiketlerini kaldirma**: `<br>`, `<p>` gibi tagler temizlenir
3. **URL temizleme**: Web adresleri cikarilir
4. **Sayi temizleme**: Rakamlar kaldirilir
5. **Noktalama temizleme**: Ozel karakterler ve noktalama isaretleri cikarilir
6. **Stopword kaldirma**: NLTK ile Ingilizce stopword'ler filtrelenir
7. **Tokenization**: Metin kelimelere ayrilir
8. **TF-IDF Ozellik Cikarimi**: Metin sayisal vektore donusturulur (max 5000 ozellik, unigram+bigram)

## Model Secim Gerekcesi

Uc farkli model egitilmistir:

1. **Logistic Regression**: Metin siniflandirmada basit ama etkili dogrusal bir model. Hizli egitim suresi ve yorumlanabilirligi ile tercih edilmistir.

2. **Multinomial Naive Bayes**: Bayes teoremine dayanan, ozellikle metin verilerinde iyi sonuc veren olasiliksal bir model. Kelime frekanslarini kullanir.

3. **Linear SVM (Support Vector Machine)**: Veri noktalarini en iyi sekilde ayiran hiperduzlemi bulan guclu bir siniflandirici. CalibratedClassifierCV ile olasilik tahmin destegi eklenmistir.

**Secim Kriteri**: F1-Score (weighted) metrigine gore en yuksek performansi gosteren model otomatik olarak secilir.

## Egitim Sureci

1. Veri seti FastText formatindan okunur ve DataFrame'e donusturulur
2. NLP on isleme pipeline uygulanir
3. TF-IDF ile ozellik cikarimi yapilir
4. 3 model sirasiyla egitilir
5. Her model test seti uzerinde degerlendirilir
6. En iyi model secilir ve `models/best_model.pkl` olarak kaydedilir
7. Tum sonuclar `reports/` klasorune ve SQL veritabanina yazilir

## Degerlendirme Sonuclari

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8564 | 0.8564 | 0.8564 | 0.8564 |
| Naive Bayes | 0.8488 | 0.8493 | 0.8487 | 0.8487 |
| **Linear SVM** | **0.8592** | **0.8592** | **0.8592** | **0.8592** |

En iyi model: **Linear SVM** (F1-Score: 0.8592)

## SQL Veritabani Yapisi

Veritabani SQLite uzerinde calisir. SQLAlchemy ORM kullanildigi icin PostgreSQL'e gecis kolaydir.

### reviews tablosu
| Sutun | Tip | Aciklama |
|-------|-----|----------|
| id | INTEGER | Primary Key |
| review_text | TEXT | Yorum metni |
| cleaned_text | TEXT | Temizlenmis metin |
| actual_sentiment | VARCHAR(20) | Gercek duygu etiketi |
| predicted_sentiment | VARCHAR(20) | Tahmin edilen duygu |
| confidence_score | FLOAT | Guven skoru |
| source | VARCHAR(50) | Veri kaynagi |
| created_at | DATETIME | Kayit zamani |

### predictions tablosu
| Sutun | Tip | Aciklama |
|-------|-----|----------|
| id | INTEGER | Primary Key |
| review_text | TEXT | Yorum metni |
| predicted_sentiment | VARCHAR(20) | Tahmin edilen duygu |
| confidence_score | FLOAT | Guven skoru |
| model_name | VARCHAR(100) | Kullanilan model |
| classification_timestamp | DATETIME | Siniflandirma zamani |

### model_metrics tablosu
| Sutun | Tip | Aciklama |
|-------|-----|----------|
| id | INTEGER | Primary Key |
| model_name | VARCHAR(100) | Model adi |
| accuracy | FLOAT | Dogruluk |
| precision | FLOAT | Kesinlik |
| recall | FLOAT | Duyarlilik |
| f1_score | FLOAT | F1 Skoru |
| created_at | DATETIME | Kayit zamani |

## Dashboard Ozellikleri

Gosterge paneli Streamlit ile gelistirilmis olup su sekmeleri icerir:

1. **Ana Sayfa**: Veri seti ozeti, yorum girisi ve anlik duygu tahmini
2. **Duygu Dagilimlari**: Pasta ve cubuk grafikleri ile duygu yuzdeleri
3. **Kelime Analizi**: En sik kullanilan kelimeler ve frekans analizi
4. **Model Karsilastirma**: 3 modelin performans karsilastirmasi ve confusion matrix
5. **Tahmin Gecmisi**: Kullanicinin girdigi yorumlarin gecmis kayitlari
6. **Veritabani**: SQL tablolari ve kayit bilgileri

## Kurulum Talimatlari

### Gereksinimler
- Python 3.10+
- pip

### Bagimliliklari Yukle
```bash
pip install -r requirements.txt
```

### Modeli Egit
```bash
python src/train_models.py
```

### Dashboard'u Baslat
```bash
streamlit run app/dashboard.py
```

Tarayicinizda `http://localhost:8501` adresini acin.

## Docker ile Calistirma

```bash
docker-compose up --build
```

Tarayicinizda `http://localhost:8501` adresini acin.

## Gosterge Paneli Ekran Goruntuleri

(Dashboard calistirildiktan sonra ekran goruntuleri buraya eklenecektir.)

## GitHub Teslim Notlari

- Proje tamamen calisir durumdadir
- Tum modeller egitilmis ve reports/ klasorune raporlanmistir
- SQL veritabani entegrasyonu aktiftir
- Dashboard localhost:8501 uzerinden erisilebilir
- Docker ile konteynerizasyon destegi mevcuttur
