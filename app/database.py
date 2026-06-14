"""
Veritabani Modulu
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Bu modul SQLite veritabani islemlerini yonetir.
SQLAlchemy ORM kullanilarak veritabani tablolari olusturulur.
PostgreSQL'e gecis icin sadece DATABASE_URL degistirilmesi yeterlidir.
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Proje kok dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Veritabani yolu
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'sentiment_analysis.db')
DATABASE_URL = f'sqlite:///{DB_PATH}'

# SQLAlchemy ayarlari
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


# ---- TABLO TANIMLARI ----

class Review(Base):
    """
    Veri setindeki yorumlarin saklandigi tablo.
    Yorumlarin hem ham hem temizlenmis hali ve duygu etiketleri tutulur.
    """
    __tablename__ = 'reviews'

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    actual_sentiment = Column(String(20), nullable=True)
    predicted_sentiment = Column(String(20), nullable=True)
    confidence_score = Column(Float, nullable=True)
    source = Column(String(50), default='dataset')
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Review(id={self.id}, sentiment='{self.actual_sentiment}')>"


class Prediction(Base):
    """
    Kullanicilarin dashboard'dan girdigi yorumlarin tahmin sonuclarini saklayan tablo.
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_text = Column(Text, nullable=False)
    predicted_sentiment = Column(String(20), nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    classification_timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction(id={self.id}, sentiment='{self.predicted_sentiment}', score={self.confidence_score:.2f})>"


class ModelMetric(Base):
    """
    Model performans sonuclarini saklayan tablo.
    Her model icin accuracy, precision, recall ve f1 degerleri tutulur.
    """
    __tablename__ = 'model_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelMetric(model='{self.model_name}', accuracy={self.accuracy:.4f})>"


# ---- VERITABANI ISLEMLERI ----

def init_db():
    """Veritabanini baslatir ve tablolari olusturur."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        Base.metadata.create_all(engine)
        print(f"Veritabani baslatildi: {DB_PATH}")
    except Exception as e:
        print(f"Veritabani baslatma hatasi: {e}")
        raise


def get_session():
    """Yeni bir veritabani oturumu olusturur."""
    return SessionLocal()


def save_reviews_to_db(df):
    """
    Islenmis DataFrame'i reviews tablosuna kaydeder.
    Mevcut verileri temizleyip yeniden yazar.
    """
    session = get_session()
    try:
        # Mevcut verileri temizle
        session.query(Review).delete()
        session.commit()

        records = []
        for _, row in df.iterrows():
            record = Review(
                review_text=row.get('review_text', ''),
                cleaned_text=row.get('cleaned_text', ''),
                actual_sentiment=row.get('sentiment', ''),
                source='dataset',
                created_at=datetime.utcnow()
            )
            records.append(record)

        session.bulk_save_objects(records)
        session.commit()
        print(f"{len(records)} yorum reviews tablosuna kaydedildi.")
    except Exception as e:
        session.rollback()
        print(f"Reviews kaydetme hatasi: {e}")
    finally:
        session.close()


def save_prediction(review_text, predicted_sentiment, confidence_score, model_name):
    """
    Dashboard'dan yapilan bir tahmini predictions tablosuna kaydeder.

    Returns:
        int: Kaydedilen tahmin ID'si
    """
    session = get_session()
    try:
        record = Prediction(
            review_text=review_text,
            predicted_sentiment=predicted_sentiment,
            confidence_score=confidence_score,
            model_name=model_name,
            classification_timestamp=datetime.utcnow()
        )
        session.add(record)
        session.commit()
        record_id = record.id
        return record_id
    except Exception as e:
        session.rollback()
        print(f"Tahmin kaydetme hatasi: {e}")
        return None
    finally:
        session.close()


def save_model_performance(model_name, accuracy, precision, recall,
                           f1_score_val, confusion_mat=None):
    """Model performans sonuclarini model_metrics tablosuna kaydeder."""
    session = get_session()
    try:
        record = ModelMetric(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score_val,
            created_at=datetime.utcnow()
        )
        session.add(record)
        session.commit()
        print(f"{model_name} performans sonuclari veritabanina kaydedildi.")
    except Exception as e:
        session.rollback()
        print(f"Performans kaydetme hatasi: {e}")
    finally:
        session.close()


def get_all_predictions():
    """Tum tahmin kayitlarini getirir (en yeni en uste)."""
    session = get_session()
    try:
        records = session.query(Prediction).order_by(
            Prediction.classification_timestamp.desc()
        ).all()
        return records
    except Exception as e:
        print(f"Tahmin sorgulama hatasi: {e}")
        return []
    finally:
        session.close()


def get_all_reviews():
    """Reviews tablosundan tum kayitlari getirir."""
    session = get_session()
    try:
        records = session.query(Review).all()
        return records
    except Exception as e:
        print(f"Reviews sorgulama hatasi: {e}")
        return []
    finally:
        session.close()


def get_model_performances():
    """Tum model performans sonuclarini getirir."""
    session = get_session()
    try:
        records = session.query(ModelMetric).order_by(
            ModelMetric.created_at.desc()
        ).all()
        return records
    except Exception as e:
        print(f"Performans sorgulama hatasi: {e}")
        return []
    finally:
        session.close()


def get_prediction_stats():
    """Tahmin istatistiklerini getirir."""
    session = get_session()
    try:
        total = session.query(Prediction).count()
        stats = {
            'total_predictions': total,
            'sentiments': {}
        }
        for sentiment in ['pozitif', 'negatif']:
            count = session.query(Prediction).filter(
                Prediction.predicted_sentiment == sentiment
            ).count()
            stats['sentiments'][sentiment] = count
        return stats
    except Exception as e:
        print(f"Istatistik sorgulama hatasi: {e}")
        return {'total_predictions': 0, 'sentiments': {}}
    finally:
        session.close()


def get_review_count():
    """Reviews tablosundaki toplam kayit sayisini dondurur."""
    session = get_session()
    try:
        return session.query(Review).count()
    except Exception:
        return 0
    finally:
        session.close()


def get_review_sentiment_counts():
    """Reviews tablosundaki duygu dagilimini dondurur."""
    session = get_session()
    try:
        counts = {}
        for sentiment in ['pozitif', 'negatif']:
            count = session.query(Review).filter(
                Review.actual_sentiment == sentiment
            ).count()
            counts[sentiment] = count
        return counts
    except Exception:
        return {}
    finally:
        session.close()


if __name__ == '__main__':
    init_db()
    print("Veritabani tablolari basariyla olusturuldu!")
    print(f"  - reviews")
    print(f"  - predictions")
    print(f"  - model_metrics")
