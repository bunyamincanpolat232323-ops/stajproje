"""
Duygu Analizi Gosterge Paneli (Dashboard)
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Streamlit ile kullanici dostu gosterge paneli.
Ana ekranda yorum girisi, ustte sekme butonlari ile
farkli analizlere erisim.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
from datetime import datetime
from collections import Counter

# Proje kok dizini ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.database import (
    init_db, get_all_predictions, get_prediction_stats,
    get_review_count, get_review_sentiment_counts,
    get_model_performances
)

# Sayfa yapilandirmasi
st.set_page_config(
    page_title="Duygu Analizi Gosterge Paneli - Bunyamin Canpolat",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sidebar'i gizle
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Veritabanini baslat
init_db()


# ---- VERI YUKLEME ----

@st.cache_data
def load_processed_data():
    """Islenmis veri setini yukler."""
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_evaluation_results():
    """Model degerlendirme sonuclarini yukler."""
    json_path = os.path.join(PROJECT_ROOT, 'reports', 'model_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@st.cache_resource
def load_ml_model():
    """Model ve vectorizer'i yukler."""
    try:
        from app.predict import load_model, load_vectorizer
        model, model_name = load_model()
        vectorizer = load_vectorizer()
        return model, model_name, vectorizer
    except Exception:
        return None, None, None


# ---- SAYFA FONKSIYONLARI ----

def render_header():
    """Sayfa basligini olusturur."""
    st.markdown("# Duygu Analizi Gosterge Paneli")
    st.markdown("**Proje 2 - NLP Projesi** | Bunyamin Canpolat - 230206063 | Ostim Teknik Universitesi")
    st.markdown("---")


def render_main_page():
    """
    Ana sayfa: Yorum girisi ve genel istatistikler.
    Kullanici buradan yorum yazip aninda tahmin alabilir.
    """
    df = load_processed_data()
    model, model_name, vectorizer = load_ml_model()

    # -- Genel istatistikler --
    st.subheader("Veri Seti Ozeti")

    if df is not None:
        total = len(df)
        sentiment_counts = df['sentiment'].value_counts()
        pos_count = sentiment_counts.get('pozitif', 0)
        neg_count = sentiment_counts.get('negatif', 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Yorum", f"{total:,}")
        col2.metric("Pozitif Yorum", f"{pos_count:,}")
        col3.metric("Negatif Yorum", f"{neg_count:,}")
    else:
        st.warning("Veri seti bulunamadi. Lutfen once modeli egitin: python src/train_models.py")

    st.markdown("---")

    # -- Yorum girisi ve tahmin --
    st.subheader("Yeni Yorum Analizi")
    st.write("Asagiya bir yorum yazin ve 'Analiz Et' butonuna basin. "
             "Sistem yorumunuzu analiz edip duygu tahmini yapacaktir.")

    if model is None:
        st.error("Model yuklenemedi. Lutfen once egitim yapin: python src/train_models.py")
        return

    user_review = st.text_area(
        "Yorumunuzu yazin (Ingilizce):",
        placeholder="Ornek: This product is really great, I love using it every day.",
        height=100
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_button = st.button("Analiz Et", type="primary")

    if analyze_button:
        if user_review and user_review.strip():
            with st.spinner("Analiz yapiliyor..."):
                from app.predict import predict_and_save
                result = predict_and_save(
                    user_review,
                    model=model,
                    vectorizer=vectorizer,
                    model_name=model_name
                )

            sentiment = result['predicted_sentiment']
            confidence = result['confidence_score']

            st.markdown("---")
            st.subheader("Analiz Sonucu")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Tahmin Edilen Duygu", sentiment.upper())
            res_col2.metric("Guven Skoru", f"{confidence * 100:.1f}%")
            res_col3.metric("Kullanilan Model", model_name)

            # Olasilik dagilimi
            if result.get('probabilities'):
                st.write("**Olasilik Dagilimi:**")
                prob_df = pd.DataFrame([
                    {'Duygu': k.capitalize(), 'Olasilik': round(v, 4)}
                    for k, v in result['probabilities'].items()
                ])
                st.dataframe(prob_df, hide_index=True, width=400)

            if result.get('record_id'):
                st.success(f"Tahmin veritabanina kaydedildi. (Kayit No: {result['record_id']})")
        else:
            st.warning("Lutfen bir yorum yazin.")


def render_sentiment_distribution():
    """Duygu dagilim grafikleri."""
    st.subheader("Duygu Dagilimlari")

    df = load_processed_data()
    if df is None:
        st.warning("Veri seti bulunamadi.")
        return

    sentiment_counts = df['sentiment'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        # Pasta grafigi
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Duygu Dagilimi (Pasta Grafigi)",
            color=sentiment_counts.index,
            color_discrete_map={'pozitif': '#2ecc71', 'negatif': '#e74c3c'}
        )
        fig_pie.update_layout(template='plotly_white')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Cubuk grafigi
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Duygu Dagilimi (Cubuk Grafigi)",
            labels={'x': 'Duygu', 'y': 'Yorum Sayisi'},
            color=sentiment_counts.index,
            color_discrete_map={'pozitif': '#2ecc71', 'negatif': '#e74c3c'}
        )
        fig_bar.update_layout(template='plotly_white', showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Yuzde tablosu
    st.write("**Duygu Yuzdeleri:**")
    pct_data = []
    for sentiment, count in sentiment_counts.items():
        pct_data.append({
            'Duygu': sentiment.capitalize(),
            'Sayi': count,
            'Yuzde': f"{count / len(df) * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(pct_data), hide_index=True, width=500)


def render_word_analysis():
    """Kelime frekans analizi."""
    st.subheader("Kelime Frekans Analizi")

    df = load_processed_data()
    if df is None or 'cleaned_text' not in df.columns:
        st.warning("Temizlenmis metin verisi bulunamadi.")
        return

    # Tum kelimeleri topla
    all_words = ' '.join(df['cleaned_text'].dropna()).split()
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(25)

    if not top_words:
        st.info("Kelime verisi bulunamadi.")
        return

    words_df = pd.DataFrame(top_words, columns=['Kelime', 'Frekans'])

    fig = px.bar(
        words_df,
        x='Frekans',
        y='Kelime',
        orientation='h',
        title="En Sik Kullanilan 25 Kelime (Stopword'ler Haric)",
        color='Frekans',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'},
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Duygu bazli kelime analizi
    st.markdown("---")
    st.write("**Duyguya Gore En Sik Kelimeler:**")

    for sentiment in ['pozitif', 'negatif']:
        subset = df[df['sentiment'] == sentiment]
        if len(subset) == 0:
            continue

        words = ' '.join(subset['cleaned_text'].dropna()).split()
        freq = Counter(words).most_common(15)

        if freq:
            freq_df = pd.DataFrame(freq, columns=['Kelime', 'Frekans'])
            fig_s = px.bar(
                freq_df,
                x='Kelime',
                y='Frekans',
                title=f"{sentiment.capitalize()} Yorumlarda En Sik 15 Kelime",
                color_discrete_sequence=['#2ecc71' if sentiment == 'pozitif' else '#e74c3c']
            )
            fig_s.update_layout(template='plotly_white')
            st.plotly_chart(fig_s, use_container_width=True)


def render_model_comparison():
    """Model karsilastirma grafikleri ve confusion matrix."""
    st.subheader("Model Karsilastirma")

    eval_results = load_evaluation_results()
    if eval_results is None:
        st.warning("Model degerlendirme sonuclari bulunamadi. Lutfen once egitim yapin.")
        return

    best_model = eval_results.get('best_model', '')
    results = eval_results.get('results', {})

    st.info(f"En iyi model: **{best_model}**")

    # Metrik karsilastirma tablosu
    st.write("**Performans Metrikleri:**")
    metrics_data = []
    for name, metrics in results.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision_macro']:.4f}",
            'Recall': f"{metrics['recall_macro']:.4f}",
            'F1-Score': f"{metrics['f1_macro']:.4f}",
            'En Iyi': 'Evet' if name == best_model else ''
        })
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

    # Karsilastirma grafigi
    chart_data = []
    for name, metrics in results.items():
        for metric_name, metric_key in [('Accuracy', 'accuracy'), ('Precision', 'precision_macro'),
                                         ('Recall', 'recall_macro'), ('F1-Score', 'f1_macro')]:
            chart_data.append({
                'Model': name,
                'Metrik': metric_name,
                'Deger': metrics[metric_key]
            })

    chart_df = pd.DataFrame(chart_data)
    fig = px.bar(
        chart_df,
        x='Metrik',
        y='Deger',
        color='Model',
        barmode='group',
        title="Model Performans Karsilastirmasi",
        text=chart_df['Deger'].apply(lambda x: f'{x:.3f}')
    )
    fig.update_layout(template='plotly_white', yaxis=dict(range=[0, 1]))
    fig.update_traces(textposition='auto')
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown("---")
    st.subheader("Confusion Matrix")

    labels_path = os.path.join(PROJECT_ROOT, 'models', 'labels.json')
    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    except FileNotFoundError:
        labels = ['negatif', 'pozitif']

    cols = st.columns(len(results))
    for i, (name, metrics) in enumerate(results.items()):
        with cols[i]:
            conf_mat = metrics.get('confusion_matrix', [])
            if conf_mat:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=conf_mat,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    text=conf_mat,
                    texttemplate='%{text}',
                    showscale=False
                ))
                fig_cm.update_layout(
                    title=name,
                    xaxis_title='Tahmin',
                    yaxis_title='Gercek',
                    template='plotly_white',
                    height=350
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # Siniflandirma raporlari
    with st.expander("Detayli Siniflandirma Raporlari"):
        for name, metrics in results.items():
            st.write(f"**{name}:**")
            st.text(metrics.get('classification_report', 'Rapor bulunamadi.'))
            st.markdown("---")


def render_prediction_history():
    """Gecmis tahminler tablosu."""
    st.subheader("Tahmin Gecmisi")

    predictions = get_all_predictions()

    if not predictions:
        st.info("Henuz tahmin yapilmamis. Ana sayfadan bir yorum yazarak ilk tahminizi yapin.")
        return

    # Istatistikler
    stats = get_prediction_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Tahmin", stats['total_predictions'])
    col2.metric("Pozitif", stats['sentiments'].get('pozitif', 0))
    col3.metric("Negatif", stats['sentiments'].get('negatif', 0))

    st.markdown("---")

    # Tahmin tablosu
    records_data = []
    for pred in predictions:
        records_data.append({
            'No': pred.id,
            'Yorum': pred.review_text[:100] + '...' if len(pred.review_text) > 100 else pred.review_text,
            'Duygu': pred.predicted_sentiment.capitalize(),
            'Guven Skoru': f"{pred.confidence_score:.1%}",
            'Model': pred.model_name,
            'Tarih': pred.classification_timestamp.strftime('%Y-%m-%d %H:%M') if pred.classification_timestamp else '-'
        })

    st.dataframe(pd.DataFrame(records_data), hide_index=True, use_container_width=True)


def render_database_info():
    """Veritabani bilgileri sayfasi."""
    st.subheader("SQL Veritabani Bilgileri")

    db_path = os.path.join(PROJECT_ROOT, 'data', 'sentiment_analysis.db')
    st.write(f"**Veritabani:** SQLite")
    st.write(f"**Dosya:** {db_path}")
    st.write(f"**ORM:** SQLAlchemy")

    st.markdown("---")

    # Tablo bilgileri
    st.write("**Tablolar:**")

    review_count = get_review_count()
    prediction_count = get_prediction_stats()['total_predictions']
    model_perfs = get_model_performances()

    table_info = [
        {'Tablo': 'reviews', 'Aciklama': 'Veri setindeki yorumlar', 'Kayit Sayisi': review_count},
        {'Tablo': 'predictions', 'Aciklama': 'Kullanici tahminleri', 'Kayit Sayisi': prediction_count},
        {'Tablo': 'model_metrics', 'Aciklama': 'Model performans sonuclari', 'Kayit Sayisi': len(model_perfs)}
    ]
    st.dataframe(pd.DataFrame(table_info), hide_index=True, width=600)

    # Reviews tablosu ornegi
    st.markdown("---")
    st.write("**Reviews Tablosu (ilk 10 kayit):**")
    from app.database import get_all_reviews
    reviews = get_all_reviews()
    if reviews:
        review_data = []
        for r in reviews[:10]:
            review_data.append({
                'ID': r.id,
                'Yorum': r.review_text[:80] + '...' if len(r.review_text) > 80 else r.review_text,
                'Duygu': r.actual_sentiment,
                'Kaynak': r.source
            })
        st.dataframe(pd.DataFrame(review_data), hide_index=True, use_container_width=True)
    else:
        st.info("Reviews tablosu bos. Egitim yapildiktan sonra veriler yuklenecektir.")

    # Model metrikleri
    if model_perfs:
        st.markdown("---")
        st.write("**Model Metrikleri Tablosu:**")
        perf_data = []
        for p in model_perfs:
            perf_data.append({
                'Model': p.model_name,
                'Accuracy': f"{p.accuracy:.4f}",
                'Precision': f"{p.precision:.4f}",
                'Recall': f"{p.recall:.4f}",
                'F1-Score': f"{p.f1_score:.4f}"
            })
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, width=700)


# ---- ANA UYGULAMA ----

def main():
    """Ana uygulama fonksiyonu."""
    render_header()

    # Ust kisimda sekme butonlari
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Ana Sayfa",
        "Duygu Dagilimlari",
        "Kelime Analizi",
        "Model Karsilastirma",
        "Tahmin Gecmisi",
        "Veritabani"
    ])

    with tab1:
        render_main_page()

    with tab2:
        render_sentiment_distribution()

    with tab3:
        render_word_analysis()

    with tab4:
        render_model_comparison()

    with tab5:
        render_prediction_history()

    with tab6:
        render_database_info()

    # Alt bilgi
    st.markdown("---")
    st.markdown(
        "Bunyamin Canpolat | 230206063 | Ostim Teknik Universitesi - Bilgisayar Muhendisligi | "
        "IYD 328 - Is Yeri Deneyimi | 2025-2026 Bahar Donemi"
    )


if __name__ == '__main__':
    main()
