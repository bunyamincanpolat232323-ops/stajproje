"""
Gorsellestirme Modulu
Proje: Duygu Analizi Gosterge Paneli
Gelistiren: Bunyamin Canpolat - 230206063
Ostim Teknik Universitesi - Bilgisayar Muhendisligi

Bu modul Plotly ile veri gorsellestirme fonksiyonlari saglar.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter


def plot_sentiment_distribution(df):
    """Duygu dagilimi pasta grafigi."""
    counts = df['sentiment'].value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Duygu Dagilimi",
        color=counts.index,
        color_discrete_map={'pozitif': '#2ecc71', 'negatif': '#e74c3c'}
    )
    fig.update_layout(template='plotly_white')
    return fig


def plot_sentiment_bar(df):
    """Duygu dagilimi cubuk grafigi."""
    counts = df['sentiment'].value_counts()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title="Duygu Dagilimi (Cubuk)",
        labels={'x': 'Duygu', 'y': 'Sayi'},
        color=counts.index,
        color_discrete_map={'pozitif': '#2ecc71', 'negatif': '#e74c3c'}
    )
    fig.update_layout(template='plotly_white', showlegend=False)
    return fig


def plot_top_words(df, sentiment=None, top_n=20):
    """En sik kullanilan kelimeleri gosterir."""
    if sentiment:
        subset = df[df['sentiment'] == sentiment]
        title = f"{sentiment.capitalize()} Yorumlarda En Sik {top_n} Kelime"
    else:
        subset = df
        title = f"En Sik Kullanilan {top_n} Kelime"

    if 'cleaned_text' not in subset.columns:
        return go.Figure()

    all_words = ' '.join(subset['cleaned_text'].dropna()).split()
    word_freq = Counter(all_words).most_common(top_n)

    if not word_freq:
        return go.Figure()

    words_df = pd.DataFrame(word_freq, columns=['Kelime', 'Frekans'])

    fig = px.bar(
        words_df,
        x='Frekans',
        y='Kelime',
        orientation='h',
        title=title,
        color='Frekans',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


def plot_model_comparison(results):
    """Model performans karsilastirma grafigi."""
    chart_data = []
    for name, metrics in results.items():
        for metric_name, metric_key in [
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision_macro'),
            ('Recall', 'recall_macro'),
            ('F1-Score', 'f1_macro')
        ]:
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
    return fig


def plot_confusion_matrix(conf_mat, labels, title="Confusion Matrix"):
    """Confusion matrix heatmap gorseli."""
    fig = go.Figure(data=go.Heatmap(
        z=conf_mat,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=conf_mat,
        texttemplate='%{text}',
        showscale=False
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Tahmin Edilen',
        yaxis_title='Gercek',
        template='plotly_white',
        height=350
    )
    return fig
