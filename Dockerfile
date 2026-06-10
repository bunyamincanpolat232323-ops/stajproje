# Duygu Analizi Gösterge Paneli - Dockerfile
# Bünyamin Canpolat - 230206063
# Ostim Teknik Üniversitesi - Bilgisayar Mühendisliği

FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK verilerini indir
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Proje dosyalarını kopyala
COPY . .

# Veri setini işle ve modelleri eğit
RUN python src/train_models.py

# Streamlit portu
EXPOSE 8501

# Sağlık kontrolü
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Dashboard'u başlat
ENTRYPOINT ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
