# Fake News Detector (Flask + Scikit-learn)

Bu projeyi, haber metinlerini **FAKE** veya **REAL** olarak siniflandirmak icin gelistirdim.  
Model tarafinda **TF-IDF + Logistic Regression**, web arayuzunde ise **Flask** kullandim.

## Features

- Haber metni girip tahmin alma (FAKE / REAL)
- Tahmin guven yuzdesi (confidence)
- Model accuracy gosterimi
- Confusion Matrix gorseli uretimi
- Model ve vectorizer dosyalarini pickle ile kaydetme/yukleme
- Basit input validation ve hata yonetimi
- Modern dark UI

## Technologies

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- HTML / CSS

## Project Structure

```bash
fake-news-detector/
|
|-- app.py
|-- train_model.py
|-- prepare_data.py
|-- model.pkl
|-- vectorizer.pkl
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- dataset/
|   |-- sample_news.csv
|   |-- news.csv
|-- static/
|   |-- style.css
|   |-- confusion_matrix.png
|-- templates/
|   |-- index.html
|   |-- result.html
|-- utils/
|   |-- preprocess.py
```

## Dataset

### Demo (hizli baslangic)

Repo icinde `dataset/sample_news.csv` bulunur:
- 1000 satir (500 FAKE + 500 REAL)
- GitHub'a sigar, clone sonrasi hemen calisir

```bash
python train_model.py --sample
```

`news.csv` yoksa script otomatik olarak demo verisini kullanir.

### Tam veri (Kaggle)

Kaggle **Fake and Real News Dataset** kullanilir.

1. Kaggle'dan zip indir ve icinden `Fake.csv` ile `True.csv` dosyalarini cikar.
2. `Fake.csv` dosyasini `dataset/news.csv` olarak kaydet.
3. `True.csv` dosyasini `dataset/True.csv` olarak kaydet.
4. Veriyi birlestir:

```bash
python prepare_data.py
```

`prepare_data.py` su islemleri yapar:
- `title` + `text` alanlarini birlestirir
- Sahte haberlere `FAKE`, gercek haberlere `REAL` etiketi ekler
- Sonucu `dataset/news.csv` dosyasina yazar

Hazir `news.csv` icinde su sutunlar olmali:

- `text`: haber metni
- `label`: `FAKE` veya `REAL`

> **Not:** `news.csv` dosyasi GitHub'a yuklenmez (dosya boyutu cok buyuk).
> Veriyi Kaggle'dan indirip `prepare_data.py` ile yerelde olusturman gerekir.

## Installation

1. Projeyi klonlayin:

```bash
git clone https://github.com/kullanici-adi/fake-news-detector.git
cd fake-news-detector
```

2. Virtual environment olusturun:

```bash
python -m venv venv
```

3. Ortami aktif edin (Windows):

```bash
venv\Scripts\activate
```

4. Kutuphaneleri yukleyin:

```bash
pip install -r requirements.txt
```

## Usage

### 0) Veriyi Hazirla (ilk kurulumda)

Kaggle'dan indirdiysen:

```bash
python prepare_data.py
```

`dataset/news.csv` zaten hazirsa bu adimi atlayabilirsin.

### 1) Modeli Egit

Demo veri ile:

```bash
python train_model.py --sample
```

Tam Kaggle verisi ile (`news.csv` hazirsa):

```bash
python train_model.py
```

Bu islem sonunda su dosyalar olusur:

- `model.pkl`
- `vectorizer.pkl`
- `static/confusion_matrix.png`

### 2) Flask Uygulamasini Calistir

```bash
python app.py
```

Tarayicida ac:
`http://127.0.0.1:5000`

### 3) Tahmin Al

- Haber metnini gir
- **Tahmin Et** butonuna bas
- Sonuc, confidence ve accuracy degerini gor

## Screenshots

- Ana sayfa (metin giris ekrani)
- Sonuc sayfasi (FAKE/REAL + confidence)
- Confusion Matrix gorseli

Ornek dosya yolu:

- `screenshots/home.png`
- `screenshots/result.png`
- `screenshots/confusion_matrix.png`

## Future Improvements

- Gelismis preprocessing (stopwords, lemmatization)
- Farkli model karsilastirmasi (SVM, Naive Bayes, BERT)
- REST API endpoint ekleme
- Docker ile containerization
- Unit test ve CI/CD pipeline

## License

Bu proje egitim ve portfoy amaclidir.
