import pytest
from unittest.mock import patch, MagicMock
from app import app, load_artifacts


@pytest.fixture
def client():
    """Flask test client oluştur."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestIndex:
    def test_index_loads_without_error(self, client):
        """Ana sayfa yüklenmesini test et."""
        with patch('app.model', None), patch('app.vectorizer', None):
            response = client.get('/')
            assert response.status_code == 200

    def test_index_shows_model_not_loaded(self, client):
        """Model yüklenmediğinde uyarı gösterildiğini test et."""
        with patch('app.model', None), patch('app.vectorizer', None):
            response = client.get('/')
            assert b'model_loaded' in response.data or response.status_code == 200


class TestPredict:
    def test_predict_without_text(self, client):
        """Boş metin gönderilmesini test et."""
        with patch('app.model', MagicMock()), patch('app.vectorizer', MagicMock()):
            response = client.post('/predict', data={'news_text': ''})
            assert response.status_code == 200
            assert b'Lutfen haber metni giriniz' in response.data

    def test_predict_with_short_text(self, client):
        """Kısa metin gönderilmesini test et."""
        with patch('app.model', MagicMock()), patch('app.vectorizer', MagicMock()):
            response = client.post('/predict', data={'news_text': 'short'})
            assert response.status_code == 200
            assert b'Metin cok kisa' in response.data or b'En az 20 karakter' in response.data

    def test_predict_without_model(self, client):
        """Model yüklenmediğinde hata gösterildiğini test et."""
        with patch('app.model', None), patch('app.vectorizer', None):
            response = client.post('/predict', data={'news_text': 'This is a longer test text'})
            assert response.status_code == 200
            assert b'Model yuklenemedi' in response.data

    def test_predict_with_valid_text(self, client):
        """Geçerli metin ile tahmin testini test et."""
        mock_model = MagicMock()
        mock_model.predict.return_value = ['FAKE']
        mock_model.predict_proba.return_value = [[0.8, 0.2]]

        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()

        with patch('app.model', mock_model), \
             patch('app.vectorizer', mock_vectorizer), \
             patch('app.model_accuracy', 0.95):
            response = client.post('/predict', data={'news_text': 'This is a valid test text for prediction'})
            assert response.status_code == 200

    def test_predict_get_method_not_allowed(self, client):
        """GET ile /predict'e erişilmesini test et."""
        response = client.get('/predict')
        assert response.status_code == 405  # Method Not Allowed

    def test_predict_handles_invalid_form_data(self, client):
        """Form verisi olmadığında hatayı test et."""
        with patch('app.model', MagicMock()), patch('app.vectorizer', MagicMock()):
            response = client.post('/predict', data={})
            assert response.status_code == 200
            assert b'Lutfen haber metni giriniz' in response.data

    def test_predict_confidence_calculation(self, client):
        """Confidence skorunun hesaplandığını test et."""
        mock_model = MagicMock()
        mock_model.predict.return_value = ['REAL']
        mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% confidence

        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()

        with patch('app.model', mock_model), \
             patch('app.vectorizer', mock_vectorizer), \
             patch('app.model_accuracy', 0.90):
            response = client.post('/predict', data={'news_text': 'Valid test news article here'})
            assert response.status_code == 200
