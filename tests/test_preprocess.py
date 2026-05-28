import pytest
from utils.preprocess import clean_text


class TestCleanText:
    def test_lowercase_conversion(self):
        """Metni küçük harfe çevirdiğini test et."""
        result = clean_text("HELLO WORLD")
        assert result == "hello world"

    def test_url_removal(self):
        """URL'leri kaldırdığını test et."""
        text = "Check this http://example.com website"
        result = clean_text(text)
        assert "http" not in result
        assert "example.com" not in result

    def test_number_removal(self):
        """Sayıları kaldırdığını test et."""
        text = "2024 yılında 100 kişi"
        result = clean_text(text)
        assert "2024" not in result
        assert "100" not in result

    def test_punctuation_removal(self):
        """Noktalama işaretlerini kaldırdığını test et."""
        text = "Hello, world! How are you?"
        result = clean_text(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_whitespace_normalization(self):
        """Fazla boşlukları temizlediğini test et."""
        text = "Hello    world   test"
        result = clean_text(text)
        assert result == "hello world test"

    def test_empty_string(self):
        """Boş string işlediğini test et."""
        result = clean_text("")
        assert result == ""

    def test_non_string_input(self):
        """String olmayan input işlediğini test et."""
        result = clean_text(None)
        assert result == ""
        result = clean_text(123)
        assert result == ""

    def test_combined_cleaning(self):
        """Tüm temizleme işlemlerinin birlikte çalıştığını test et."""
        text = "FAKE NEWS 2024: Check http://fake.com NOW!!!!"
        result = clean_text(text)
        assert result == "fake news check now"

    def test_stripping_whitespace(self):
        """Başındaki ve sonundaki boşlukları kaldırdığını test et."""
        text = "   hello world   "
        result = clean_text(text)
        assert result == "hello world"

    def test_mixed_urls_and_text(self):
        """Farklı türdeki URL'leri kaldırdığını test et."""
        text = "Visit https://example.com or www.example.com today"
        result = clean_text(text)
        assert "https" not in result
        assert "www" not in result
        assert "example.com" not in result
