from unittest import TestCase
from model import sentiment


class TestSentiment(TestCase):
    def test_sentiment_1(self):
        text = "Hồ bơi đẹp , yên tĩnh , phục vụ nhiệt tình và dễ thương Bảo vệ chua chuyen nghiệp"
        actual = sentiment(text)
        expected = ('FACILITIES#DESIGN&FEATURES#POSITIVE', 'HOTEL#COMFORT#POSITIVE', 'SERVICE#GENERAL#POSITIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_2(self):
        text = "nhận phòng không tốt giống như hình khi book rất là giận về vấn đề này. "
        actual = sentiment(text)
        expected = ('HOTEL#COMFORT#NEGATIVE', 'ROOMS#GENERAL#NEGATIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_3(self):
        text = "Bữa sáng đa dạng, nhân viên nhiệt tình và thân thiện "
        actual = sentiment(text)
        expected = ('FOOD&DRINKS#STYLE&OPTIONS#POSITIVE', 'SERVICE#GENERAL#POSITIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_4(self):
        text = "Bữa sáng đơn giản nhưng ngon. Phòng sinh hoạt chung có piano điện và PS4. Căn hộ rộng rãi, có đủ bếp, " \
               "lò vi sóng, nướng... Bụi và ồn do bên cạnh đang xây dựng. Ăn sáng trên rooftop nên hơi nóng. "
        actual = sentiment(text)
        expected = ('FOOD&DRINKS#QUALITY#POSITIVE', 'HOTEL#COMFORT#NEGATIVE', 'HOTEL#MISCELLANEOUS#NEGATIVE', 'ROOMS#DESIGN&FEATURES#POSITIVE')
        self.assertEquals(actual, expected)
