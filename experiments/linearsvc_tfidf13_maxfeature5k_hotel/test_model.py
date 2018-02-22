from unittest import TestCase
from model import sentiment


class TestSentiment(TestCase):
    def test_sentiment_1(self):
        text = "Phòng sạch sẽ, thoáng, nhân viên nhiệt tình, giá cả phải chăng, thủ tục nhanh gọn. Nói chung là oke :) "
        actual = sentiment(text)
        expected = ('HOTEL#PRICES#POSITIVE', 'ROOMS#CLEANLINESS#POSITIVE', 'SERVICE#GENERAL#POSITIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_2(self):
        text = "nhận phòng không tốt giống như hình khi book rất là giận về vấn đề này. "
        actual = sentiment(text)
        expected = ('HOTEL#COMFORT#NEGATIVE', 'ROOMS#GENERAL#NEGATIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_3(self):
        text = "Bữa sáng hơi đăt Thức ăn không ngon, không hợp khẩu vị cho các vùng miền khác qua nghỉ dưỡng "
        actual = sentiment(text)
        expected = ('FOOD&DRINKS#QUALITY#NEGATIVE', 'FOOD&DRINKS#QUALITY#POSITIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_4(self):
        text = "Sạch sẽ, thân thiện, giá hợp lý, nhân viên dể thương(candy, happy, funny...) "
        actual = sentiment(text)
        expected = ('HOTEL#CLEANLINESS#POSITIVE', 'HOTEL#PRICES#POSITIVE', 'SERVICE#GENERAL#POSITIVE')
        self.assertEquals(actual, expected)
