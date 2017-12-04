from unittest import TestCase
from model import sentiment


class TestSentiment(TestCase):
    def test_sentiment(self):
        sentence = "Thật tuyệt vời"
        tags = "POSITIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])

    def test_sentiment_1(self):
        sentence = "Nhân viên BIDV dễ thương nhiệt tình lắm ạ"
        tags = "POSITIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])

    def test_sentiment_2(self):
        sentence = "Không tin tưởng vào ngân hàng BIDV."
        tags = "NEGATIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])

    def test_sentiment_3(self):
        sentence = "Dịch vụ rắc rối."
        tags = "NEGATIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])

    def test_sentiment_4(self):
        sentence = "Thật sự mình rất hài lòng khi làm việc với VietcomBank"
        actual = sentiment(sentence)
        expected = "POSITIVE"
        self.assertEquals(expected, actual[0])

    def test_sentiment_5(self):
        sentence = "Vietcombank là lũ lừa đảo"
        tags = "NEGATIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])

    def test_sentiment_6(self):
        sentence = "Đkmm"
        tags = "NEGATIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])

    def test_sentiment_7(self):
        sentence = "éo tin"
        tags = "NEGATIVE"
        actual = sentiment(sentence)
        self.assertEquals(tags, actual[0])
