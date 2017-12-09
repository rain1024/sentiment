from bank_sentiment_svm.model import sentiment
from unittest import TestCase


class TestSentiment(TestCase):
    def test_sentiment(self):
        text = "Gọi mấy lần mà lúc nào cũng là các chuyên viên đang bận hết ạ "
        actual = sentiment(text)
        expected = "CUSTOMER SUPPORT#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_1(self):
        text = "Không tin tưởng vào ngân hàng BIDV "
        actual = sentiment(text)
        expected = "TRADEMARK#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_2(self):
        text = "Bạn ra ngân hàng hỏi luôn cho nhanh giải đáp qua đây ko ăn thua lắm."
        actual = sentiment(text)
        expected = "CUSTOMER SUPPORT#NEGATIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_3(self):
        text = "Vậy tốt quá, giờ sài thẻ an toàn lại tiện lợi nữa."
        actual = sentiment(text)
        expected = "CARD#POSITIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_4(self):
        text = "VCB Mobile B@bking. Chuyển rất nhanh. Giao diện đẹp, thân thiện. Dễ thao tác"
        actual = sentiment(text)
        expected = "INTERNET BANKING#POSITIVE"
        self.assertEquals(actual[0], expected)

    def test_sentiment_5(self):
        text = "Em rất thích ứng dụng BIDV smart banking.Thật tiện lợi.Từ 1 tr là gửi được rùi."
        actual = sentiment(text)
        expected = "INTERNET BANKING#POSITIVE"
        self.assertEquals(actual[0], expected)
