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
        text = 'Bữa sáng ngon, đa dạng, đổi món hàng ngày. Phòng sạch sẽ, tôi ở phòng giá tiền thấp nhất nhưng view ' \
               'rất đẹp. 1 ngày họ dọn phòng 3 lần nên lúc nào phòng cũng rất sạch và gọn gàng. Nhân viên của resort ' \
               'rất nhiệt tình. Bãi biển riêng rất đẹp. 2 hồ bơi sạch và view ra biển đẹp. Resort ở xa trung tâm,' \
               'xung quanh hoang vu, muốn đổi món phải đi xa vào trung tâm mới có quán ăn. '
        actual = sentiment(text)
        expected = ('FACILITIES#CLEANLINESS#POSITIVE', 'FACILITIES#DESIGN&FEATURES#POSITIVE', 'FOOD&DRINKS#QUALITY#POSITIVE', 'ROOMS#CLEANLINESS#POSITIVE', 'ROOMS#DESIGN&FEATURES#POSITIVE', 'SERVICE#GENERAL#POSITIVE')
        self.assertEquals(actual, expected)

    def test_sentiment_4(self):
        text = "Sạch sẽ, thân thiện, giá hợp lý, nhân viên dể thương(candy, happy, funny...) "
        actual = sentiment(text)
        expected = ('HOTEL#CLEANLINESS#POSITIVE', 'HOTEL#PRICES#POSITIVE', 'SERVICE#GENERAL#POSITIVE')
        self.assertEquals(actual, expected)
