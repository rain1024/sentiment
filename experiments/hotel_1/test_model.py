from unittest import TestCase

from exported.svc import sentiment


class TestSentiment(TestCase):
    def test_sentiment_1(self):
        text = "Chưa có thang máy. Chưa chấp nhận thanh toán bằng thẻ. Địa điểm dễ tìm, bày trí bằng tre nứa rất mát mẻ, bạn lễ tân nhiệt tình, niềm nở, thân thiện, tốt bụng cực kỳ. Tôi đặt phòng 1 giường đôi nhưng biết tôi đi cùng 2 con nhỏ nên ks đã chủ động chuẩn bị thêm 1 chiếc giường tầng cho 2 bé. Phòng rộng rãi, bày trí đẹp, giường êm, nhà vệ sinh cực sạch sẽ. Vị trí thuận lợi, dễ di chuyển. Bạn lễ tân đã giới thiệu tôi các điểm ăn ngon trên đường Lê Thị Riêng, Nguyễn Trãi gần đó. Tóm lại, Nguyễn Shack xứng đáng để tôi giới thiệu bạn bè."
        actual = sentiment(text)
        expected = [('LOCATION#GENERAL#POSITIVE', 'SERVICE#GENERAL#POSITIVE')]
        self.assertEquals(actual, expected)

    def test_sentiment_2(self):
        text = "Chúng tôi không cảm thấy thoải mái vì ở chỉ 1 ngày mà cúp điện 3,4 lần . Thang máy thì quá nóng và quá chậm chạp. Bữa sáng tuyệt. Phòng ngủ tiện nghi đẹp. Nhân viên phục phụ rất tốt. "
        actual = sentiment(text)
        expected = [('ROOMS#COMFORT#POSITIVE',)]
        self.assertEquals(actual, expected)

    def test_sentiment_3(self):
        text = "Sạch sẽ, vị trí rất thuận tiện, nhân viên thân thiện, giuờng gra rất sạch và thoải mái. "
        actual = sentiment(text)
        expected = [('HOTEL#CLEANLINESS#POSITIVE', 'HOTEL#COMFORT#POSITIVE', 'LOCATION#GENERAL#POSITIVE',
                     'ROOMS#CLEANLINESS#POSITIVE', 'SERVICE#GENERAL#POSITIVE')]
        self.assertEquals(actual, expected)

    def test_sentiment_4(self):
        text = "Không có gì để không thích. Phòng vệ sinh sạch, nhân viên thân thiện và dễ thương. Giường ngủ riêng tư hơn homestay khác."
        actual = sentiment(text)
        expected = [('SERVICE#GENERAL#POSITIVE',)]
        self.assertEquals(actual, expected)
