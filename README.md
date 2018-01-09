# Sentiment Analysis Experiments

This repository contains experiments in Vietnamese sentiment analysis problems. It is a part of [underthesea](https://github.com/magizbox/underthesea) project.

# Results

![](https://img.shields.io/badge/F1-0.43-red.svg)

| Model                                                          | F1 Score (%) |
|----------------------------------------------------------------|--------------|
| Logistic Regression (Tfidf_ngrams(1,2) + max_df=0.8+ min_df=8) | 53.6         |
| Logistic Regression (Count_ngrams(1,2) + max_df=0.8+ min_df=8) | 58.3         |
| SVC (Count_ngrams(1,2) + max_df=0.8 + min_df=0.005)            | 57.6         |
| SVC (Count_ngrams(1,2) + max_df=0.5 + min_df=8)                | 59.5         |
| LinearSVC (Tfidf_ngrams(1,2)                                   | 63.0         |
| LinearSVC (Tfidf_ngrams(1,2) + max_features=5000)              | 63.5         |

Guide

Tiếng Việt
### Huấn luyện mô hình:

Các bước thực hiện để đánh giá mô hình với dữ liệu đã có:

#### Bước 1: Tiền xử lí dữ liệu

Chạy preprocess.py tại thư mục data/fb_bank_sentiments_2200. Với dữ liệu đầu vào tại thư mục raw. Dữ liệu đầu ra lưu trong thư mục corpus gồm các tệp định dạng excel: data, train, test. Tiếp tục chạy eda.py phân tích dữ liệu thăm dò để lập bảng thống kê đơn giản. Hình ảnh phân tích lưu tại thư mục eda.

#### Bước 2: Huấn luyện mô hình
Chạy train.py tại thư mục experiments/[Các_thử_nghiệm_với_model_và_features_tương_ứng]. Với mỗi thử nghiệm gồm model và feature tương ứng mục đích để tìm ra model và features thích hợp với dữ liệu. Đầu vào là các dữ liệu train lấy từ tệp data/fb_bank_sentiments_2200/corpus/train.xlsx. Đầu ra là các tệp .bin lưu tai thư mục model.

#### Bước 3: Kiểm tra mô hình
Chạy test.py tại thư mục experiments/[Các_thử_nghiệm_với_model_và_features_tương_ứng]. Ở đây gồm các hàm kiểm tra đơn giản với đầu vào là các câu và đầu ra là kết quả tương ứng, mục tiêu là kiểm tra tính đúng đắn của model. Các test gọi tới hàm sentiment trong model/__init__.py.

#### Bước 4: Phân tích dữ liệu
Với các mô hình thu được sau khi huấn luyện dữ liệu, tiến hành kiểm tra với dữ liệu kiểm thử tại: data/fb_bank_sentiments_2200/corpus/test.xlsx. Đầu ra được thể hiện trên Board bao gồm: các kết quả (F1 Weighted, Accuracy...), mô tả dữ liệu và màn hình hiện thị các câu kết hợp các nhãn tương ứng.

English
### Training your customer model
Step to step:
#### Step 1: Preprocessing data
Run preprocess.py file in data/fb_bank_sentiments_2200 folder. Input data in raw folder. Output data in corpus folder with excel file format: data, train, test. Exploratory data analysis with eda.py file for plotting simple statistics. Result store in eda folder include image data analysis.

### Step 2: Train model
Run train.py file in experiments/[experiment_with_model_and_feature]. Experiment include model and feature for model valid with data. Input data is data/fb_bank_sentiments_2200/corpus/train.xlsx. Output are .bin file in model folder.

### Step 3: Test model
Run test.py file in experiments/[experiment_with_model_and_feature]. They are simple test with input text and output include result of text. The goal is correctness of the model. Tests call sentiment function in model/__init__.py.

### Step 4: Analyze data
From the models obtained after the data trainer, conduct the test with the test data at data/fb_bank_sentiments_2200/corpus/test.xlsx. The output is shown on the board include: results (F1 Weighted, Accuracy ...), data descriptive, and display of matching sentence combinations.

# Reproduce

Setup Environment

```
# clone project
$ git clone https://github.com/undertheseanlp/sentiment

# create environment
$ cd sentiment
$ conda create -n sentiment python=3.5
$ pip install -r requirements.txt
$ pip install git+https://github.com/undertheseanlp/languageflow
```

# Related Works

* Vietnamese Sentiment Analysis publications, [link](https://github.com/magizbox/underthesea/wiki/Vietnamese-NLP-Publications#sentiment-analysis)
