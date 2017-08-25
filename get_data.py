import requests
import json

from underthesea.util.file_io import write

LABELS = ["BRAND", "INSTALLATION_SERVICE", "COMPETITOR_COMPARTION", "SPEED_BANDWIDTH", "PRICE", "LICENSE", "HARDWARE",
          "CUSTOMER_SERVICE"]


def parse_post(post):
    data = dict()
    data["text"] = post["content"]
    labels = [label["name"] for label in post["labels"]]
    labels = [label for label in labels if label in LABELS]
    data["labels"] = labels
    return data


if __name__ == '__main__':
    labels_params = "&".join(["labels_name=" + label for label in LABELS])
    url = "http://192.168.0.242:6619/itrack-api-v2/api/events/?campaigns=1limit=10&embed=true&" + labels_params
    data = {}
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InN1cGVyYWRtaW4iLCJ1c2VyX2lkIjoyLCJlbWFpbCI6InN1cGVyYWRtaW5AZ21haWwuY29tIiwiZXhwIjoxNTAwNjk2MDg1fQ.9MqxpozPep0lmQVRQVBW2qbRQCndzYuLJrI1WEryoa4'
    }
    r = requests.get(url, data=json.dumps(data), headers=headers)
    posts = r.json()
    posts = [parse_post(p) for p in posts]
    write("data.json", json.dumps(posts, ensure_ascii=False))
    print(0)
