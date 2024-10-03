import base64
import json
import ssl
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ssl._create_default_https_context = ssl._create_unverified_context

API_KEY = 'dYEh1aWXnVwxUVKYzG4qmT2r'
SECRET_KEY = 'YAwaBU2lxT11muRU6nbOiqSWduhHlXL6'
OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'


def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params).encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    result_str = urlopen(req, timeout=5).read().decode()
    result = json.loads(result_str)
    return result.get('access_token')


def read_file(image_path):
    with open(image_path, 'rb') as f:
        return f.read()


def request_ocr(url, data):
    req = Request(url, data.encode('utf-8'))
    result_str = urlopen(req).read().decode()
    return result_str


def ocr_process(image_path):
    token = fetch_token()
    image_url = OCR_URL + "?access_token=" + token
    file_content = read_file(image_path)
    result = request_ocr(image_url, urlencode({'image': base64.b64encode(file_content)}))
    result_json = json.loads(result)
    text = "".join([words_result["words"] for words_result in result_json["words_result"]])
    return text
