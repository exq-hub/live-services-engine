import requests

url = "http://127.0.0.1:8000/exq/search/text"
payload = {
    "text": "a man sitting on a table that is blue",
    "n": 5,
    "session_info": {"session": "test", "modelId": "2", "collection": "V3C"},
}
headers = {"Connection": "close"}

try:
    resp = requests.post(url=url, json=payload, headers=headers)
    print(resp)
except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
