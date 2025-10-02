import requests

url = "http://127.0.0.1:8000/exq/search/aggregate"
payload = {
    "texts": ["a man sitting on a bench", "it is day"],
    "n": 5,
    "session_info": {"session": "test", "modelId": "2", "collection": "V3C"},
}
headers = {"Connection": "close"}

try:
    resp = requests.post(url=url, json=payload, headers=headers)
    print(resp)
except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
