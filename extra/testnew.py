import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/hello/"

# Send a GET request to the FastAPI endpoint
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Print the JSON response
    print("Response from FastAPI:", response.json())
else:
    print(f"Failed to get response. Status code: {response.status_code}")
