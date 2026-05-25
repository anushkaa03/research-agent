import requests


def fetch_url_content(url: str):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        return response.text

    return None