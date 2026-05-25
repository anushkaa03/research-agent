from bs4 import BeautifulSoup


def extract_clean_text(html_content: str):

    soup = BeautifulSoup(
        html_content,
        "lxml"
    )

    for script in soup(
        ["script", "style"]
    ):
        script.extract()

    text = soup.get_text(
        separator=" "
    )

    cleaned_text = " ".join(
        text.split()
    )

    return cleaned_text