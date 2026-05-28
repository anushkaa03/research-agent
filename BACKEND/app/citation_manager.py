def generate_citations(search_results):

    citations = []

    for index, item in enumerate(search_results, start=1):

        citation = {
            "id": f"[Source {index}]",
            "url": item["url"]
        }

        citations.append(citation)

    return citations