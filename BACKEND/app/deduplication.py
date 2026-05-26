def deduplicate_facts(research_data):

    unique_data = []

    seen = set()

    for item in research_data:

        normalized = (
            item["summary"]
            .strip()
            .lower()
        )

        if normalized not in seen:

            seen.add(normalized)

            unique_data.append(item)

    return unique_data