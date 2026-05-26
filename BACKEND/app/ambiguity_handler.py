AMBIGUOUS_TERMS = [
    "ai",
    "agent",
    "agents",
    "robotics",
    "technology",
    "machine learning"
]


def is_ambiguous(query):

    cleaned_query = query.strip().lower()

    if len(cleaned_query.split()) <= 1:
        return True

    if cleaned_query in AMBIGUOUS_TERMS:
        return True

    return False