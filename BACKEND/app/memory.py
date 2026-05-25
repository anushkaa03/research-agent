class WorkingMemory:

    def __init__(self):
        self.facts = []

    def store_fact(self, fact: str):
        self.facts.append(fact)

    def get_facts(self):
        return self.facts


memory = WorkingMemory()