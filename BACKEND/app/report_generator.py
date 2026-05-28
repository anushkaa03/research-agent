def generate_report(state):

    report = f"""
# Research Report

## Research Query
{state["query"]}

---

## Research Plan
{state["plan"]}

---

## Analysis and Findings
{state["final_answer"]}

---
"""

    return report