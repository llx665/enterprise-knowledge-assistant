# ============================================================
# Initialize sample documents
# ============================================================
import os

SAMPLE_FILES = {
    "python_basics.md": """# Python Basics

## Functions

Define functions using def keyword.

## List Comprehension

`python
squares = [x**2 for x in range(10)]
`

## Exception Handling

Use try-except to catch exceptions.
""",
    "company_rules.md": """# Company Rules

## Attendance

Working hours: Mon-Fri 9:00-18:00.
Late arrival over 30min counts as absence.

## Leave Policy

5 days annual leave after 1 year.
Sick leave requires medical certificate.

## Confidentiality

Employees shall not disclose trade secrets.
""",
    "python_advanced.md": """# Python Advanced

## Decorators

Decorators modify function behavior.

## Context Managers

Use with statement for resource management.

## Multi-threading

Python uses threading module.

## Package Management

Use pip install for packages.
""",
}


def create_sample_documents():
    output_dir = "./data/documents"
    os.makedirs(output_dir, exist_ok=True)
    for fname, content in SAMPLE_FILES.items():
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {fname}")
    print("Done! Sample documents are ready for upload.")


if __name__ == "__main__":
    create_sample_documents()
