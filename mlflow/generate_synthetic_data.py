"""
Generate Large Synthetic Dataset

Generates a large synthetic dataset for MLflow experiments.
Target: 2,000 queries and 20,000 chunks.
Uses templates and random combinations to create diverse content.
"""

import json
import random
import uuid

# -----------------------------
# Configuration
# -----------------------------
NUM_CHUNKS = 20000
NUM_QUERIES = 2000

DOMAINS = [
    "Physics", "Computer Science", "Biology",
    "Mathematics", "Economics", "History",
    "Psychology", "Chemistry"
]

TOPICS = {
    "Physics": ["motion", "energy", "waves", "relativity", "quantum effects"],
    "Computer Science": ["algorithms", "data structures", "machine learning", "networks", "databases"],
    "Biology": ["cells", "genetics", "evolution", "metabolism", "ecology"],
    "Mathematics": ["calculus", "linear algebra", "probability", "number theory"],
    "Economics": ["markets", "inflation", "game theory", "growth", "trade"],
    "History": ["revolutions", "wars", "civilizations", "colonialism"],
    "Psychology": ["cognition", "behavior", "learning", "emotion"],
    "Chemistry": ["reactions", "bonding", "thermodynamics", "kinetics"]
}

QUERY_TEMPLATES = [
    "Can you explain the main idea behind this?",
    "Why is this concept important?",
    "What problem does this address?",
    "How does this influence its field?",
    "What is the practical significance of this?",
    "What should a student understand about this?",
    "What are the key principles involved here?"
]

HARD_QUERY_TEMPLATES = [
    "How does this compare to similar ideas?",
    "Why was this approach developed?",
    "What limitations does this concept have?",
    "In what situations does this fail?",
    "How is this applied in real scenarios?"
]

# -----------------------------
# Helper functions
# -----------------------------
def generate_chunk(domain, topic):
    return (
        f"This section discusses fundamental principles related to {topic}. "
        f"It focuses on theoretical foundations, practical implications, "
        f"and how these ideas are applied within the broader context of {domain}. "
        f"Examples and interpretations are provided to clarify the concept."
    )

def paraphrase_query(base):
    replacements = {
        "explain": ["describe", "clarify", "outline"],
        "important": ["significant", "essential", "valuable"],
        "concept": ["idea", "notion", "principle"]
    }
    for k, v in replacements.items():
        if k in base.lower() and random.random() < 0.4:
            base = base.replace(k, random.choice(v))
    return base

# -----------------------------
# Generate chunks (ground truth)
# -----------------------------
chunks = []
for _ in range(NUM_CHUNKS):
    domain = random.choice(DOMAINS)
    topic = random.choice(TOPICS[domain])
    chunk_id = str(uuid.uuid4())

    chunks.append({
        "chunk_id": chunk_id,
        "text": generate_chunk(domain, topic),
        "domain": domain,
        "topic": topic,
        "source": f"{domain.lower()}_book.pdf",
        "page": random.randint(1, 500)
    })

chunk_index = {c["chunk_id"]: c for c in chunks}

# -----------------------------
# Generate evaluation queries
# -----------------------------
queries = []

for _ in range(NUM_QUERIES):
    difficulty = random.random()

    # Pick a seed chunk
    seed_chunk = random.choice(chunks)
    domain = seed_chunk["domain"]
    topic = seed_chunk["topic"]

    # Choose template
    if difficulty < 0.7:
        template = random.choice(QUERY_TEMPLATES)
    else:
        template = random.choice(HARD_QUERY_TEMPLATES)

    query_text = template

    # Sometimes add implicit context
    if random.random() < 0.5:
        query_text += " in an academic context"

    query_text = paraphrase_query(query_text)

    # Relevant chunks: same domain, same topic preferred
    same_topic = [
        c for c in chunks
        if c["domain"] == domain and c["topic"] == topic
    ]

    same_domain = [
        c for c in chunks
        if c["domain"] == domain
    ]

    if len(same_topic) >= 3:
        relevant_chunks = random.sample(same_topic, random.randint(2, 3))
    else:
        relevant_chunks = random.sample(same_domain, random.randint(2, 3))

    queries.append({
        "query": query_text,
        "relevant_chunks": [c["chunk_id"] for c in relevant_chunks],
        "difficulty": "hard" if difficulty > 0.7 else "normal",
        "domain": domain
    })

# -----------------------------
# Save files
# -----------------------------
with open("ground_truth.json", "w", encoding="utf-8") as f:
    json.dump({c["chunk_id"]: c for c in chunks}, f, indent=2)

with open("eval_queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2)

print("✅ Dataset generated successfully")
print(f"Chunks: {NUM_CHUNKS}")
print(f"Queries: {NUM_QUERIES}")
