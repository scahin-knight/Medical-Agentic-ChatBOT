from app.knowledge import search_kb

queries = [
    "hi",
    "how to repair a car engine",
    "anemia symptoms",
    "what is appendicitis",
    "diabetes treatment",
]

for q in queries:
    print(f"\n{q}")
    for r in search_kb(q, top_k=3):
        print(f"  {r['score']:.3f}  p.{r['page']}  {r['text'][:80]!r}")