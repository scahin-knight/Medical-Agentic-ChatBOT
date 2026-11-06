import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="What are the symptoms of diabetes?"
)

embedding = response.embeddings[0]

print("Embedding successful!")
print("Dimensions:", len(embedding))
print("First 5 values:", embedding[:5])