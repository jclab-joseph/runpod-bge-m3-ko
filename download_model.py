from sentence_transformers import SentenceTransformer

print("Downloading BGE-m3-ko model...")
model = SentenceTransformer("dragonkue/BGE-m3-ko")
print("Model downloaded and saved to the cache directory.")
