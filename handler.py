import torch
from sentence_transformers import SentenceTransformer

class ModelHandler:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def load_model(self):
        self.model = SentenceTransformer("dragonkue/BGE-m3-ko")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
    def encode(self, texts, **kwargs):
        if not isinstance(texts, list):
            texts = [texts]
        
        embeddings = self.model.encode(texts, **kwargs)
        return embeddings.tolist()
    
    def __call__(self, job):
        job_input = job["input"]
        texts = job_input.get("texts", [])
        
        # 추가 파라미터 처리
        kwargs = {}
        if "batch_size" in job_input:
            kwargs["batch_size"] = job_input["batch_size"]
        if "normalize_embeddings" in job_input:
            kwargs["normalize_embeddings"] = job_input["normalize_embeddings"]
            
        try:
            embeddings = self.encode(texts, **kwargs)
            return {"embeddings": embeddings}
        except Exception as e:
            return {"error": str(e)}
