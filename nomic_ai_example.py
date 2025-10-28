from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from PIL import Image
import requests

# Use of nomic-ai embedding model for image RAG.

# Load models
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
text_model   = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
processor    = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
tokenizer    = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

# --- Embed an image ---
image = Image.open(requests.get("https://example.com/dog.jpg", stream=True).raw)
img_inputs = processor(image, return_tensors="pt")
img_emb = vision_model(**img_inputs).last_hidden_state[:, 0, :]
img_emb = F.normalize(img_emb, p=2, dim=1)

# --- Embed text query ---
text_inputs = tokenizer("a brown dog running on grass", return_tensors="pt")
text_emb = text_model(**text_inputs).last_hidden_state[:, 0, :]
text_emb = F.normalize(text_emb, p=2, dim=1)

# --- Compare similarity ---
similarity = (img_emb @ text_emb.T).item()
print("Similarity:", similarity)
