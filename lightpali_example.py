# LightPali that can embed images for RAG.
# The library can use ColQwen and other models.
# https://medium.com/@pankaj_pandey/litepali-lightweight-document-retrieval-with-vision-language-models-5af143b78eb7

from pdf2image import convert_from_path
from pathlib import Path

# Converting PDFs to Images
def pdf_to_images(pdf_path, out_dir, dpi=200):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(convert_from_path(pdf_path, dpi=dpi), start=1):
        img_path = out / f"{Path(pdf_path).stem}_p{i:03}.jpg"
        img.save(img_path, "JPEG", quality=90)
    return sorted(out.glob("*.jpg"))

from litepali import LitePali, ImageFile
# Step 1: Initialise
litepali = LitePali()
# Step 2: Add images
litepali.add(ImageFile(path="doc1_p1.jpg", document_id=1, page_id=1,
                       metadata={"title": "Introduction"}))
litepali.add(ImageFile(path="doc1_p2.png", document_id=1, page_id=2,
                       metadata={"title": "Results"}))
litepali.add(ImageFile(path="doc2_p1.jpg", document_id=2, page_id=1,
                       metadata={"title": "Abstract"}))
# Step 3: Process and build index
litepali.process()
# Step 4: Search
results = litepali.search("ROC curve for model B", k=5)
for r in results:
    print(f"Image: {r['image'].path}, Score: {r['score']:.3f}")
# Step 5: Save and reload index
litepali.save_index("index_dir")
new_engine = LitePali()
new_engine.load_index("index_dir")

#You can expose LitePali as a FastAPI service:
from fastapi import FastAPI, Query
from litepali import LitePali

app = FastAPI()
engine = LitePali()
engine.load_index("index_dir")
@app.get("/search")
def search(q: str = Query(...), k: int = 5):
    hits = engine.search(q, k=k)
    return [{"path": h["image"].path, "score": float(h["score"])} for h in hits]

# Models
#vidore/colpali-v1.3 → balanced (84.8 NDCG@5).
#vidore/colqwen2-v1.0 → higher accuracy (~89 NDCG@5) but heavier.
#vidore/colSmol-256M/500M → much smaller, Apache-2.0, good for edge/CPU tests.
   
