# Example of using images(jpg, png...) as RAG with ColQwen2
# Medium article(sub needed to read) https://ai.gopubby.com/efficient-multimodal-document-retrieval-with-colqwen2-b8f5afa8f524
# https://huggingface.co/vidore/colqwen2-v1.0-hf
# https://arxiv.org/pdf/2409.12191

# ColQwen2 is mainly embedding model, not LLM(though you can query for search.)
# Combine ColQwen2 with other LLM model.

#If you have documents (e.g., PDFs, scanned reports, layouts, tables) 
# that you need to retrieve by query (e.g., find relevant policy documents,
# claim reports) then ColQwen2 could be a good fit, especially where 
# layout matters (charts, tables, visual cues).

# First, convert pdf into images.
from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder='output_images', dpi=200, image_format='jpeg'):
    """
    Converts a PDF file to a set of images (one per page).
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Directory to save the output images.
        dpi (int): Dots per inch (resolution) for image rendering.
        image_format (str): Image format (e.g., 'jpeg', 'png').
        
    Returns:
        List[str]: Paths to the generated image files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, img in enumerate(images):
        image_filename = os.path.join(output_folder, f'page_{i + 1}.{image_format}')
        img.save(image_filename, image_format.upper())
        image_paths.append(image_filename)
        print(f"Saved: {image_filename}")
    
    return image_paths


#Second, search images using LLM.
import torch
from PIL import Image

from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

# Load the model and the processor
model_name = "vidore/colqwen2-v1.0-hf"

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # "cpu", "cuda", or "mps" for Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
    )
processor = ColQwen2Processor.from_pretrained(model_name)


image_paths = pdf_to_images('/path/to/my-pdf-file.pdf')
images = [Image.open(img) for img in image_paths] # Read input files

# The queries you want to retrieve documents for
queries = ["Whats the definition of late interaction?"]

# Process the inputs
inputs_images = processor(images=images).to(model.device)
inputs_text = processor(text=queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

# Score the queries against the images (MaxSim)
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print(f"Retrieval scores (query x image): {scores}") # shape (n_queries, n_documents)
print(f"Highest scoring document: {image_paths[torch.argmax(scores)]}")

