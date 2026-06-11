import fitz
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# -----------------------
# Load PDF
# -----------------------

pdf_path = "Python and Machine Learning.pdf"

doc = fitz.open(pdf_path)

lines = []

for page in doc:
    text = page.get_text("text")

    for line in text.split("\n"):
        if line.strip():
            lines.append(line.strip())

# -----------------------
# Chunking
# -----------------------

chunk_size = 5

chunks = [
    "\n".join(lines[i:i+chunk_size])
    for i in range(0, len(lines), chunk_size)
]

print("Chunks:", len(chunks))

# -----------------------
# Embedding Model
# -----------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

print("Embedding Shape:", embeddings.shape)

# -----------------------
# FAISS Index
# -----------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings)
)

print("Vectors stored:", index.ntotal)

# -----------------------
# Search Function
# -----------------------

def search_pdf(question):

    query_embedding = model.encode([question])

    D, I = index.search(
        np.array(query_embedding),
        k=3
    )

    results = [
        chunks[i]
        for i in I[0]
    ]

    return "\n\n".join(results)

# -----------------------
# UI Design
# -----------------------

custom_css = """
body {
    background:#2f3136;
}

textarea {
    font-size:18px !important;
}

button {
    background:#2563eb !important;
    color:white !important;
    border-radius:12px !important;
}

footer {
    visibility:hidden;
}
"""

with gr.Blocks(
    title="RAG PDF Chatbot",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.Markdown(
        """
# 📚 RAG PDF Chatbot

Ask questions about your PDF
"""
    )

    question = gr.Textbox(
        label="Question",
        lines=4
    )

    answer = gr.Textbox(
        label="Answer",
        lines=12
    )

    ask_btn = gr.Button(
        "Search"
    )

    ask_btn.click(
        fn=search_pdf,
        inputs=question,
        outputs=answer
    )

demo.launch()