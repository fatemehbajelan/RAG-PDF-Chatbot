import fitz
import faiss
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------
# OpenRouter
# -----------------------

client = OpenAI(
    api_key="YOUR_API",
    base_url="https://openrouter.ai/api/v1"
)

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

chunk_size = 15

chunks = [
    "\n".join(lines[i:i + chunk_size])
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

    try:

        
        query_embedding = model.encode([question])

        
        D, I = index.search(
            np.array(query_embedding),
            k=7
        )

        retrieved_chunks = [
            chunks[i]
            for i in I[0]
        ]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a PDF Question Answering Assistant.

Rules:
1. Answer ONLY from the provided context.
2. If the answer is not found in the context, say:
   "The answer was not found in the PDF."
3. Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You answer only from the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=2000
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error: {str(e)}"

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

Ask Questions About Your PDF
"""
    )

    question = gr.Textbox(
        label="Question",
        lines=4,
        placeholder="Ask a question..."
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