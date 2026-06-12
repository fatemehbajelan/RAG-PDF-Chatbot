# 📄 RAG PDF Chatbot

This project is a simple Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF files and ask questions based on their content.

---

## 🚀 Features

- Upload PDF documents
- Extract and process text from PDFs
- Split text into chunks for better search
- Semantic search using FAISS
- Text embeddings using SentenceTransformers
- Answer questions using an LLM (OpenRouter API)
- Simple web interface using Gradio

---

## 🧠 How it works

1. User uploads a PDF file  
2. Text is extracted and split into chunks  
3. Each chunk is converted into embeddings  
4. FAISS stores embeddings for fast similarity search  
5. When user asks a question:
   - The most relevant chunks are retrieved
   - They are sent to the LLM
   - The model generates an answer

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

This project uses OpenRouter API.

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

⚠️ Never upload your API key to GitHub.

---

## ▶️ Run the Project

```bash
python app.py
```

Then open the local link shown in terminal (Gradio interface).

---

## 📦 Requirements

- Python 3.9+
- gradio
- faiss-cpu
- sentence-transformers
- numpy
- requests
- python-dotenv
- pypdf

---

## 📁 Project Structure

```
rag-pdf-chatbot/
│
├── app.py
├── requirements.txt
├── .env (not uploaded)
├── README.md
└── data/
```

---

## ⚠️ Notes

- Make sure your API key is valid
- Large PDFs may take time to process
- For better results, use clean and structured PDFs

---

## 👩‍💻 Author

Fatemeh
```
