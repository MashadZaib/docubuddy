# 📄 Local PDF Chatbot with LangChain & Streamlit

A lightweight, local Retrieval-Augmented Generation (RAG) chatbot built using [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Ollama](https://ollama.com/) — allowing you to upload PDFs and chat with them using an open-source model like `mistral`.

---

## 🚀 Features

- 📁 Upload **any PDF**
- 🤖 Ask **natural language questions** about its content
- 🧠 Powered by **LangChain ConversationalRetrievalChain**
- 🔍 Answers with **context + source documents**
- 💬 Keeps track of conversation **within session**
- 💻 **Runs fully locally** — no OpenAI key needed
- 🧩 Uses **Ollama** + `mistral` model or your own

---

## 🛠 Requirements

- Python 3.9 or above
- [Ollama](https://ollama.com/) installed locally (must have the `mistral` model pulled)
- Recommended: Chrome or modern browser

---

## 📦 Installation

### 1. Clone the repo or copy files
```bash
git clone https://github.com/yourname/local-pdf-chatbot
cd rag-pdf-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull mistral
streamlit run app.py
.
├── app.py               # Main Streamlit app
├── requirements.txt     # All required dependencies
└── README.md            # You're reading it!
git commit -m "all commit"
