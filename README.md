# 📚 RefQuest — Your Research Assistant

RefQuest is a powerful research assistant that allows users to upload **PDF files** or input **URLs**, processes them into vector embeddings using **HuggingFace**, stores them in a **FAISS vector database**, and answers user questions using **semantic search** and **Google Gemini 1.5 Flash LLM**.

---

## 🚀 Features

* 🔎 Input support for **PDFs** and **web articles (URLs)**
* 📄 Smart document splitting using recursive character splitting
* 🧠 Embedding generation with `sentence-transformers/all-mpnet-base-v2`
* 📦 Efficient document retrieval using **FAISS vector database**
* 🤖 Answer generation via **Gemini 1.5 Flash** LLM
* 🧠 Context-aware semantic search and response generation
* 🌐 Web app built using **Streamlit**

---

## 🏗️ How It Works

1. **Choose a Data Source**: Upload one or more PDF files or provide article URLs.
2. **Chunking and Embedding**:

   * Text is split into manageable chunks.
   * Each chunk is converted into vector embeddings using HuggingFace.
3. **Storage**: Chunks are saved in a **FAISS vector database** (`vectorstore.pkl`).
4. **Ask Questions**:

   * Your query is enhanced using retrieved document context.
   * RefQuest performs semantic similarity search in the vector database.
   * The **Gemini LLM** uses relevant chunks to generate accurate, comprehensive answers.
5. **Sources & Transparency**: RefQuest shows you the exact source content and documents used to generate each answer.

---

## 🧪 Tech Stack

| Component        | Technology                          |
| ---------------- | ----------------------------------- |
| Frontend         | Streamlit                           |
| Embeddings       | HuggingFace (`all-mpnet-base-v2`)   |
| Vector DB        | FAISS                               |
| LLM              | Google Gemini 1.5 Flash             |
| PDF Parsing      | LangChain's `PyPDFLoader`           |
| URL Parsing      | LangChain's `UnstructuredURLLoader` |
| Environment Mgmt | Python `.env` file with API key     |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/refquest.git
cd refquest
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

> You can get a Gemini API key from [Google AI Studio](https://ai.google.dev).

---

## ▶️ Run the App

```bash
streamlit run main.py
```

Then open the app in your browser at: `http://localhost:8501`

---


