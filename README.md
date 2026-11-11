# Mini-RAG Pipeline: Sherlock Holmes Story summaries

This project is an end-to-end Mini-RAG (Retrieval-Augmented Generation) pipeline built as part of my portfolio. It demonstrates how to take a collection of public-domain Sherlock Holmes story summaries, ingest them into a vector database, and provide a question-answering interface that uses those documents to generate accurate answers.

## Project Overview

The project mirrors the core retrieval loop of a larger RAG application. It is designed to be fully self-contained, capable of running either locally or with a remote Qdrant cloud instance.

- **Stack:** Python 3.x, OpenAI API (`gpt-4o-mini` + `text-embedding-3-small`), Qdrant Vector Database.
- **Key Concepts:** Basic RAG, document chunking, embeddings generation, vector similarity search, and answer synthesis using LLMs.
- **Corpus:** Summaries of 5 public-domain Sherlock Holmes stories (Arthur Conan Doyle, 1892).

## Getting Started

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

- Python 3.x installed on your machine.
- An OpenAI API Key.
- (Optional) A Qdrant Cloud URL and API Key if you prefer not to run it locally.

### 2. Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
    *(Remember to replace `your-username` and `your-repository-name` with your actual information.)*

2.  **Create and activate a virtual environment (optional but recommended):**
    ```powershell
    python -m venv .venv
    # PowerShell
    .\.venv\Scripts\Activate.ps1
    # Bash/Linux
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your environment variables:**
    - Copy the template file: `cp .env.example .env`
    - Open the new `.env` file and add your `OPENAI_API_KEY`.
    - If you are using Qdrant locally (default), you can leave `QDRANT_URL` and `QDRANT_API_KEY` blank.

### 3. Running the Pipeline

The project includes three main commands to interact with the pipeline:

- **Ingest Data:** This command reads the corpus, chunks the documents, generates embeddings, and stores them in Qdrant.
  ```powershell
  python .\mp2_rag.py ingest

  Run Validation: This runs a predefined set of questions (including golden questions and learner-defined questions) to test the pipeline's performance. The results are automatically saved to mp2_validation.txt.

PowerShell
python .\mp2_rag.py validate
Interactive Ask: This enters an interactive mode where you can ask your own questions of the Sherlock Holmes corpus.

PowerShell
python .\mp2_rag.py ask

Customization
You are free to experiment with the pipeline:

Change the Corpus: Add or remove .txt files in the corpus/ folder to make the RAG system answer questions about different documents.

Update the Evaluation Questions: Modify the .jsonl files in the data/ folder to add your own test questions and expected sources.

Adjust the Chunker: Experiment with the TARGET_CHUNK_SIZE and CHUNK_OVERLAP settings in mp2_rag.py to optimize retrieval performance.

