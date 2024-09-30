# Document QA Bot
Document QA Bot is an interactive tool that allows users to interact with documents mainly PDFs. It allows the users to ask questions from the documents. It leverages the power of GEMMA, GROQ, Sentence Transformers, CTransformers, Langchain, and Streamlit to provide a seamless and intuitive user experience.
# Features
* Natural Language Interaction: Use natural language queries to extract information from CSV files.
* NLP Models: Leverage Llama 2 for language understanding and Sentence Transformers for sentence embedding.
* User-Friendly Interface: A Streamlit-based UI for easy interaction
* Faster load time: It uses GROQ API it is able to utilize LPU(Language Processing Unit) that is able to handle computationally intensive applications with a sequential component to them such as LLMs.
#Installation
Follow these steps to set up the Document-QA-Bot on your local machine:
1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/Document-QA-Bot.git
    cd Document-QA
    ```
2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
## Usage

1. **Prepare your Documents:**
   Ensure that you store the documents are stored in a proper folder.

2. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

## Project Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Technologies Used

- **Llama 2:** Advanced language model for natural language understanding.
- **Sentence Transformers:** Efficient sentence embedding for semantic similarity.
- **CTransformers:** Optimized transformer inference.
- **Langchain:** Chain of language models for complex NLP tasks.
- **Streamlit:** Fast way to build and share data apps.
  
