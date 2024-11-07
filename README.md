# Conversational RAG Chatbot with LangChain and Hugging Face Models

This project demonstrates a Retrieval-Augmented Generation (RAG) approach using LangChain to build a conversational chatbot capable of answering specific questions about complex documents. It uses PDF files, vector databases for context retrieval, and a pretrained question-answering model from Hugging Face to respond to user queries.

## Table of Contents
- [Overview](#overview)
- [Data Description](#data-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Step-by-Step Implementation](#step-by-step-implementation)
  - [1. Document Loading](#1-document-loading)
  - [2. Text Splitting](#2-text-splitting)
  - [3. Embedding and Vector Store Creation](#3-embedding-and-vector-store-creation)
  - [4. Conversational Retrieval Chain](#4-conversational-retrieval-chain)
  - [5. Interaction Function](#5-interaction-function)
  - [6. Example Queries](#6-example-queries)
- [Example Outputs](#example-outputs)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)


## Overview
RAG is a framework where a language model retrieves relevant context from an external knowledge source and uses it to generate answers. This setup is highly effective for tasks involving specific knowledge bases or complex documents, as it helps maintain relevance in generated answers.

In this project, LangChain’s retrieval and conversational modules, Hugging Face models for question-answering, and FAISS (for fast similarity search) are combined to create a responsive and accurate chatbot.

## Data Description

The data used in this project consists of a 37-page PDF document containing detailed instructions provided by **North America Immigration Law Group (Chen Immigration Law Associates)**. This document is intended for clients seeking guidance on filing the I-140 NIW (National Interest Waiver) form. The instructions outline the necessary steps, required documents, recommendations, and other essential information clients need to successfully prepare their NIW applications.
You can download the data from [here]()

## Project Structure
The project is divided into the following sections:
1. **Document Loading**: Load PDF documents.
2. **Text Splitting**: Split documents into smaller, manageable chunks for better retrieval accuracy.
3. **Embedding and Vector Store Creation**: Use sentence embeddings to create a FAISS index for similarity search.
4. **Conversational Retrieval Chain**: Combine LangChain’s conversational chain with Hugging Face’s pipeline for retrieval and QA.
5. **Interaction**: Implement a function to handle user queries, retrieving relevant context and generating responses.

## Installation
To replicate this setup, you’ll need the following libraries:

```bash
pip install torch transformers langchain faiss-gpu
```
Ensure GPU availability for optimal performance by setting up PyTorch with CUDA support.

## Step-by-Step Implementation

### 1. Document Loading
First, we load the PDF file containing information on case preparation instructions.

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("Attorney_instruction_case_prepration_NIWI_140.pdf")
documents = loader.load()
```

### 2. Text Splitting
To handle large documents effectively, we split the document into smaller chunks using LangChain’s 
`RecursiveCharacterTextSplitter`.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
```
This step ensures that each chunk contains enough context without being too large, enabling efficient retrieval.


### 3. Embedding and Vector Store Creation
We use a Hugging Face embedding model to generate embeddings for each chunk and store them in a FAISS
vector store for efficient retrieval.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings.client = embeddings.client.to("cuda")  # Move the model to GPU

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
```

### 4. Conversational Retrieval Chain
Set up a conversational retrieval chain with LangChain’s `ConversationalRetrievalChain`, using the Hugging Face QA pipeline as the language model.

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load pretrained question-answering model
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# Create a question-answering pipeline with device set to GPU
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Initialize memory to retain conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational retrieval chain
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)
```

### 5. Interaction Function
The `chat_with_bot` function retrieves relevant context from the document and generates an answer based on the user’s question.

```python
def chat_with_bot(question):
    # Retrieve relevant documents based on the question
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_context = " ".join([doc.page_content for doc in retrieved_docs])
    
    # Generate an answer using the QA pipeline with retrieved context
    response = qa_pipeline(question=question, context=retrieved_context)
    return response['answer']
```

### 6. Example Queries

We can now test our chatbot by asking questions. It retrieves relevant sections from the PDF and uses them to answer accurately.

```python
# Example usage
print(chat_with_bot("How much does the filing fee cost?"))  # Expected answer: $700
print(chat_with_bot("Why do I need letters of recommendation?"))  # Expected answer: Explanation of recommendation purpose
```


## Example Outputs

Here are some example questions and expected responses from the chatbot:

- **Question**: "How much does the filing fee cost?"  
  **Answer**: "$700"

- **Question**: "Why do I need letters of recommendation?"  
  **Answer**: "To provide an authoritative perspective on the value and significance of your research."

- **Question**: "How many types of recommenders are there?"  
  **Answer**: "Two."

- **Question**: "What are those two types?"  
  **Answer**: "Dependent and independent."

## Future Enhancements

- **Expand Document Loading**: Support multiple document formats and storage locations.
- **Fine-Tune QA Model**: Improve response quality by fine-tuning the question-answering model on domain-specific data.
- **User Interface**: Implement a web-based UI for real-time interaction with the chatbot.

This setup provides a scalable foundation for building custom RAG-based conversational agents tailored to specific document collections or knowledge bases.

## Conclusion

This project demonstrates the practical use of LangChain and Hugging Face in creating a Retrieval-Augmented Generation (RAG) conversational chatbot. By leveraging FAISS and high-quality embeddings, the chatbot retrieves relevant information, making it suitable for complex knowledge-based applications.









