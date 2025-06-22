import os
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except PdfReadError as e:
            print(f"Error reading PDF {pdf}: {str(e)}. The file might be corrupted or invalid.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing PDF {pdf}: {str(e)}")
            continue
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Using a sentence-transformer model for embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = SentenceTransformer(model_name)
    
    # Convert texts to embeddings format that FAISS expects
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vector_store

def get_conversational_chain(vector_store):
    # Initialize Hugging Face model
    model_name = "google/flan-t5-base"  # A good default model for conversation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Create LangChain HuggingFace pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def list_available_models():
    """List popular Hugging Face models suitable for conversation and embeddings."""
    conversation_models = [
        "google/flan-t5-base",
        "facebook/blenderbot-400M-distill",
        "microsoft/DialoGPT-medium",
    ]
    
    embedding_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]
    
    print("\nRecommended Conversation Models:")
    print("==============================")
    for model in conversation_models:
        print(f"- {model}")
        
    print("\nRecommended Embedding Models:")
    print("==========================")
    for model in embedding_models:
        print(f"- {model}")
