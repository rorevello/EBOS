import os
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from . import config

@st.cache_resource
def load_model():
    """Load and cache the SentenceTransformer model."""
    # SentenceTransformer handles downloading and caching internally
    # It also handles the backend (torch/safetensors) automatically
    model = SentenceTransformer(config.MODEL_NAME)
    return None, model # Tokenizer is internal to the model

def get_embedding_path(repo_name):
    """Get the path to the cached embeddings file."""
    # Include model name hash or simplified name to avoid collisions
    model_suffix = config.MODEL_NAME.split('/')[-1]
    return os.path.join(config.DATA_DIR, f"{repo_name}_{model_suffix}_embeddings.pkl")

def save_embeddings(repo_name, embeddings, ids):
    """Save embeddings and IDs to disk."""
    path = get_embedding_path(repo_name)
    with open(path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "ids": ids}, f)

def load_embeddings(repo_name):
    """Load embeddings and IDs from disk."""
    path = get_embedding_path(repo_name)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data["embeddings"], data["ids"]
        except Exception as e:
            print(f"Error loading embeddings for {repo_name}: {e}")
    return None, None

def get_embedding(text, tokenizer, model):
    """
    Get the embedding of a given text using SentenceTransformer.
    
    Args:
        text (str): The input text.
        tokenizer: Unused (kept for compatibility).
        model: The SentenceTransformer model.
    
    Returns:
        np.ndarray: The embedding of the text.
    """
    # model.encode returns a numpy array by default
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding
