import faiss
import numpy as np
import pandas as pd
import streamlit as st

def create_index(embeddings):
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings (np.ndarray): The embeddings to index.
        
    Returns:
        faiss.IndexFlatL2: The FAISS index.
    """
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_ontologies(query_embedding, index, ontology_ids, num_results=5, ontologies_data=None, query_text=""):
    """
    Search for similar ontologies and apply keyword boosting.
    
    Args:
        query_embedding (np.ndarray): The embedding of the query.
        index (faiss.Index): The FAISS index.
        ontology_ids (list): List of ontology IDs corresponding to the index.
        num_results (int): Number of results to return.
        ontologies_data (dict, optional): Dictionary of ontology data for re-ranking.
        query_text (str, optional): Original query text for keyword matching.
        
    Returns:
        list of tuple: List of (ontology_id, score).
    """
    if index is None or len(ontology_ids) == 0:
        return []
    
    # Fetch a large pool of candidates for re-ranking
    # If the user query has a keyword match, we want to find it even if semantic score is low.
    search_k = min(100, len(ontology_ids)) 
    
    if search_k == 0: 
        return []

    D, I = index.search(np.array([query_embedding]), search_k)
    
    results = []
    # Initial Semantic Results
    for i in range(search_k):
        idx = I[0][i]
        if idx < 0 or idx >= len(ontology_ids): 
            continue
            
        ontology_id = ontology_ids[idx]
        score = D[0][i] # L2 Distance
        
        # Keyword Boosting
        if ontologies_data and query_text:
            data = ontologies_data.get(ontology_id)
            if data:
                term = query_text.lower().strip()
                title = str(data.get('title', '')).lower()
                desc = str(data.get('description', '')).lower()
                
                # Boost Factor 
                # L2 distances are typically between 0.5 and 1.5 for these models.
                # A boost of 1.0 is massive, essentially guaranteeing top placement.
                boost = 0.0
                if term and term in title:
                    boost += 1.0 
                if term and term in desc:
                    boost += 0.5 
                
                score = max(0, score - boost)
        
        results.append((ontology_id, score))
    
    # Re-sort based on boosted score (ascending distance)
    results.sort(key=lambda x: x[1])
    
    return results[:num_results]

def format_results(results, ontologies):
    """
    Format search results for display.
    
    Args:
        results (list): List of tuples (ontology_id, distance).
        ontologies (dict): Dictionary of ontology data.
        
    Returns:
        pd.DataFrame: Formatted DataFrame.
    """
    table_data = []
    for ontology_id, _ in results:
        data = ontologies.get(ontology_id)
        if data:
            domain = data.get("domain", "Unknown")
            if isinstance(domain, list):
                domain = ", ".join(domain)
            elif domain is None:
                domain = "Unknown"
            
            table_data.append({
                "ID": ontology_id,
                "Title": data.get("title", "No Title"),
                "Definition": data.get("description", "No Description"),
                "Urls": f"Documentation: {data.get('url_doc', 'N/A')} | Download: {data.get('url_download', 'N/A')}",
                "Status": "Check Enabled" if data.get('url_doc') else "N/A", # Placeholder, logic moved to main
                "Domain": domain,
                "url_doc": data.get('url_doc', '') # keep hidden for status check
            })
            
    return pd.DataFrame(table_data)
