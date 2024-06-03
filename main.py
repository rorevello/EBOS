import os
import requests
import pandas as pd
import streamlit as st
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import faiss
import numpy as np

# Page configuration
st.set_page_config(layout="wide")

# Load the model and tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    """
    Get the embedding of a given text using the pre-trained model.
    
    Args:
        text (str): The input text.
    
    Returns:
        np.ndarray: The embedding of the text.
    """
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy().flatten()
    return embedding

def check_url_status(url):
    """
    Check the status code of a given URL.
    
    Args:
        url (str): The URL to check.
    
    Returns:
        str: The status code of the URL.
    """
    try:
        response = requests.head(url, timeout=20)
        return str(response.status_code)
    except requests.RequestException:
        return "500"

def apply_status_color(val):
    """
    Apply color to the status column based on the status code.
    
    Args:
        val (str): The status code.
    
    Returns:
        str: The CSS style for the color.
    """
    color = 'green' if val == '200' else 'red'
    return f"color: {color};"

def apply_id_color(val):
    """
    Apply color to the ID column.
    
    Args:
        val (str): The ID.
    
    Returns:
        str: The CSS style for the color.
    """
    return "color: Green;"

def get_obo_ontologies():
    """
    Get OBO ontologies from the OBO Foundry.
    
    Returns:
        dict: Dictionary with the ontologies.
        set: Set of domains.
    """
    ontologies = {}
    domains = set()
    try:
        response = requests.get("http://obofoundry.org/registry/ontologies.jsonld")
        response.raise_for_status()
        ontologies_data = response.json()

        for element in ontologies_data.get("ontologies", []):
            if "description" in element and "ontology_purl" in element:
                domain = element.get("domain", "Unknown")  # Use "Unknown" if domain is missing
                ontologies[element["id"]] = {
                    "title": element["title"],
                    "description": element["description"],
                    "url_doc": f"http://obofoundry.org/ontology/{element['id']}.html",
                    "url_download": element["ontology_purl"],
                    "domain": domain,
                    "repository": "obo",
                }
                domains.add(domain)
    except requests.RequestException as e:
        st.error(f"Error fetching OBO ontologies: {e}")
    return ontologies, domains

def get_local_repo(path_file):
    """
    Get ontologies from a local repository.
    
    Args:
        path_file (str): The path to the local repository file.
    
    Returns:
        dict: Dictionary with the ontologies.
        set: Set of domains.
    """
    ontologies = {}
    domains = set()
    try:
        with open(path_file, "r", encoding="utf-8") as file:
            content = eval(file.read())
        for key, value in content.items():
            if value["descripcion"]:
                domain = value.get("domain", "Unknown")  # Use "Unknown" if domain is missing
                ontologies[key] = {
                    "title": value["title"],
                    "description": value["descripcion"],
                    "url_doc": value["url_doc"],
                    "url_download": value["url_download"],
                    "domain": domain,
                    "repository": "local",
                }
                if isinstance(domain, list):
                    domains.update(domain)
                else:
                    domains.add(domain)
    except (FileNotFoundError, SyntaxError) as e:
        st.error(f"Error reading local repository: {e}")
    return ontologies, domains

# Streamlit application
st.title("EBOS")
st.header("Ontology search system using natural language writing.")

user_query = st.text_area("Enter your text here:", "")

# Fetch list of available ontologies
db_folder = "db_ontology_local"
list_ontology = os.listdir(db_folder) + ["OBO Foundry"]

selected_repositories = st.multiselect("Select Repository(s):", list_ontology, default=["OBO Foundry"])

if "ontologies" not in st.session_state:
    st.session_state.ontologies = {}
    st.session_state.domains = set()
    st.session_state.ontology_embeddings = []
    st.session_state.ontology_ids = []

if selected_repositories:
    for repository in selected_repositories:
        if repository == "OBO Foundry":
            repo_ontologies, repo_domains = get_obo_ontologies()
        else:
            path_file = os.path.join(db_folder, repository)
            repo_ontologies, repo_domains = get_local_repo(path_file)
        st.session_state.ontologies.update(repo_ontologies)
        st.session_state.domains.update(repo_domains)

    all_domains = list(st.session_state.domains)
    selected_domains = st.multiselect("Select Domain(s):", ["Select All"] + all_domains, default=["Select All"])

    if "Select All" in selected_domains:
        selected_domains = all_domains

    num_results = st.number_input("Number of Results:", min_value=1, max_value=100, value=5, step=1)

    # Vectorize ontologies if not already done
    if not st.session_state.ontology_embeddings:
        with st.spinner("Vectorizing ontologies..."):
            for ontology_id, info in st.session_state.ontologies.items():
                if isinstance(info["domain"], list):
                    info["domain"] = ", ".join(info["domain"])
                ontology_embedding = get_embedding(info["description"])
                st.session_state.ontology_embeddings.append(ontology_embedding)
                st.session_state.ontology_ids.append(ontology_id)
            st.session_state.ontology_embeddings = np.array(st.session_state.ontology_embeddings)

        # Prepare FAISS index
        dimension = st.session_state.ontology_embeddings.shape[1]
        st.session_state.index = faiss.IndexFlatL2(dimension)
        st.session_state.index.add(st.session_state.ontology_embeddings)

    if selected_domains and st.button("Search"):
        if user_query:
            with st.spinner("Searching for relevant ontologies..."):
                query_embedding = get_embedding(user_query).reshape(1, -1)

                D, I = st.session_state.index.search(query_embedding, num_results)
                recommended_ontologies = [(st.session_state.ontology_ids[i], D[0][j]) for j, i in enumerate(I[0])]

                table_data = [
                    {
                        "ID": ontology_id,
                        "Title": st.session_state.ontologies[ontology_id]["title"],
                        "Definition": st.session_state.ontologies[ontology_id]["description"],
                        "Urls": f" Documentation: {st.session_state.ontologies[ontology_id]['url_doc']} Download: {st.session_state.ontologies[ontology_id]['url_download']}",
                        "Status": check_url_status(st.session_state.ontologies[ontology_id]["url_doc"]),
                        "Domain": st.session_state.ontologies[ontology_id]["domain"],
                    }
                    for ontology_id, _ in recommended_ontologies
                ]

                data_df = pd.DataFrame(table_data)
                styled_df = data_df.style.applymap(apply_status_color, subset=["Status"]).applymap(apply_id_color, subset=["ID"])

                st.table(styled_df)
        else:
            st.warning("Please enter a query.", icon="🚨")
else:
    st.warning("Please select at least one repository.", icon="🔥")
