import os
import requests
import streamlit as st
import ast
from . import config

@st.cache_data
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
        print("Fetching OBO ontologies...")
        # Reduced timeout to prevents hangs
        response = requests.get("http://obofoundry.org/registry/ontologies.jsonld", timeout=3)
        response.raise_for_status()
        ontologies_data = response.json()

        for element in ontologies_data.get("ontologies", []):
            if "description" in element and "ontology_purl" in element:
                domain = element.get("domain", "Unknown")
                if isinstance(domain, str):
                    domain = [domain]

                ontologies[element["id"]] = {
                    "title": element["title"],
                    "description": element["description"],
                    "url_doc": f"http://obofoundry.org/ontology/{element['id']}.html",
                    "url_download": element["ontology_purl"],
                    "domain": domain,
                    "repository": "obo",
                    "license": element.get("license", {}).get("label", "Unknown"),
                    "activity_status": element.get("activity_status", "active"),
                    "products": len(element.get("products", [])),
                }
                domains.update(domain)
    except requests.RequestException as e:
        print(f"Error fetching OBO ontologies: {e}")
    return ontologies, domains

@st.cache_data
def get_local_repo(filename):
    """
    Get ontologies from a local repository file.
    
    Args:
        filename (str): The name of the local repository file.
    
    Returns:
        dict: Dictionary with the ontologies.
        set: Set of domains.
    """
    ontologies = {}
    domains = set()
    path_file = os.path.join(config.DATA_DIR, filename)
    
    if not os.path.exists(path_file):
        st.error(f"File not found: {path_file}")
        return ontologies, domains

    try:
        with open(path_file, "r", encoding="utf-8") as file:
            content = file.read()
            # Use ast.literal_eval for safety instead of eval
            content = ast.literal_eval(content)
            
        for key, value in content.items():
            # Handle domain (ensure list)
            domain = value.get("domain", ["Unknown"])
            if isinstance(domain, str):
                domain = [domain]

            # Handle description (fallback to 'descripcion')
            description = value.get("description") or value.get("descripcion", "No description available")

            ontologies[key] = {
                "title": value.get("title", "Unknown"),
                "description": description,
                "url_doc": value.get("url_doc", ""),
                "url_download": value.get("url_download", ""),
                "domain": domain,
                "repository": "local",
                "license": "Unknown",
                "activity_status": "active",
                "products": 0,
            }
            domains.update(domain) # domain is now guaranteed to be a list
    except (FileNotFoundError, SyntaxError, ValueError) as e:
        st.error(f"Error reading local repository {filename}: {e}")
            
    return ontologies, domains

def get_available_repositories():
    """List available local repositories."""
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        return []
    return [f for f in os.listdir(config.DATA_DIR) if f.endswith('.txt') or f.endswith('.json')]
