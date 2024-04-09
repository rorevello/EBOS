import os
import requests

import pandas as pd
import streamlit as st
import tensorflow as tf
from scipy.spatial.distance import cosine
from transformers import TFAutoModel, AutoTokenizer


st.set_page_config(layout="wide")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained(MODEL_NAME)


def get_embedding(text):
    """
    Get the embedding of a given text using the pre-trained model.

    Args:
        text (str): The input text.

    Returns:
        tf.Tensor: The embedding of the text.
    """
    inputs = tokenizer(
        text, return_tensors="tf", padding=True, truncation=True, max_length=512
    )
    outputs = model(inputs)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1)


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
        if response.status_code == 200:
            return str(response.status_code)
        else:
            return str(response.status_code)
    except Exception as e:
        return "500"


def apply_status_color(val):
    """
    Apply color to the status column based on the status code.

    Args:
        val (str): The status code.

    Returns:
        str: The CSS style for the color.
    """
    try:
        if "200" in val:
            color = "green"
        else:
            color = "red"
    except:
        color = "red"
    return f"color: {color};"


def apply_id_color(val):
    """
    Apply color to the ID column.

    Args:
        val (str): The ID.

    Returns:
        str: The CSS style for the color.
    """
    color = "Green"
    return f"color: {color}"


def get_obo_ontologies(ontologies, domains):
    """
    Get OBO ontologies from the OBO Foundry.

    Args:
        ontologies (dict): The dictionary to store the ontologies.
        domains (set): The set to store the domains.

    Returns:
        dict: The updated ontologies dictionary.
        set: The updated domains set.
    """
    response_obo = requests.get("http://obofoundry.org/registry/ontologies.jsonld")
    ontologies_data_obo = response_obo.json()

    for element in ontologies_data_obo["ontologies"]:
        if "description" in element and "ontology_purl" in element:
            ontologies[element["id"]] = {
                "title": element["title"],
                "description": element["description"],
                "url_doc": "http://obofoundry.org/ontology/"
                + str(element["id"])
                + ".html",
                "url_download": element["ontology_purl"],
                "domain": element["domain"],
                "repository": "obo",
            }
            if "domain" in element:
                domains.add(element["domain"])
    return ontologies, domains


def get_local_repo(path_file, ontologies, domains):
    """
    Get ontologies from a local repository.

    Args:
        path_file (str): The path to the local repository file.
        ontologies (dict): The dictionary to store the ontologies.
        domains (set): The set to store the domains.

    Returns:
        dict: The updated ontologies dictionary.
        set: The updated domains set.
    """
    with open(path_file, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()

    diccionario = eval(contenido)
    for clave, valor in diccionario.items():
        if valor["descripcion"]:
            ontologies[clave] = {
                "title": valor["title"],
                "description": valor["descripcion"],
                "url_doc": valor["url_doc"],
                "url_download": valor["url_download"],
                "domain": valor["domain"],
                "repository": "bioportal",
            }
            if len(valor["domain"]) > 0:
                for i in valor["domain"]:
                    domains.add(i)
    return ontologies, domains


st.title("paragraph2OWL")
st.header("Ontology search system using natural language writing.")

user_query = st.text_area("Enter your text here:", "")

list_ontology = ["OBO Foundry"]

db_folder = "db_ontology_local"
files_in_db = os.listdir(db_folder)
list_ontology = files_in_db + ["OBO Foundry"]

selected_domains = st.multiselect(
    "Select Repository(s):", list_ontology, default=["OBO Foundry"]
)
if selected_domains:
    ontologies = {}
    domains = set()
    for repository in selected_domains:
        if repository == "OBO Foundry":
            get_obo_ontologies(ontologies, domains)
        else:
            path_file = db_folder + "/" + repository
            get_local_repo(path_file, ontologies, domains)

    all_domains = list(domains)
    all_domains_option = ["Select All"] + all_domains
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_domains = st.multiselect(
            "Select Domain(s):", ["Select All"] + list(domains), default=["Select All"]
        )

    with col2:
        num_results = st.number_input(
            "Number of Results:", min_value=1, max_value=100, value=5, step=1
        )

    if "Select All" in selected_domains:
        selected_domains = all_domains
    if selected_domains:
        if st.button("Search"):
            with st.spinner("Searching for relevant ontologies..."):
                if user_query:
                    query_embedding = get_embedding(user_query).numpy()
                    similarities = {}
                    for ontology_id, info in ontologies.items():
                        if type(info["domain"]) == list:
                            info["domain"] = ", ".join(info["domain"])
                        if info["domain"] in selected_domains:
                            ontology_embedding = (
                                get_embedding(info["description"]).numpy().flatten()
                            )
                            similarity = 1 - cosine(
                                query_embedding.flatten(), ontology_embedding
                            )
                            similarities[ontology_id] = similarity

                    recommended_ontologies = sorted(
                        similarities.items(), key=lambda x: x[1], reverse=True
                    )[:num_results]
                    table_data = [
                        {
                            "ID": ontology_id,
                            "Title": ontologies[ontology_id]["title"],
                            "Definition": ontologies[ontology_id]["description"],
                            "Urls": " Documentation: "
                            + ontologies[ontology_id]["url_doc"]
                            + " Download: "
                            + ontologies[ontology_id]["url_download"],
                            "Status": check_url_status(
                                ontologies[ontology_id]["url_doc"]
                            ),
                            "Domain": ontologies[ontology_id]["domain"],
                        }
                        for ontology_id, _ in recommended_ontologies
                    ]

                    data_df = pd.DataFrame(table_data)
                    styled_df = data_df.style.applymap(
                        apply_status_color, subset=["Status"]
                    ).applymap(apply_id_color, subset=["ID"])

                    st.table(styled_df)

                else:
                    st.warning("Please enter a query.", icon="🚨")
    else:
        st.warning("Please select at least one domain.", icon="🔥")
else:
    st.warning("Please select only one repository.", icon="🤖")
