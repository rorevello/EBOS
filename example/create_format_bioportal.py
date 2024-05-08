import requests


api_key = 
headers = {"Authorization": "apikey token="+api_key}
url_portal = "https://data.bioontology.org/ontologies"
response_bioportal = requests.get(
    url_portal, headers=headers
)

ontologies = response_bioportal.json()
ontology_acronyms = [ontology["acronym"] for ontology in ontologies]
dict_ontologies = {}
for acronym in ontology_acronyms:
    response = requests.get(
        url_portal + acronym + "/submissions",
        headers=headers,
    )
    try:
        if response.status_code == 200 and response.json() != []:
            ontology_data = response.json()
            title = ontology_data[0]["ontology"]["name"]
            descripcion = ontology_data[0]["description"]
            url_link_ui = ontology_data[0]["ontology"]["links"]["ui"]
            url_download = ontology_data[0]["ontology"]["links"]["download"]
            repository = "bioportal"
            autor = ontology_data[0]["contact"][0]["name"]
            response2 = requests.get(
                url_portal + acronym + "/categories",
                headers=headers,
            )
            if response2.status_code == 200:
                categories = response2.json()
                categories = [category["name"] for category in categories]
            elif response2.status_code == 404:
                categories = []
            info = {
                "title": str(title),
                "descripcion": str(descripcion),
                "url_doc": url_link_ui,
                "url_download": url_download,
                "domain": categories,
                "repository": repository,
                "autor": autor,
            }
            dict_ontologies[acronym] = info

    except:
        pass

with open("dict_ontology.txt", "w") as file:
    for ontology in dict_ontologies.values():
        file.write(str(ontology) + "\n")
