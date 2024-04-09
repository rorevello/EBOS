# paragraph2OWL: Ontology Search Tool

`paragraph2OWL` is an advanced software tool designed to assist researchers in identifying and utilizing relevant ontologies in their fields of study. Capable of accessing the vast database of OBO Foundry, along with functionality for incorporating custom ontologies, `paragraph2OWL` is an indispensable platform for life sciences research and beyond.

## Key Features

- **Access to OBO Foundry Ontologies**: Allows searching among the standardized and well-maintained ontologies available in OBO Foundry, facilitating access to reliable and up-to-date resources.
- **Integration of Custom Ontologies**: Users can extend the utility of `paragraph2OWL` by incorporating their own ontologies, enabling more personalized and relevant searches.
- **Streamlit User Interface**: Provides an intuitive and accessible user interface, designed to simplify the user experience and facilitate the search for ontologies.
- **Direct Visualization and Access to Results**: Presents search results clearly, including detailed information and direct links for documentation and download.

## Format for Custom Ontologies

To incorporate custom ontologies into `paragraph2OWL`, documents must follow the following dictionary format:

```python
{
    'ACRONYM': {
        'title': 'Full Name of the Ontology',
        'description': 'Detailed description of the ontology.',
        'url_doc': 'http://link-to-documentation.com',
        'url_download': 'http://link-to-download-ontology.com',
        'domain': ['Domain1', 'Domain2'],
        'repository': 'Name of the Repository',
        'author': 'Author’s Name'
    }
}
```
This structured format ensures that all custom ontologies are seamlessly integrated into `paragraph2OWL`, allowing for easy access and use by researchers.

## Installation

To install and run `paragraph2OWL`, follow these steps:

```bash
git clone https://github.com/your-username/paragraph2OWL.git
cd paragraph2OWL
pip install -r requirements.txt
```
## Usage

Start the application with the following command:

```bash
streamlit run app.py
```
Then, open your browser and go to `http://localhost:8501` to start using the tool.

## Contributing
We welcome contributions of any kind. If you're interested in helping to improve paragraph2OWL, follow these steps:

Fork the project
Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
Commit your Changes (`git commit -m 'Add some AmazingFeature`')
Push to the Branch (`git push origin feature/AmazingFeature`)
Open a Pull Request

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
