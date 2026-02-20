import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "ontology_repository_metadata")
# If the folder doesn't exist, try the old name or creating it.
if not os.path.exists(DATA_DIR):
     DATA_DIR = os.path.join(BASE_DIR, "db_ontology_local")

print(f"DEBUG: BASE_DIR={BASE_DIR}")
print(f"DEBUG: DATA_DIR={DATA_DIR}")

# Model
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Search
DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 100
