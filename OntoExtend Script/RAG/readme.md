# RAG2 Flask API — Quick Start
You run the script, it reads the core ontology in the Dataset folder, and you get output.ttl, which is a subset of the core ontology wrt the CQ. If it is the first time running the script, put the start_rag flag on 'ture'


Bug or feature?  It returns a certain number of Classes and properties, but if the scores are too low, it does not return classes.

## Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- `ollama`
- `curl` (or Postman) for testing; `jq` is optional but useful for pretty output.

> If you haven't installed Ollama, do so according to Ollama's official docs. Then ensure any required models are available locally.

## Installation
from parent directory install:
`pip install -r requirements.txt`
## Start Ollama (if your RAG uses a local Ollama server)
`ollama serve`

## Start the Flask server
`python service.py`

`pip install gunicorn`

`gunicorn -w 1 -b 0.0.0.0:8000 app_preload:app`

## Test the api in the third terminal:
```
curl -sS -X POST http://localhost:8000/run   -H "Content-Type: application/json"   -d '{
    "cq":"Who owns a product?",
    "start_rag": true,
    "core": "OntoDESIDECoreOntology",
    "class_count": 12,
    "op_count": 4,
    "dp_count": 5
  }' | jq
```



