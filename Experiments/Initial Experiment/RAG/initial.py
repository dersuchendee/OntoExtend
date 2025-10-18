import requests
import json

url = "http://localhost:8000/run"
headers = {"Content-Type": "application/json"}


# INITIAL Experiment:
CQs = [
    [4,'Is a material recognized as a recycled material?'],
    [1,'What are the components of a product?'],
    [2,'What is the composition of a material?'],
    [3,'What is the guidance of of disassembly method?'],
    [0,'What materials does a fibre contain?'],
]

LLMs = [
    "text-embedding-ada-002",
    "qwen3-embedding:4b",
    # 'snowflake-arctic-embed:22m',
    "text-embedding-3-large",
    "text-embedding-3-small",
]

core = "OntoDESIDECoreOntology"

prompts = ['newline','pipe']

for llm in LLMs:
    for prompt in prompts:
        start_rag = 'true'
        for CQ in CQs:
            question = CQ[1]
            payload = {
                "cq": question,
                "start_rag": start_rag,
                "core": core,
                "class_count": 15,
                "op_count": 4,
                "dp_count": 5,
                'prompt': prompt,
                'llm': llm
            }
            print("Sending payload:", json.dumps(payload, indent=2))
            response = requests.post(url, headers=headers, json=payload)
            try:
                result = response.json()
            except json.JSONDecodeError:
                print("Response is not valid JSON:")
                print(response.text)
                result = None

            print(json.dumps(result, indent=2))
            start_rag = 'false'
        # 1/0
        # 1/0