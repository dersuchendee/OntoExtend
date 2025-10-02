import sys
import os
from EmbeddingSystem.ReadOntologies import merge_ontologies, Fetch_components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from OllamaEmbedder import OllamaEmbedderQWEN


def init_rag():
    merged_ttl_content = merge_ontologies('..\..\..\Dataset\OntoDESIDECoreOntology')
    print((merged_ttl_content)) 



# init_rag()
Fetch_components('merged.ttl')


# Add OntoExtend Script directory to sys.path

# print(OllamaEmbedderQWEN(['hi','hello']))