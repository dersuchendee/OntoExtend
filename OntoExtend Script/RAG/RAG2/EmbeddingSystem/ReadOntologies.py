import os
import tempfile
from owlready2 import get_ontology, default_world
from rdflib import Graph, RDF, RDFS, OWL, URIRef

def merge_ontologies(input_dir = "..\..\..\..\Dataset\OntoDESIDECoreOntology"):
# Set this to your directory containing .ttl files
    

    # Dictionary to collect all prefixes
    prefix_map = {}

    # Step 1: Load all TTL files into owlready2 world
    for fname in os.listdir(input_dir):
        if not fname.endswith(".ttl"):
            continue

        ttl_path = os.path.join(input_dir, fname)

        # Parse TTL with rdflib to extract prefixes and convert to N-Triples
        g = Graph()
        g.parse(ttl_path, format="turtle")

        # Collect prefixes
        for prefix, ns in g.namespaces():
            if prefix not in prefix_map:
                prefix_map[prefix] = ns

        # Save as temporary N-Triples for owlready2 to load
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nt") as tmp:
            g.serialize(destination=tmp.name, format="nt")
            tmp_path = tmp.name

        # Load with owlready2
        get_ontology(f"file://{tmp_path}").load()

    # Step 2: Export the combined RDF graph from owlready2's world
    merged_graph = default_world.as_rdflib_graph()

    # Step 3: Re-bind original prefixes
    for prefix, ns in prefix_map.items():
        merged_graph.bind(prefix, ns)

    # Step 4: Save merged graph as Turtle
    merged_graph.serialize(destination="RAG2/merged.ttl", format="turtle")
    return open("RAG2/merged.ttl", "r",encoding= 'UTF-8').read()  # Ensure newline at end of file


def Fetch_components(owl_path = "RAG2/merged.ttl"):
    # print(owl_path)
    # Path to your TTL
    

    g = Graph()
    g.parse(owl_path, format="turtle")

    def local_name(node: URIRef) -> str:
        """Return the fragment or last path segment of a URIRef."""
        uri = str(node)
        if "#" in uri:
            return uri.split("#")[-1]
        else:
            return uri.rstrip("/").split("/")[-1]

    def get_labels(uri):
        return [str(label) for label in g.objects(uri, RDFS.label)]

    def get_comments(uri):
        return [str(comment) for comment in g.objects(uri, RDFS.comment)]

    def get_parents(uri):
        return [local_name(parent) for parent in g.objects(uri, RDFS.subClassOf)]

    def get_domains(uri):
        domains = list(g.objects(uri, RDFS.domain))
        if len(domains) > 1:
            return " AND ".join(local_name(d) for d in domains)
        elif domains:
            return local_name(domains[0])
        return None

    def get_ranges(uri):

        ranges = list(g.objects(uri, RDFS.range))
        if len(ranges) > 1:
            return " AND ".join(local_name(r) for r in ranges)
        elif ranges:
            return local_name(ranges[0])
        return None

    def print_entity_info(uri, entity_type=0):
        Result_temp = {
            'Type': entity_type or '',#class, object property, data property
            'Labels':", ".join(get_labels(uri)) or '',#all labels
            'Comments':", ".join(get_comments(uri)) or '',#all comments
            'Parents':", ".join(get_parents(uri)) or "",#all parents
            'URI':str(uri) or '',#full URI
            'Domain':get_domains(uri) or "",#for properties
            'Range':get_ranges(uri) or "",#for properties
        }
        Result = {k:v for k,v in Result_temp.items() if v != ''}
        return Result
    Results = []
    # 1) Classes
    for s in g.subjects(RDF.type, OWL.Class):
        Results .append(print_entity_info(s, "Class"))
    # print('len of classes:',len(Results))
    # 2) Object Properties
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        Results .append(print_entity_info(s, "ObjectProperty"))
    # print('len of classes:',len(Results))

    # 3) Data Properties
    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        Results .append(print_entity_info(s, "DatatypeProperty"))
    # print('len of classes:',len(Results))

    # print(Results)
    return Results

