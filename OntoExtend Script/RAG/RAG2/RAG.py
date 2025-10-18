import sys
import os
import numpy as np
import pandas as pd
import os
from RAG2.EmbeddingSystem.ReadOntologies import merge_ontologies, Fetch_components
from pathlib import Path
import re
import faiss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from JavadEmbedders import OllamaEmbedderQWEN,LiUAzureEmbedder
import time
from owlready2 import *
from rdflib import Graph, Namespace, URIRef, BNode, RDF, RDFS, OWL
from rdflib.namespace import SKOS
import sys









def init_rag(core_ontology_path,batch_size = 40,llm='qwen3-embedding:4b',prompt='newline'):
    merged_ttl_content = merge_ontologies(core_ontology_path)
    def make_prompt(comp,prompt):
        Prompt = ''
        for key, value in comp.items():
            if key == 'Internal_RAG_ID' or key == 'Type':
                continue
            if prompt == 'newline':
                Prompt += value + '\n'
            if prompt == 'pipe':
                Prompt += value + ' |'
        return Prompt

    components = Fetch_components('RAG2/merged.ttl')
    Prompts_classes = {}
    Prompts_classes_full_info = {}
    Prompts_OP = {}
    Prompts_OP_full_info = {}
    Prompts_DP = {}
    Prompts_DP_full_info = {}
    Internal_RAG_ID = 0
    for comp in components:
        if comp['Type'] == 'Class':
            Prompts_classes[Internal_RAG_ID] = make_prompt(comp,prompt)
            Prompts_classes_full_info[Internal_RAG_ID] = comp
        elif comp['Type'] == 'ObjectProperty':
            Prompts_OP[Internal_RAG_ID] = make_prompt(comp,prompt)
            Prompts_OP_full_info[Internal_RAG_ID] = comp
        elif comp['Type'] == 'DatatypeProperty':
            Prompts_DP[Internal_RAG_ID] = make_prompt(comp,prompt)
            Prompts_DP_full_info[Internal_RAG_ID] = comp
        else:
            continue
        Internal_RAG_ID += 1

    def CreateEmbeddingSpace(Prompts_dict,type_str,llm): #embeds all classes and properties and save to files
        keys = list(Prompts_dict.keys())
        values = list(Prompts_dict.values())
        print('Length of the keys: ', len(keys))
        # open('loglog.txt','a').write('\n'.join(values))
        # print(keys)
        
        # print(values)
        for i in range(len(keys)):
            f = open('RAG2/EmbeddingSystem/'+type_str+'es_prompts.txt','a',encoding='utf-8')
            f.write(str(keys[i]) + '\n')
            p = values[i].replace('\n','\t\t')
            f.write(p + '\n')
        for i in range(0, len(keys), batch_size):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f" Processing batch {i//batch_size + 1} / {(len(keys)-1)//batch_size + 1}")
            batch_keys = keys[i:i+batch_size]
            batch_values = values[i:i+batch_size]

            # if llm == 'qwen3-embedding:4b':
            #     embeddings = OllamaEmbedderQWEN(batch_values)
            if 'text-embedding' in llm:
                embeddings = LiUAzureEmbedder(batch_values,llm)
            else:
                embeddings = OllamaEmbedderQWEN(batch_values,llm=llm)

   
            try:
                prev_embeddings = open('RAG2/EmbeddingSystem/'+type_str+'es_vecs.txt','r',encoding='utf-8').readlines()
            except:
                prev_embeddings = []
            f = open('RAG2/EmbeddingSystem/'+type_str+'es_vecs.txt','w',encoding='utf-8')
            for prev_line in prev_embeddings:
                f.write(prev_line)
            for j in range(len(batch_keys)):
                # try:
                    temp = [str(x) for x in embeddings[j]]
                    f.write(str(batch_keys[j]) + '\n')
                    f.write(','.join(temp) + '\n')
                # except:
                #     if type_str == 'Class':
                #         prompt_temp = Prompts_classes[j]
                #     if type_str == 'DataProperty':
                #         prompt_temp = Prompts_DP[j]
                #     if type_str == 'ObjectProperty':
                #         prompt_temp = Prompts_OP[j]
                #     if 'text-embedding' in llm:
                #         temp = LiUAzureEmbedder(prompt_temp,llm)[0]
                #     else:
                #         temp = OllamaEmbedderQWEN(prompt_temp,llm=llm)[0]

                #     f.write(str(batch_keys[j]) + '\n')
                #     f.write(','.join(temp) + '\n')

    CreateEmbeddingSpace(Prompts_classes,'Class',llm=llm)
    CreateEmbeddingSpace(Prompts_OP,'ObjectProperty',llm=llm)
    CreateEmbeddingSpace(Prompts_DP,'DataProperty',llm=llm)

    np.save('RAG2/EmbeddingSystem/Prompts_full_info.npy', np.array([Prompts_classes_full_info, Prompts_OP_full_info, Prompts_DP_full_info], dtype=object))


def RAG_extract_URIs(Query,class_count=15, op_count=5, dp_count=5,llm='qwen3-embedding:4b'):

    def RetriveComponents(Query_vector, class_count=15, op_count=5, dp_count=5):
        def Retrive(Query_vector, top_k=5,type_str='Class'):
            prompts_keys = {}
            lines = open('RAG2/EmbeddingSystem/'+type_str+'es_prompts.txt','r',encoding='utf-8').readlines()
            for i in range(0,len(lines),2):
                prompts_keys[int(lines[i].strip())] = lines[i+1].strip().replace('\t\t','\n')
            data = {}
            raw_data = open('RAG2/EmbeddingSystem/'+type_str+'es_vecs.txt','r',encoding='utf-8').readlines()
            for i in range(0,len(raw_data),2):
                data[int(raw_data[i])] = [float(x) for x in raw_data[i+1].strip().split(',')]
            data = {k:np.array(v,dtype=np.float32) for k,v in data.items()}

            keys = list(data.keys())
            vectors = np.stack(list(data.values()))  # shape: (num_items, dim)
            
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product
            

            index.add(vectors)
            if Query_vector.ndim == 1:
                Query_vector = Query_vector.reshape(1, -1)
            scores, indices = index.search(np.array(Query_vector,dtype=np.float32), top_k)#;print('hi1')
            nearest_keys = [keys[i] for i in indices[0]]
            return nearest_keys, scores[0],{k:prompts_keys[k] for k in nearest_keys}
            
        Class_nearest_keys, Class_scores, Class_info = Retrive(Query_vector, top_k=class_count,type_str='Class') 
        OP_nearest_keys, OP_scores, OP_info = Retrive(Query_vector, top_k=op_count,type_str='ObjectProperty') 
        DP_nearest_keys, DP_scores, DP_info = Retrive(Query_vector, top_k=dp_count,type_str='DataProperty')  

        return Class_nearest_keys, Class_scores, Class_info, \
            OP_nearest_keys, OP_scores, OP_info,\
            DP_nearest_keys, DP_scores, DP_info

    # if llm == 'qwen3-embedding:4b':
    #     Query_vector = OllamaEmbedderQWEN([Query])[0]
    if  'text-embedding' in llm:
        Query_vector = LiUAzureEmbedder([Query],llm)[0]
    else:
        Query_vector = OllamaEmbedderQWEN([Query],llm=llm)[0]


    Class_nearest_keys, Class_scores, Class_info, \
            OP_nearest_keys, OP_scores, OP_info,\
            DP_nearest_keys, DP_scores, DP_info = RetriveComponents(Query_vector,class_count, op_count, dp_count)

    RAG_return = [[Class_nearest_keys, Class_scores, Class_info],
                [OP_nearest_keys, OP_scores, OP_info],
                    [DP_nearest_keys, DP_scores, DP_info]]

    Prompts_classes, Prompts_OP, Prompts_DP = np.load('RAG2/EmbeddingSystem/Prompts_full_info.npy', allow_pickle=True)
    components = {}
    components.update(Prompts_classes)
    components.update(Prompts_OP)
    components.update(Prompts_DP)
    URIs = []
    for item in RAG_return:
        nearestClasses, nearestClassesScores, info = item
        for k,v in info.items():
            URIs.append(components[k]['URI'])

    labels,comments = [],[]
    for item in RAG_return:
        nearestClasses, nearestClassesScores, info = item
        for k,v in info.items():
            lbl = '' if 'Labels' not in components[k] else components[k]['Labels']
            cms = '' if 'Comments' not in components[k] else components[k]['Comments']
            labels.append(lbl)
            comments.append(cms)

    return URIs ,RAG_return,labels,comments
  

def extract_blank_node_triples(g, bnode, output_g, visited=None, uri_refs_to_add=None):
    if visited is None:
        visited = set()
    if uri_refs_to_add is None:
        uri_refs_to_add = set()
    if bnode in visited:
        return uri_refs_to_add
    visited.add(bnode)
    # Get all triples where the blank node is the subject
    for s, p, o in g.triples((bnode, None, None)):
        output_g.add((s, p, o))
        # If this is an OWL restriction with onProperty, collect the property URI
        if p == OWL.onProperty and isinstance(o, URIRef):
            uri_refs_to_add.add(o)
        
        # If the object is also a blank node, recurse
        if isinstance(o, BNode):
            extract_blank_node_triples(g, o, output_g, visited, uri_refs_to_add)
    
    # Get all triples where the blank node is the object
    for s, p, o in g.triples((None, None, bnode)):
        output_g.add((s, p, o))
        # If the subject is also a blank node, recurse
        if isinstance(s, BNode):
            extract_blank_node_triples(g, s, output_g, visited, uri_refs_to_add)
    
    return uri_refs_to_add

def extract_ontology(input_ttl, uri_list, output_ttl):
    g = Graph()
    g.parse(input_ttl, format='turtle')

    output_g = Graph()
    
    # Copy namespace bindings
    for prefix, namespace in g.namespaces():
        output_g.bind(prefix, namespace)
    
    # Convert URI strings to URIRef objects
    uri_refs = [URIRef(uri) if isinstance(uri, str) else uri for uri in uri_list]
    additional_uris = set()

    extracted_count = 0
    for uri in uri_refs:
        for s, p, o in g.triples((uri, None, None)):
            output_g.add((s, p, o))
            extracted_count += 1
            # If object is a blank node, extract its triples too
            if isinstance(o, BNode):
                referenced_uris = extract_blank_node_triples(g, o, output_g)
                additional_uris.update(referenced_uris)
    if additional_uris:
        for uri in additional_uris:
            # print(f"  - {uri}")
            for s, p, o in g.triples((uri, None, None)):
                output_g.add((s, p, o))
                extracted_count += 1
                # Also handle blank nodes in these properties
                if isinstance(o, BNode):
                    extract_blank_node_triples(g, o, output_g)
    output_g.serialize(destination=output_ttl, format='turtle', encoding='utf-8')



def RAG(Query="What are the components of a product?",init_rag_flag=False,class_count=10, op_count=5,
         dp_count=3,core='OntoDESIDECoreOntology',llm='qwen3-embedding:4b',prompt='newline'):
    dataset_path = '../../Dataset/'+core

    if init_rag_flag:
        directory = Path("RAG2/EmbeddingSystem")

        # Regex pattern
        pattern = re.compile(r".*\.(txt|npy)$")

        # Loop through files and delete
        for file_path in directory.iterdir():
            if pattern.match(file_path.name):
                file_path.unlink()  # deletes the file
                print(f"Deleted: {file_path}")
        file_path = 'output.ttl'
        if os.path.exists(file_path):
            os.remove(file_path)
            print("output removed.")
        else:
            print("output does not exist.")
        file_path = 'RAG2/merged.ttl'
        if os.path.exists(file_path):
            os.remove(file_path)
            print("merged removed.")
        else:
            print("merged does not exist.")
        init_rag(core_ontology_path = dataset_path,llm=llm,prompt=prompt)

    input_file = "RAG2/merged.ttl"
    #RAG_return = [[Class_nearest_keys, Class_scores, Class_info],
                #[OP_nearest_keys, OP_scores, OP_info],
                 #   [DP_nearest_keys, DP_scores, DP_info]]



    URIs , RAG_return,labels,comments = RAG_extract_URIs(Query, class_count,
                                                          op_count, dp_count,llm=llm)

    # print('URIs',URIs)
    import datetime
    log  = {
    'TimeStamp' : [datetime.datetime.now() for i in range(len(URIs))],
    "CQ": [Query for i in range(len(URIs))],
    "LLM": [llm for i in range(len(URIs))],
    'Entity URI':URIs,
    'Entity Type':  ['Class' for i in range(len(RAG_return[0][1]))]+\
                    ['ObjectProperty' for i in range(len(RAG_return[1][1]))]+\
                    ['DataProperty' for i in range(len(RAG_return[2][1]))],
    'Similarity Score':list(RAG_return[0][1])+list(RAG_return[1][1])+list(RAG_return[2][1]),
    'PromptType':[prompt for i in range(len(URIs))],
    }
    # print('printing log size')
    # for k,v in log.items():
    #     print(k,len(v))
    df = pd.DataFrame(log)
    filename = "Log.csv"
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, index=False)

    output_file = "output.ttl"




    for uri in URIs:
        print(uri)
    extract_ontology(input_file, URIs, output_file)
    return open(output_file, 'r', encoding='utf-8').read()



