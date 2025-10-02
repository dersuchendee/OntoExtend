import sys
import os
import numpy as np
from EmbeddingSystem.ReadOntologies import merge_ontologies, Fetch_components
import faiss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from OllamaEmbedder import OllamaEmbedderQWEN
import time


def init_rag(batch_size = 20,core_ontology_path = '..\..\..\Dataset\OntoDESIDECoreOntology'):
    merged_ttl_content = merge_ontologies(core_ontology_path)
    def make_prompt(comp):
        Prompt = ''
        for key, value in comp.items():
            if key == 'Internal_RAG_ID' or key == 'Type':
                continue
            Prompt += value + '\n'
        return Prompt

    components = Fetch_components('merged.ttl')
    Prompts_classes = {}
    Prompts_classes_full_info = {}
    Prompts_OP = {}
    Prompts_OP_full_info = {}
    Prompts_DP = {}
    Prompts_DP_full_info = {}
    Internal_RAG_ID = 0
    for comp in components:
        if comp['Type'] == 'Class':
            Prompts_classes[Internal_RAG_ID] = make_prompt(comp)
            Prompts_classes_full_info[Internal_RAG_ID] = comp
        elif comp['Type'] == 'ObjectProperty':
            Prompts_OP[Internal_RAG_ID] = make_prompt(comp)
            Prompts_OP_full_info[Internal_RAG_ID] = comp
        elif comp['Type'] == 'DatatypeProperty':
            Prompts_DP[Internal_RAG_ID] = make_prompt(comp)
            Prompts_DP_full_info[Internal_RAG_ID] = comp
        else:
            continue
        Internal_RAG_ID += 1

    def CreateEmbeddingSpace(Prompts_dict,type_str): #embeds all classes and properties and save to files
        keys = list(Prompts_dict.keys())
        values = list(Prompts_dict.values())
        for i in range(len(keys)):
            f = open('EmbeddingSystem/'+type_str+'es_prompts.txt','a',encoding='utf-8')
            f.write(str(keys[i]) + '\n')
            p = values[i].replace('\n','\t\t')
            f.write(p + '\n')
        for i in range(0, len(keys), batch_size):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f" Processing batch {i//batch_size + 1} / {(len(keys)-1)//batch_size + 1}")
            batch_keys = keys[i:i+batch_size]
            batch_values = values[i:i+batch_size]
            embeddings = OllamaEmbedderQWEN(batch_values)
            try:
                prev_embeddings = open('EmbeddingSystem/'+type_str+'es_vecs.txt','r',encoding='utf-8').readlines()
            except:
                prev_embeddings = []
            f = open('EmbeddingSystem/'+type_str+'es_vecs.txt','w',encoding='utf-8')
            for prev_line in prev_embeddings:
                f.write(prev_line)
            for j in range(len(batch_keys)):
                f.write(str(batch_keys[j]) + '\n')
                f.write(','.join([str(x) for x in embeddings[j]]) + '\n')

    CreateEmbeddingSpace(Prompts_classes,'Class')
    CreateEmbeddingSpace(Prompts_OP,'ObjectProperty')
    CreateEmbeddingSpace(Prompts_DP,'DataProperty')

    np.save('EmbeddingSystem/Prompts_full_info.npy', np.array([Prompts_classes_full_info, Prompts_OP_full_info, Prompts_DP_full_info], dtype=object))


def RetriveComponents(Query_vector, class_count=15, op_count=5, dp_count=5):
    def Retrive(Query_vector, top_k=5,type_str='Class'):
        prompts_keys = {}
        lines = open('EmbeddingSystem/'+type_str+'es_prompts.txt','r',encoding='utf-8').readlines()
        for i in range(0,len(lines),2):
            prompts_keys[int(lines[i].strip())] = lines[i+1].strip().replace('\t\t','\n')
        data = {}
        raw_data = open('EmbeddingSystem/'+type_str+'es_vecs.txt','r',encoding='utf-8').readlines()
        for i in range(0,len(raw_data),2):
            data[int(raw_data[i])] = [float(x) for x in raw_data[i+1].strip().split(',')]
        data = {k:np.array(v,dtype=np.float32) for k,v in data.items()}

        keys = list(data.keys())
        vectors = np.stack(list(data.values()))  # shape: (num_items, dim)
        
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product
        

        index.add(vectors)
        scores, indices = index.search(np.array(Query_vector,dtype=np.float32), top_k)#;print('hi1')
        nearest_keys = [keys[i] for i in indices[0]]
        return nearest_keys, scores[0],{k:prompts_keys[k] for k in nearest_keys}
        
    Class_nearest_keys, Class_scores, Class_info = Retrive(Query_vector, top_k=class_count,type_str='Class') 
    OP_nearest_keys, OP_scores, OP_info = Retrive(Query_vector, top_k=op_count,type_str='ObjectProperty') 
    DP_nearest_keys, DP_scores, DP_info = Retrive(Query_vector, top_k=dp_count,type_str='DataProperty')  

    return Class_nearest_keys, Class_scores, Class_info, \
           OP_nearest_keys, OP_scores, OP_info,\
           DP_nearest_keys, DP_scores, DP_info

# init_rag()


Query = "What are the components of a product?"
Query_vector = OllamaEmbedderQWEN([Query])


Class_nearest_keys, Class_scores, Class_info, \
           OP_nearest_keys, OP_scores, OP_info,\
           DP_nearest_keys, DP_scores, DP_info = RetriveComponents(Query_vector)

RAG_return = [[Class_nearest_keys, Class_scores, Class_info],
              [OP_nearest_keys, OP_scores, OP_info],
                [DP_nearest_keys, DP_scores, DP_info]]


print(Query)

Prompts_classes, Prompts_OP, Prompts_DP = np.load('EmbeddingSystem/Prompts_full_info.npy', allow_pickle=True)

for item in RAG_return:
    nearestClasses, nearestClassesScores, info = item
    print('---')
    print(nearestClasses)
    print('---')
    print(nearestClassesScores)
    for k,v in info.items():
        print(k,v)
        print('---')
        print()


