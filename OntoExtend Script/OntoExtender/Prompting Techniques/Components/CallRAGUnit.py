import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from RAG.RAG2_Winner1.RAG import RAG
import importlib.util

spec = importlib.util.find_spec("RAG.RAG2_Winner1.RAG")

class RAGCaller:
    def CallRAG(self, CQ, topK=20, searchK=10000):
        
        return RAG(CQ)


test = RAGCaller()
print(test.CallRAG("What are the components of a product?"))