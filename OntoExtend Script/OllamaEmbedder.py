def OllamaEmbedderQWEN(text = []):
    import time
    import ollama
    response = ollama.embed(model="qwen3-embedding:4b", input=text)#qwen3-embedding:4b
    embedings = response['embeddings']
    return embedings


