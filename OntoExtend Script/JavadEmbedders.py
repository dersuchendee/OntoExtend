def OllamaEmbedderQWEN(text = []):
    import time
    import ollama
    response = ollama.embed(model="qwen3-embedding:4b", input=text)#qwen3-embedding:4b
    embedings = response['embeddings']
    return embedings


import os
from openai import AzureOpenAI

endpoint = "https://liu-ida-kmacs-gpt4-oai.openai.azure.com/"
model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    endpoint=endpoint,
    credential=AzureKeyCredential("<API_KEY>")
)

response = client.embeddings.create(
    input=["first phrase","second phrase","third phrase"],
    model=deployment
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)