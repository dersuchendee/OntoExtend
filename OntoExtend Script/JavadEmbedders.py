def OllamaEmbedderQWEN(text = [],llm = 'qwen3-embedding:4b'):
    import time
    import ollama
    import numpy as np
    response = ollama.embed(model=llm, input=text)#qwen3-embedding:4b
    embedings = response['embeddings']
    return np.array(embedings)


def LiUAzureEmbedder(prompt_text,llm= 'text-embedding-3-small'):
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())


    import re,ast, os
    import numpy as np
    import openai
    from openai import AzureOpenAI

    endpoint = "https://liu-ida-kmacs-gpt4-oai.openai.azure.com/"
    api_key = os.environ.get('APIKEY_Embedder')

    client = AzureOpenAI(
    azure_endpoint = endpoint,
    api_version="2025-03-01-preview",
    api_key=api_key # Add the api_key parameter here
    )
    def embedder(prompt_text):

        response = client.embeddings.create(
        model=  llm,
        input = prompt_text
        )
        return response.data#[0].message.content # Access the content attribute

    res = embedder(prompt_text)
    return np.array([res[i].embedding for i in range(len(res)) ])



if __name__ == "__main__":
    print(OllamaEmbedderQWEN(['Hi this is a test']).shape)
    print(LiUAzureEmbedder(['Hi this is a test'],llm= 'text-embedding-3-small').shape)
