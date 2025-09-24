"""
So, we want only one function here, regardless of the model, API type, or provider.
The function will take in: API (values: OpenAI, LiU Azure,...), API inputs: python dict, LLM, Prompt, some parameters with default values, and return the response.
To make API key save, we import it from another file, not here.
"""
from openai import AzureOpenAI
import re,ast, os
import openai
from dotenv import load_dotenv
import os
load_dotenv()


def call_LLM_API(API, API_inputs, LLM, Prompt):
    if API == 'OpenAI':
        #anna sofia
        pass
    elif API == 'LiU_Azure':
        #javad
        api_key     = os.getenv("API_KEY")
        api_version = os.getenv("api_version")
        endpoint    = os.getenv('endpoint')
        prompt = Prompt
        model = LLM if LLM else 'GPT-5'

        client = AzureOpenAI(
        azure_endpoint = endpoint,
        api_version=api_version,
        api_key=api_key 
        )

        response = client.chat.completions.create(
        messages = [{"role":"system","content":prompt_text}],
        stop=None)
        return response.choices[0].message.content

    return 'not working'

call_LLM_API('LiU_Azure', {}, LLM='GPT-5', Prompt='test')