import os
from openai import AzureOpenAI

api_key_path = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt'
ENDPOINT_URL = 'https://text-db.openai.azure.com/'
api_version_name = '2025-01-01-preview'

def gpt_4o_azure(prompt, 
                key_path=api_key_path,
                max_tokens=800,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0):
    """
    Get response from Azure OpenAI API.
    
    Args:
        prompt (str or list(dict)): The text prompt to send to the model
        key_path (str): Path to the API key file
        max_tokens (int): Maximum tokens for response
        temperature (float): Response randomness (0-1)
        
    Returns:
        str: The response content from the model
    """
    # Read API key
    with open(key_path, 'r') as f:
        api_key = f.read().strip()
    
    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=os.getenv("ENDPOINT_URL", ENDPOINT_URL),
        api_key=api_key,
        api_version=api_version_name,
    )
    
    # Generate response
    if isinstance(prompt, list):
        completion = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False
        )
    else:
        completion = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False
        )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # test the function
    input = "What is the capital of France?"
    print(gpt_4o_azure(input))