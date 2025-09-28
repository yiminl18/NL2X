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
                presence_penalty=0,
                return_usage=False,
                response_format=None):
    """
    Get response from Azure OpenAI API.

    Args:
        prompt (str or list(dict)): The text prompt to send to the model
        key_path (str): Path to the API key file
        max_tokens (int): Maximum tokens for response
        temperature (float): Response randomness (0-1)
        return_usage (bool): If True, return usage information along with content
        response_format (dict): Optional response format for structured output

    Returns:
        str or tuple: The response content from the model, or (content, usage) if return_usage=True
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
    
    # Prepare completion parameters
    completion_params = {
        "model": os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": False
    }

    # Add response_format if provided
    if response_format:
        completion_params["response_format"] = response_format

    # Generate response
    if isinstance(prompt, list):
        completion_params["messages"] = prompt
    else:
        completion_params["messages"] = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(**completion_params)
    content = completion.choices[0].message.content
    
    if return_usage:
        return content, completion.usage
    else:
        return content

if __name__ == "__main__":
    # test the function
    input = "What is the capital of France?"
    print(gpt_4o_azure(input))