import os
import json
import requests

def llm_commentator(decision, shap_values):
    """
    Call Gemini's text generation API to get an explanation comment
    Parameters:
    decision (str): The model's decision ("Approved" or "Denied")
    shap_values (dict/str): SHAP values information (could be a dict or formatted string)

    Returns:
    str: Generated comment explaining the decision.
    """
    # Construct a prompt which includes the decision and a summary of the SHAP values.
    # Customize the prompt style as needed.
    prompt = (
        f"Explain in simple terms why the loan was {decision.lower()} "
        f"given the following SHAP values: {shap_values}. "
        f"Focus on the most influential factors affecting the decision."
    )

    # Gemini API details: ensure you update api_url, headers, and payload based on the documentation.
    api_url = "https://api.gemini.ai/v1/complete"  # Replace with the correct endpoint

    # Retrieve your API key from environment variables for security.
    # api_key = os.getenv("GEMINI_API_KEY")
    api_key = 'AIzaSyCz0wrw5NWmdTSfMjOrHk2jZUS8ybN-3uo'

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "max_tokens": 100,       # adjust max tokens as needed
        "temperature": 0.7,      # adjust temperature for creativity vs. consistency
        "top_p": 0.9             # adjust top_p for sampling
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        # Adjust key names based on the actual response format from Gemini API.
        return response.json().get("completion", "No explanation provided.")
    else:
        return f"Error generating explanation (status code {response.status_code})."

if __name__ == '__main__':
    llm_commentator()