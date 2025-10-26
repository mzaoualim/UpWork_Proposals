from dotenv import load_dotenv
import os

# Load environment variables from .env file for local testing
# NOTE: The .env file MUST be excluded from your GitHub repository via .gitignore!
load_dotenv()

# --- Connection Details ---

# MongoDB URI: Loaded from environment variables/Streamlit secrets for security.
# This variable must be defined in your .env file or Streamlit Cloud secrets.
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://PLACEHOLDER_USER:PLACEHOLDER_PASS@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority")

# Database and Collection Names
DATABASE_NAME = "chatbot_db"
CONVERSATION_COLLECTION = "chat_history"

# --- LLM Endpoint Configuration (For TinyLlama/Ollama/vLLM) ---
# This is the generic endpoint URL where you would deploy your self-hosted model.
CUSTOM_MODEL_ENDPOINT = os.getenv("CUSTOM_MODEL_ENDPOINT", "http://your-tinyllama-api-host:8000/v1/chat/completions")
API_AUTHENTICATION_TOKEN = os.getenv("API_AUTHENTICATION_TOKEN", "") 
LLM_MODEL_NAME = "TinyLlama-1.1B"

# --- TAILORED SYSTEM PROMPT for Plain Text Tool Use (Critical for small models) ---
SYSTEM_PROMPT = f"""
You are a MongoDB Agent powered by a fast, small-scale model ({LLM_MODEL_NAME}). Your goal is to answer data queries.

You have access to ONE tool:
TOOL: `query_mongodb_data(query_type: str, collection_name: str, query_parameters: str)`
DESCRIPTION: Executes a MongoDB operation to find or aggregate data.
- `query_type`: 'find', 'aggregate', or 'count'.
- `collection_name`: The collection name, e.g., 'users'.
- `query_parameters`: The exact JSON MQL query or aggregation pipeline.

**INSTRUCTION:**
If the user asks a data-related question, you MUST ONLY reply with a structured tool call.
Example: user asks 'How many users?':
Response: query_mongodb_data(query_type='count', collection_name='users', query_parameters='{}')

If the user asks a general question, respond conversationally.
"""
