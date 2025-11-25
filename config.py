import os 
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# API key for accessing the language model service
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")  


