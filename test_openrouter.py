from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the OpenRouter API
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("OPENROUTER_API_KEY not found in environment")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# List available models first
try:
    print("Testing OpenRouter API with a free model...")

    # Try with a free model from OpenRouter
    model_name = 'google/gemma-2-9b-it:free'  # Free model option

    # Simple test query
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        temperature=0.3
    )

    print(f"\nOpenRouter API is working correctly with model {model_name}!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"Error with OpenRouter API: {str(e)}")

    try:
        print("\nTrying with another model...")
        # Try an alternative model
        model_name = 'mistralai/mistral-7b-instruct:free'
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            temperature=0.3
        )
        print(f"OpenRouter API is working correctly with model {model_name}!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e2:
        print(f"Error with alternative model: {str(e2)}")