import os
import base64
from typing import Optional, Dict, Union

from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Global error messages
TRANSLATION_ERROR = "Invalid translation"
TRANSCRIBE_ERROR = "No text found"

# Default model configurations
# DEFAULT_GPT_MODEL = "gpt-4o-mini"
DEFAULT_GPT_MODEL = "gpt-4o"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"


def initialize_clients(
    openai_api_key: str = os.environ.get("OPENAI_API_KEY"),
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY"),
) -> tuple:
    """
    Initialize OpenAI and Anthropic API clients.

    Args:
        openai_api_key (str, optional): OpenAI API key
        anthropic_api_key (str, optional): Anthropic API key

    Returns:
        tuple: OpenAI and Anthropic clients
    """
    try:
        gpt_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        claude_client = (
            Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        )
        return gpt_client, claude_client
    except Exception as e:
        print(f"Client initialization error: {e}")
        return None, None


def generate_response(
    model: str = DEFAULT_GPT_MODEL,
    system: str = "",
    prompt: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.5,
    client_type: str = "GPT",
) -> Optional[Dict[str, Union[str, int]]]:
    """
    Generate response from either GPT or Claude model.

    Args:
        model (str): Model to use
        system (str): System prompt
        prompt (str): User prompt
        max_tokens (int): Maximum token count
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling threshold
        client_type (str): 'GPT' or 'Claude'

    Returns:
        dict: Response details or None
    """
    gpt_client, claude_client = initialize_clients()
    client = gpt_client if client_type == "GPT" else claude_client

    if not client:
        print(f"Unable to get {client_type} client")
        return None

    try:
        if client_type == "GPT":
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": prompt.strip()},
                ],
            )
            return {
                "response": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        # Claude response
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": system.strip()},
            ],
        )
        return {
            "response": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    except Exception as e:
        print(f"Response generation error for {model}: {e}")
        return None


def gpt_response(
    model: str = DEFAULT_GPT_MODEL, system: str = "", prompt: str = "", **kwargs
) -> Optional[Dict[str, Union[str, int]]]:
    """Wrapper for GPT response generation."""
    return generate_response(
        model=model, system=system, prompt=prompt, client_type="GPT", **kwargs
    )


def claude_response(
    model: str = DEFAULT_CLAUDE_MODEL, system: str = "", prompt: str = "", **kwargs
) -> Optional[Dict[str, Union[str, int]]]:
    """Wrapper for Claude response generation."""
    return generate_response(
        model=model, system=system, prompt=prompt, client_type="Claude", **kwargs
    )


def transcribe_image(
    image: bytes,
    content_type: str,
    client_type: str = "GPT",
    model: Optional[str] = None,
) -> str:
    """
    Transcribe text from an image using specified AI model.

    Args:
        image (bytes): Image data
        content_type (str): MIME type of the image
        client_type (str): 'GPT' or 'Claude'
        model (str, optional): Specific model to use

    Returns:
        str: Transcribed text
    """
    gpt_client, claude_client = initialize_clients()
    client = gpt_client if client_type == "GPT" else claude_client

    if not client:
        return f"Unable to initialize {client_type} client"

    # Encode image to base64
    base64_image = base64.b64encode(image).decode("utf-8")

    # Default models
    default_models = {"GPT": DEFAULT_GPT_MODEL, "Claude": DEFAULT_CLAUDE_MODEL}
    model = model or default_models[client_type]

    try:
        if client_type == "GPT":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional Japanese transcriber. Transcribe only clear, legible text exactly as seen.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Transcribe the text from this image. If no text, respond with '{TRANSCRIBE_ERROR}'.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()

        # Claude transcription
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "assistant",
                    "content": "You are a professional Japanese transcriber. Transcribe only clear, legible text exactly as seen.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Transcribe the clear, legible text from this image. If no relevant text, respond with '{TRANSCRIBE_ERROR}'.",
                        },
                    ],
                },
            ],
        )
        return response.content[0].text.strip()

    except Exception as e:
        return f"Transcription error: {str(e)}"


def gpt_transcribe(
    image: bytes, content_type: str, model: Optional[str] = DEFAULT_GPT_MODEL
) -> str:
    """Wrapper for GPT image transcription."""
    return transcribe_image(image, content_type, client_type="GPT", model=model)


def claude_transcribe(
    image: bytes, content_type: str, model: Optional[str] = DEFAULT_CLAUDE_MODEL
) -> str:
    """Wrapper for Claude image transcription."""
    return transcribe_image(image, content_type, client_type="Claude", model=model)
