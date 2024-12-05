import os
import math
import base64

from openai import OpenAI
from filetype import filetype
from typing import Optional, List
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Body

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(docs_url="/")

# LLM constants
TRANSCRIBE_ERROR = "No text found"
TRANSLATION_ERROR = "Invalid translation"
MODEL = "gpt-4o-mini"

# ✋🏻🛑⛔️
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


async def transcribe_image(image: bytes, content_type: str) -> str:
    """
    Transcribe text from an image using OpenAI's GPT-4o vision model.

    Args:
        image (bytes): Image data
        content_type (str): MIME type of the image

    Returns:
        str: Transcribed text or error message
    """
    gpt_client = OpenAI(api_key=OPENAI_API_KEY)

    # Encode image to base64
    base64_image = base64.b64encode(image).decode("utf-8")

    # Create chat completion request
    chat_response = gpt_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": 'You are an AI transcribing text from images. Guidelines: 1. Transcribe clear, legible text. 2. Respect original formatting. 3. If no clear text is visible or if the content is irrelevant, respond with "No recognizable text content". 4. Transcribe exactly as seen.',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Transcribe the clear, legible text from this image. If no relevant text, respond with '{TRANSCRIBE_ERROR}'.",
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

    return chat_response.choices[0].message.content.strip()


def split_text_into_batches(text: str, max_tokens: int = 1000) -> List[str]:
    """
    Split long text into batches respecting approximate token limit.

    Args:
        text (str): Input text to be split
        max_tokens (int): Maximum tokens per batch

    Returns:
        List[str]: List of text batches
    """
    # Rough estimate: 1 token is approximately 4 characters
    tokens_estimate = len(text) // 4

    # If text is short enough, return as single batch
    if tokens_estimate <= max_tokens:
        return [text]

    # Calculate number of batches
    num_batches = math.ceil(tokens_estimate / max_tokens)

    # Split text into roughly equal parts
    batch_size = len(text) // num_batches
    batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < num_batches - 1 else len(text)
        batches.append(text[start:end])

    return batches


def translate_text(text: str, source_lang: Optional[str] = None) -> str:
    """
    Translate text to English using OpenAI's GPT model.

    Args:
        text (str): Text to translate
        source_lang (str, optional): Source language (if known)

    Returns:
        str: Translated text
    """
    gpt_client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        # Split text into batches if too long
        text_batches = split_text_into_batches(text)
        translated_batches = []

        for batch in text_batches:
            # Prepare translation prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given text to English accurately while preserving the original meaning and tone.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following text to English{' from ' + source_lang if source_lang else ''}:\n\n{batch}",
                },
            ]

            # Create translation request
            translation_response = gpt_client.chat.completions.create(
                model=MODEL, messages=messages
            )

            # Add translated batch
            translated_batches.append(
                translation_response.choices[0].message.content.strip()
            )

        # Combine translated batches
        return " ".join(translated_batches)

    except Exception as e:
        print(f"Error translating text: {e}")
        return TRANSLATION_ERROR


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


@app.post("/transcribe")
async def upload_and_transcribe(file: UploadFile = File(...)):
    """
    Endpoint to upload and transcribe an image file.

    Args:
        file (UploadFile): Uploaded image file

    Returns:
        JSONResponse with transcription details
    """
    try:
        # Read the file contents
        contents = await file.read()

        # Detect file type
        detected_type = filetype.guess(contents)

        # Validate file type (optional: add more image mime types as needed)
        if not detected_type or not detected_type.mime.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Transcribe the image
        transcription = await transcribe_image(contents, detected_type.mime)

        # Return response
        return JSONResponse(
            content={
                "transcription": {
                    "filename": file.filename,
                    "file_type": detected_type.mime,
                    "text": transcription,
                }
            }
        )

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Handle other unexpected errors
        print(f"Unexpected error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")


@app.post("/translate")
async def translate_endpoint(
    text: str = Body(...), source_lang: Optional[str] = Body(None)
):
    """
    Endpoint to translate text to English.

    Args:
        text (str): Text to translate
        source_lang (str, optional): Source language

    Returns:
        JSONResponse with translation details
    """
    try:
        # Validate input
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Translate text
        translated_text = translate_text(text, source_lang)

        # Return response
        return JSONResponse(
            content={
                "translation": {
                    "original_text": text,
                    "source_language": source_lang,
                    "translated_text": translated_text,
                }
            }
        )

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Handle other unexpected errors
        print(f"Unexpected error translating text: {e}")
        raise HTTPException(status_code=500, detail="Error translating text")
