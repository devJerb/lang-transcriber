import math

from filetype import filetype
from typing import Optional, List
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Body

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(docs_url="/")

# Import constants from model module
from model import (
    generate_response,
    transcribe_image,
    TRANSLATION_ERROR,
    DEFAULT_GPT_MODEL,
    DEFAULT_CLAUDE_MODEL,
)


def split_text_batches(text: str, max_tokens: int = 1000) -> List[str]:
    """
    Split text into batches respecting approximate token limit.

    Args:
        text (str): Input text to be split
        max_tokens (int): Maximum tokens per batch

    Returns:
        List[str]: List of text batches
    """
    # Rough token estimate: 1 token ≈ 4 characters
    if len(text) / 4 <= max_tokens:
        return [text]

    # Calculate number of batches
    num_batches = math.ceil(len(text) / (max_tokens * 4))
    batch_size = len(text) // num_batches

    return [text[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]


async def transcribe_async(image: bytes, content_type: str) -> str:
    """
    Async wrapper for image transcription.

    Args:
        image (bytes): Image data
        content_type (str): MIME type of the image

    Returns:
        str: Transcribed text or error message
    """
    return transcribe_image(image, content_type)


async def translate_async(text: str, model: str = "GPT") -> str:
    """
    Async wrapper for text translation.

    Args:
        text (str): Text to translate
        model (str, optional): Translation model to use

    Returns:
        str: Translated text
    """
    try:
        # Determine translation function's specific system prompt
        system_prompt = "You are a professional translator. Translate the given text to English accurately while preserving the original meaning and tone."

        # Split and translate batches
        translated_batches = []
        for batch in split_text_batches(text):
            # Prepare translation prompt
            prompt = f"Translate the following text to English:\n\n{batch}"

            # Use the appropriate model
            model_to_use = DEFAULT_GPT_MODEL if model == "GPT" else DEFAULT_CLAUDE_MODEL
            response = generate_response(
                model=model_to_use,
                system=system_prompt,
                prompt=prompt,
                client_type=model,
            )

            if response and "response" in response:
                translated_batches.append(response["response"].strip())
            else:
                return TRANSLATION_ERROR

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
        transcription = await transcribe_async(contents, detected_type.mime)

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
    text: str = Body(...),
    model: Optional[str] = Body("GPT"),
):
    """
    Endpoint to translate text to English.

    Args:
        text (str): Text to translate
        model (str, optional): Translation model to use (GPT or Claude)

    Returns:
        JSONResponse with translation details
    """
    try:
        # Validate input
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Validate model
        if model not in ["GPT", "Claude"]:
            raise HTTPException(
                status_code=400, detail="Invalid model. Use 'GPT' or 'Claude'."
            )

        # Translate text
        translated_text = await translate_async(text, model)

        # Return response
        return JSONResponse(
            content={
                "translation": {
                    "original_text": text,
                    "model_used": model,
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
