import os
import base64
import math
from typing import Optional, List

import streamlit as st
from openai import OpenAI
from filetype import filetype
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# LLM constants
TRANSCRIBE_ERROR = "No text found"
TRANSLATION_ERROR = "Invalid translation"

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
gpt_client = OpenAI(api_key=OPENAI_API_KEY)


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


def transcribe_image(
    image: bytes, content_type: str, model: Optional[str] = "gpt-4o-mini"
) -> str:
    """
    Transcribe text from an image using OpenAI's GPT-4o vision model.

    Args:
        image (bytes): Image data
        content_type (str): MIME type of the image

    Returns:
        str: Transcribed text or error message
    """
    # Encode image to base64
    base64_image = base64.b64encode(image).decode("utf-8")

    # Create chat completion request
    chat_response = gpt_client.chat.completions.create(
        model=model,
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


def translate_text(
    text: str, source_lang: Optional[str] = None, model: Optional[str] = "gpt-4o-mini"
) -> str:
    """
    Translate text to English using OpenAI's GPT model.

    Args:
        text (str): Text to translate
        source_lang (str, optional): Source language (if known)

    Returns:
        str: Translated text
    """
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
                model=model, messages=messages
            )

            # Add translated batch
            translated_batches.append(
                translation_response.choices[0].message.content.strip()
            )

        # Combine translated batches
        return " ".join(translated_batches)

    except Exception as e:
        st.error(f"Error translating text: {e}")
        return TRANSLATION_ERROR


def main():
    # Add this line to start with the sidebar collapsed
    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("Language Transcriber")

    # Sidebar for source language selection
    st.sidebar.header("Translation Settings")
    source_lang = st.sidebar.text_input(
        "Source Language (Optional)", help="Language to be translated"
    )

    source_model = st.sidebar.selectbox(
        "Model Usage (Optional)", ("gpt-4o-mini", "gpt-4o"), help="GPT model to be used"
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image to transcribe", type=["png", "jpg", "jpeg", "gif", "bmp"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Detect file type
        file_details = filetype.guess(uploaded_file.getvalue())

        if file_details and file_details.mime.startswith("image/"):
            # Transcription
            with st.spinner("Transcribing text from image..."):
                transcription = transcribe_image(
                    uploaded_file.getvalue(), file_details.mime, source_model
                )

            # Display transcription
            st.subheader("Transcribed Text")
            st.text_area("Transcription", value=transcription, height=500)

            # Translation
            if transcription != TRANSCRIBE_ERROR:
                with st.spinner("Translating text..."):
                    translation = translate_text(
                        transcription, source_lang, source_model
                    )

                # Display translation
                st.subheader("Translated Text")
                st.text_area("Translation", value=translation, height=500)
            else:
                st.warning("No text could be transcribed from the image.")
        else:
            st.error("Invalid image file. Please upload a valid image.")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not OPENAI_API_KEY:
        st.error(
            "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
        )
    else:
        main()
