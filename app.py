import math
import streamlit as st
from typing import List

from model import (
    gpt_transcribe,
    gpt_response,
    claude_transcribe,
    claude_response,
    TRANSCRIBE_ERROR,
    TRANSLATION_ERROR,
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


def translate_text(text: str, model: str = "GPT") -> str:
    """
    Translate text to English using specified model.

    Args:
        text (str): Text to translate
        model (str, optional): Translation model to use

    Returns:
        str: Translated text
    """
    try:
        # Determine translation function based on model
        translation_func = gpt_response if model == "GPT" else claude_response

        # Split and translate batches
        translated_batches = [
            translation_func(
                system="You are a professional Japanese translator. Translate the text to English accurately.",
                prompt=f"Translate the following text to English:\n\n{batch}",
            )["response"]
            for batch in split_text_batches(text)
        ]

        return " ".join(translated_batches).strip()

    except Exception as e:
        st.error(f"Translation error: {e}")
        return TRANSLATION_ERROR


def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("Japanese Transcriber")

    # Translation settings
    with st.sidebar:
        st.header("Translation Settings")
        source_model = st.selectbox(
            "Translation Model", ("GPT", "Claude"), help="Choose translation model"
        )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Image", type=["png", "jpg", "jpeg", "gif", "bmp"]
    )

    if not uploaded_file:
        return

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Transcribe text
    with st.spinner("Transcribing text..."):
        transcription_func = (
            claude_transcribe if source_model == "Claude" else gpt_transcribe
        )
        transcription = transcription_func(uploaded_file.getvalue(), uploaded_file.type)

    # Handle transcription
    if transcription == TRANSCRIBE_ERROR:
        st.warning("No text could be transcribed from the image.")
        return

    # Display transcribed text
    st.subheader("Transcribed Text")
    st.text_area("Transcription", value=transcription, height=300)

    # Translate text
    with st.spinner("Translating text..."):
        translation = translate_text(transcription, source_model)

    # Display translation
    st.subheader("Translated Text")
    st.text_area("Translation", value=translation, height=300)


if __name__ == "__main__":
    main()
