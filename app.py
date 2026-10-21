import streamlit as st
import cv2
import numpy as np
import pytesseract

from PIL import Image
from deep_translator import GoogleTranslator
from io import BytesIO
import time

# ==================================================
# TESSERACT CONFIGURATION
# ==================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PalmScript AI",
    page_icon="🌴",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #101820;
    color: #F5F5F5;
}

h1, h2, h3 {
    color: #E8B86D;
}

.stButton > button {
    background-color: #D4A056;
    color: #101820;
    border-radius: 8px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #F0C987;
    color: #101820;
}

[data-testid="stFileUploader"] {
    background-color: #1B2833;
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.title("🌴 PalmScript AI")

st.markdown(
    "### AI-Powered Palm-Leaf Manuscript Recognition"
)

st.write(
    "Upload a manuscript, enhance its image, "
    "extract text, and translate the result into English."
)

st.divider()

# ==================================================
# LANGUAGE OPTIONS
# ==================================================

languages = {
    "Tamil": "tam",
    "Hindi": "hin",
    "Telugu": "tel",
    "Kannada": "kan",
    "Malayalam": "mal",
    "Bengali": "ben",
    "Gujarati": "guj",
    "Marathi": "mar",
    "Sanskrit": "san"
}

language = st.selectbox(
    "Select the manuscript language",
    list(languages.keys())
)

# ==================================================
# CHECK TESSERACT
# ==================================================

try:
    available_languages = set(
        pytesseract.get_languages(config="")
    )

    tesseract_ready = True

except Exception as error:
    available_languages = set()
    tesseract_ready = False

    st.error(
        f"Tesseract could not be initialized: {error}"
    )

if tesseract_ready:

    if languages[language] in available_languages:
        st.success(
            f"Tesseract is ready. "
            f"{language} language data is installed."
        )

    else:
        st.warning(
            f"The {language} OCR language model "
            "is not available."
        )

# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload a palm-leaf manuscript image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        # ------------------------------------------
        # ORIGINAL IMAGE
        # ------------------------------------------

        st.subheader("1. Original Manuscript")

        st.image(
            image,
            caption="Uploaded manuscript",
            use_container_width=True
        )

        # ------------------------------------------
        # IMAGE ENHANCEMENT
        # ------------------------------------------

        st.subheader("2. Image Enhancement")

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # Threshold image for OCR
        _, threshold_image = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                gray,
                caption="Grayscale Image",
                use_container_width=True
            )

        with col2:
            st.image(
                enhanced,
                caption="Enhanced Image",
                use_container_width=True
            )

        with st.expander("View thresholded image"):

            st.image(
                threshold_image,
                caption="Thresholded Image",
                use_container_width=True
            )

        # ------------------------------------------
        # DOWNLOAD ENHANCED IMAGE
        # ------------------------------------------

        image_buffer = BytesIO()

        Image.fromarray(enhanced).save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            label="Download Enhanced Image",
            data=image_buffer.getvalue(),
            file_name="enhanced_manuscript.png",
            mime="image/png"
        )

        st.divider()

        # ------------------------------------------
        # TEXT RECOGNITION
        # ------------------------------------------

        st.subheader("3. Text Recognition")

        if st.button("Recognize Manuscript Text"):

            if not tesseract_ready:

                st.error(
                    "Tesseract is unavailable. "
                    "Check its installation path."
                )

            elif languages[language] not in available_languages:

                st.error(
                    f"The {language} language model is missing. "
                    "Install its .traineddata file."
                )

            else:

                with st.spinner(
                    "Recognizing manuscript text..."
                ):

                    try:
                        lang_code = languages[language]

                        # OCR attempt 1
                        text1 = pytesseract.image_to_string(
                            enhanced,
                            lang=lang_code,
                            config="--oem 1 --psm 6"
                        )

                        # OCR attempt 2
                        text2 = pytesseract.image_to_string(
                            threshold_image,
                            lang=lang_code,
                            config="--oem 1 --psm 11"
                        )

                        # Select the longer result
                        extracted_text = max(
                            [text1.strip(), text2.strip()],
                            key=len
                        )

                        # Store OCR result
                        st.session_state[
                            "extracted_text"
                        ] = extracted_text

                        # Update text editor before rendering
                        st.session_state[
                            "recognized_text_editor"
                        ] = extracted_text

                        # Save selected language
                        st.session_state[
                            "selected_language"
                        ] = language

                        # Clear previous translation
                        st.session_state[
                            "translated_text"
                        ] = ""

                        st.session_state[
                            "translation_editor"
                        ] = ""

                        if extracted_text:

                            st.success(
                                "Text recognition completed. "
                                "Review the result for accuracy."
                            )

                            st.write("OCR output preview:")

                            st.code(extracted_text)

                        else:

                            st.warning(
                                "No text was detected. Try a clearer "
                                "image or crop the manuscript. "
                                "Historical scripts may require "
                                "a specialized OCR model."
                            )

                    except Exception as error:

                        st.error(
                            f"OCR failed: {error}"
                        )

        # ------------------------------------------
        # RECOGNIZED TEXT EDITOR
        # ------------------------------------------

        if "recognized_text_editor" not in st.session_state:

            st.session_state[
                "recognized_text_editor"
            ] = st.session_state.get(
                "extracted_text", ""
            )

        extracted_text = st.text_area(
            "Recognized Text (you can edit this)",
            height=200,
            key="recognized_text_editor",
            placeholder="Recognized manuscript text will appear here..."
        )

        # Save any edits made by the user
        st.session_state[
            "extracted_text"
        ] = extracted_text

        if extracted_text.strip():

            st.download_button(
                label="Download Recognized Text",
                data=extracted_text,
                file_name="recognized_text.txt",
                mime="text/plain"
            )

            st.divider()

            # --------------------------------------
            # ENGLISH TRANSLATION
            # --------------------------------------

            st.subheader("4. English Translation")

            st.caption(
                "Translation requires an internet connection. "
                "Incorrect OCR results may produce incorrect translations."
            )

            if st.button("Translate to English"):

                if not extracted_text.strip():

                    st.warning(
                        "Please enter text before translating."
                    )

                else:

                    try:
                        with st.spinner(
                            "Translating text..."
                        ):

                            # Translate the entire text in one request
                            # for ordinary short manuscripts.
                            if len(extracted_text) <= 4500:

                                translated_text = GoogleTranslator(
                                    source="auto",
                                    target="en"
                                ).translate(extracted_text)

                            else:
                                # Split long text into chunks.
                                # A short pause reduces rapid requests,
                                # but does not bypass provider rate limits.

                                words = extracted_text.split()
                                chunks = []
                                current_chunk = ""

                                for word in words:

                                    candidate = (
                                        current_chunk + " " + word
                                    ).strip()

                                    if len(candidate) > 4000:

                                        if current_chunk:
                                            chunks.append(current_chunk)

                                        current_chunk = word

                                    else:
                                        current_chunk = candidate

                                if current_chunk:
                                    chunks.append(current_chunk)

                                translated_chunks = []

                                for index, chunk in enumerate(chunks):

                                    if index > 0:
                                        time.sleep(1.5)

                                    result = GoogleTranslator(
                                        source="auto",
                                        target="en"
                                    ).translate(chunk)

                                    if result:
                                        translated_chunks.append(result)

                                translated_text = "\n\n".join(
                                    translated_chunks
                                )

                            translated_text = translated_text or ""

                            # Save translated text
                            st.session_state[
                                "translated_text"
                            ] = translated_text

                            # Update translation widget state
                            st.session_state[
                                "translation_editor"
                            ] = translated_text

                            if translated_text:

                                st.success(
                                    "Translation completed!"
                                )

                            else:

                                st.warning(
                                    "The translation service returned no text."
                                )

                    except Exception as error:

                        error_message = str(error)

                        if (
                            "too many requests" in error_message.lower()
                            or "429" in error_message
                            or "server error" in error_message.lower()
                        ):

                            st.error(
                                "Google Translate is temporarily rate-limiting "
                                "requests. Wait before trying again. "
                                "Repeated attempts may prolong the restriction."
                            )

                        else:

                            st.error(
                                "Translation failed. Check your internet "
                                f"connection and try again. Details: {error}"
                            )

            # --------------------------------------
            # TRANSLATION EDITOR
            # --------------------------------------

            if "translation_editor" not in st.session_state:

                st.session_state[
                    "translation_editor"
                ] = st.session_state.get(
                    "translated_text", ""
                )

            translated_text = st.text_area(
                "English Translation",
                height=200,
                key="translation_editor",
                placeholder="English translation will appear here..."
            )

            # Preserve manual edits
            st.session_state[
                "translated_text"
            ] = translated_text

            if translated_text.strip():

                st.download_button(
                    label="Download English Translation",
                    data=translated_text,
                    file_name="english_translation.txt",
                    mime="text/plain"
                )

    except Exception as error:

        st.error(
            f"Unable to process this image: {error}"
        )

else:

    st.info(
        "Upload a manuscript image to begin."
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "PalmScript AI | Multilingual Manuscript Recognition "
    "and Translation | AI & Data Science Project"
)