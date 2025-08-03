from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text: str) -> str:
    """
    Detects the language of a given text.
    Returns 'fa' for Persian, 'en' for English, or the detected code.
    """
    try:
        lang_code = detect(text)
        return lang_code
    except:
        return 'en'

def translate_text(text: str, target_lang: str) -> str:
    """
    Translates text to the target language using Google Translate.
    """
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text