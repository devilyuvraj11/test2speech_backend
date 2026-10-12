import time
from deep_translator import GoogleTranslator


def translate_text(text: str, source: str = "auto", target: str = "hi") -> str:
    if not text or not text.strip():
        return ""
    return GoogleTranslator(source=source, target=target).translate(text)


def translate_batch(texts: list[str], source: str = "auto", target: str = "hi") -> list[str]:
    if not texts:
        return []

    translator = GoogleTranslator(source=source, target=target)
    results = []

    for text in texts:
        for attempt in range(4):
            try:
                results.append(translator.translate(text))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))  # backoff on rate limit
        time.sleep(0.25)  # ~4 requests/sec, under the limit

    return results