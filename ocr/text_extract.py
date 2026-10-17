import os
import threading

_ocr = None
_lock = threading.Lock()


def ocr_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("paddleocr") is not None


def _get_ocr():
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                from paddleocr import PaddleOCR
                _ocr = PaddleOCR(
                    enable_mkldnn=False,
                    use_textline_orientation=True,
                    lang="en",
                )
    return _ocr


def warm_ocr():
    if ocr_available():
        try:
            _get_ocr()
        except Exception:
            pass


def ocr_model(img_path):
    ocr = _get_ocr()
    result = ocr.predict(img_path)

    pages = []
    for page in result:
        data = page.json
        if callable(data):
            data = data()
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        pages.append("\n".join(data.get("rec_texts", [])))  
    return "\n\n".join(pages)


def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages).strip()
        if len(text) > 40:
            return text
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(path)
            out = []
            for i in range(min(len(pdf), 6)):
                img_path = f"{path}.p{i}.png"
                pdf[i].render(scale=2).to_pil().save(img_path)
                out.append(ocr_model(img_path))
                os.remove(img_path)
            return "\n\n".join(out)
        except ImportError:
            return text
    return ocr_model(path)
