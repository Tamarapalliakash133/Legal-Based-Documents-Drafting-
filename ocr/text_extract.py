from paddleocr import PaddleOCR

ocr = PaddleOCR(
    enable_mkldnn=False,
    use_textline_orientation = True,
    lang = "en"
)

def ocr_model(img_path):
    result = ocr.predict(img_path)

    for page in result:
        data = page.json
        if callable(data):
            data = data()

        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        final_text = "\n".join(data.get("rec_texts", []))
        return final_text
