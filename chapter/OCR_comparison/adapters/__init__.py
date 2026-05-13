from .chandra_adapter import run_chandra
from .deepseek_ocr_adapter import run_deepseek_ocr
from .dots_ocr_adapter import run_dots_ocr
from .paddleocr_adapter import run_paddleocr


ADAPTERS = {
    "paddleocr": run_paddleocr,
    "chandra": run_chandra,
    "dots_ocr": run_dots_ocr,
    "deepseek_ocr": run_deepseek_ocr,
}

