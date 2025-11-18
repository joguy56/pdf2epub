"""OCR module for pdf2epub."""

from pdf2epub.ocr.base import OCREngine, OCRResult, TextBlock
from pdf2epub.ocr.tesseract import TesseractOCR
from pdf2epub.ocr.easyocr import EasyOCREngine

__all__ = ["OCREngine", "OCRResult", "TextBlock", "TesseractOCR", "EasyOCREngine"]
