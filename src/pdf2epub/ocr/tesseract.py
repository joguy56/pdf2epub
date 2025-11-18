"""Tesseract OCR implementation."""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import pytesseract
from pytesseract import Output

from pdf2epub.ocr.base import OCREngine, OCRError, OCRResult, TextBlock

logger = logging.getLogger(__name__)


class TesseractOCR(OCREngine):
    """Tesseract OCR engine implementation."""
    
    def __init__(
        self,
        language: str = "fra",
        confidence_threshold: int = 80,
        tesseract_dir: Optional[str] = None
    ):
        """
        Initialize Tesseract OCR.
        
        Args:
            language: Language code for OCR
            confidence_threshold: Minimum confidence threshold
            tesseract_dir: Path to Tesseract data directory
            
        Raises:
            OCRError: If Tesseract is not properly configured
        """
        super().__init__(language, confidence_threshold)
        
        # Set Tesseract data directory
        if tesseract_dir:
            os.environ["TESSDATA_PREFIX"] = tesseract_dir
        elif "TESSDATA_PREFIX" not in os.environ:
            raise OCRError(
                "Tesseract data directory not configured. "
                "Set TESSDATA_PREFIX environment variable or pass tesseract_dir parameter."
            )
        
        # Verify Tesseract is available
        if not self.is_available():
            raise OCRError("Tesseract OCR is not available on this system")
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def process_image(self, image_path: Path) -> OCRResult:
        """
        Process image with Tesseract OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            OCR result with hierarchical text data
            
        Raises:
            OCRError: If processing fails
        """
        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise OCRError(f"Failed to load image: {image_path}")
            
            # Extract page number from filename
            page_number = self._extract_page_number(image_path)
            
            # Run Tesseract
            logger.debug(f"Running Tesseract OCR on {image_path}")
            results = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=Output.DICT
            )
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(results)
            
            # Get page dimensions
            page_width = df["width"].iloc[0] if len(df) > 0 else 0
            page_height = df["height"].iloc[0] if len(df) > 0 else 0
            
            # Parse results into TextBlocks
            blocks = self._parse_results(results, page_number)
            
            # Get full text
            full_text = " ".join([block.text for block in blocks if block.text.strip()])
            
            result = OCRResult(
                page_number=page_number,
                blocks=blocks,
                page_width=page_width,
                page_height=page_height,
                full_text=full_text
            )
            
            logger.info(
                f"Page {page_number}: Extracted {len(blocks)} blocks, "
                f"avg confidence: {result.calculate_average_confidence():.1f}"
            )
            
            return result
            
        except Exception as e:
            raise OCRError(f"Tesseract processing failed for {image_path}: {e}") from e
    
    def _extract_page_number(self, image_path: Path) -> int:
        """Extract page number from filename."""
        try:
            # Expect format: page_001.jpg or page_001_thresh.png
            name = image_path.stem
            if "_thresh" in name:
                name = name.replace("_thresh", "")
            # Extract number
            parts = name.split("_")
            return int(parts[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract page number from {image_path}, using 0")
            return 0
    
    def _parse_results(self, results: dict, page_number: int) -> list[TextBlock]:
        """
        Parse Tesseract results dictionary into TextBlock objects.
        
        Args:
            results: Tesseract results dictionary
            page_number: Page number for these results
            
        Returns:
            List of TextBlock objects
        """
        blocks = []
        
        for i in range(len(results["text"])):
            text = results["text"][i]
            conf = float(results["conf"][i])
            
            # Skip empty text or invalid confidence
            if not text.strip() or conf < 0:
                continue
            
            block = TextBlock(
                block_id=results["block_num"][i],
                text=text,
                confidence=conf,
                x=results["left"][i],
                y=results["top"][i],
                width=results["width"][i],
                height=results["height"][i],
                page_number=page_number,
                level=results["level"][i],
                paragraph_id=results["par_num"][i],
                line_id=results["line_num"][i],
                word_id=results["word_num"][i]
            )
            
            blocks.append(block)
        
        return blocks
    
    def get_detailed_data(self, image_path: Path) -> pd.DataFrame:
        """
        Get detailed OCR data as pandas DataFrame.
        
        Useful for debugging and analysis.
        
        Args:
            image_path: Path to image file
            
        Returns:
            DataFrame with detailed OCR results
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        results = pytesseract.image_to_data(
            image,
            lang=self.language,
            output_type=Output.DICT
        )
        return pd.DataFrame(results)
