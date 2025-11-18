"""EasyOCR engine implementation."""

import logging
from pathlib import Path

import cv2
import easyocr

from pdf2epub.ocr.base import OCREngine, OCRError, OCRResult, TextBlock

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""
    
    def __init__(
        self,
        language: str = "fra",
        confidence_threshold: int = 80
    ):
        """
        Initialize EasyOCR engine.
        
        Args:
            language: Language code for OCR (fra, en, etc.)
            confidence_threshold: Minimum confidence threshold
        """
        super().__init__(language, confidence_threshold)
        
        # Map language codes
        lang_map = {
            "fra": "fr",
            "eng": "en"
        }
        self.easy_language = lang_map.get(language, "fr")
        
        # Initialize reader (lazy loading)
        self._reader: easyocr.Reader | None = None
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def _get_reader(self) -> easyocr.Reader:
        """Get or create EasyOCR reader instance."""
        if self._reader is None:
            logger.info(f"Initializing EasyOCR reader for language: {self.easy_language}")
            self._reader = easyocr.Reader([self.easy_language, 'en'])
        return self._reader
    
    def process_image(self, image_path: Path) -> OCRResult:
        """
        Process image with EasyOCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            OCR result
            
        Raises:
            OCRError: If processing fails
        """
        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise OCRError(f"Failed to load image: {image_path}")
            
            # Extract page number
            page_number = self._extract_page_number(image_path)
            
            # Get page dimensions
            page_height, page_width = image.shape[:2]
            
            # Run EasyOCR
            logger.debug(f"Running EasyOCR on {image_path}")
            reader = self._get_reader()
            
            results = reader.readtext(
                image,
                width_ths=0.7,
                ycenter_ths=0.5,
                height_ths=0.7,
                paragraph=True,
                detail=1  # Return coordinates and confidence
            )
            
            # Parse results
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
            raise OCRError(f"EasyOCR processing failed for {image_path}: {e}") from e
    
    def _extract_page_number(self, image_path: Path) -> int:
        """Extract page number from filename."""
        try:
            name = image_path.stem
            if "_thresh" in name:
                name = name.replace("_thresh", "")
            parts = name.split("_")
            return int(parts[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract page number from {image_path}, using 0")
            return 0
    
    def _parse_results(self, results: list, page_number: int) -> list[TextBlock]:
        """
        Parse EasyOCR results into TextBlock objects.
        
        EasyOCR returns: (bbox, text, confidence)
        where bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        
        Args:
            results: EasyOCR results list
            page_number: Page number
            
        Returns:
            List of TextBlock objects
        """
        blocks = []
        
        for block_id, (bbox, text, confidence) in enumerate(results):
            # Calculate bounding box
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            
            x = int(min(xs))
            y = int(min(ys))
            width = int(max(xs) - min(xs))
            height = int(max(ys) - min(ys))
            
            # Convert confidence to percentage
            conf_pct = confidence * 100
            
            block = TextBlock(
                block_id=block_id,
                text=text,
                confidence=conf_pct,
                x=x,
                y=y,
                width=width,
                height=height,
                page_number=page_number,
                level=2  # EasyOCR works at block level
            )
            
            blocks.append(block)
        
        return blocks
