"""Base OCR engine interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TextBlock:
    """Represents a block of recognized text with position information."""
    
    block_id: int
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    page_number: int
    level: int = 2  # Hierarchy level (page/block/paragraph/line/word)
    paragraph_id: Optional[int] = None
    line_id: Optional[int] = None
    word_id: Optional[int] = None


@dataclass
class OCRResult:
    """Complete OCR result for a page."""
    
    page_number: int
    blocks: list[TextBlock] = field(default_factory=list)
    page_width: int = 0
    page_height: int = 0
    full_text: str = ""
    
    def get_blocks_by_level(self, level: int) -> list[TextBlock]:
        """Get all blocks at a specific hierarchy level."""
        return [block for block in self.blocks if block.level == level]
    
    def get_block_text(self, block_id: int) -> str:
        """Get concatenated text for a specific block."""
        block_texts = [
            block.text
            for block in self.blocks
            if block.block_id == block_id
        ]
        return " ".join(block_texts)
    
    def calculate_average_confidence(self) -> float:
        """Calculate average confidence across all blocks."""
        if not self.blocks:
            return 0.0
        confidences = [block.confidence for block in self.blocks if block.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    def __init__(self, language: str = "fra", confidence_threshold: int = 80):
        """
        Initialize OCR engine.
        
        Args:
            language: Language code for OCR
            confidence_threshold: Minimum confidence for text acceptance
        """
        self.language = language
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def process_image(self, image_path: Path) -> OCRResult:
        """
        Process an image and extract text.
        
        Args:
            image_path: Path to image file
            
        Returns:
            OCR result with extracted text and metadata
            
        Raises:
            OCRError: If OCR processing fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the OCR engine is available on the system.
        
        Returns:
            True if engine is available, False otherwise
        """
        pass
    
    def validate_result(self, result: OCRResult) -> bool:
        """
        Validate OCR result quality.
        
        Args:
            result: OCR result to validate
            
        Returns:
            True if result meets quality threshold
        """
        avg_confidence = result.calculate_average_confidence()
        return avg_confidence >= self.confidence_threshold


class OCRError(Exception):
    """Base exception for OCR-related errors."""
    pass
