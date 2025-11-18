"""Chapter detection module."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from pdf2epub.config import ChapterDetectionConfig
from pdf2epub.ocr.base import OCRResult, TextBlock
from pdf2epub.utils import CHAPTER_MARKER, SECTION_MARKER, calculate_text_similarity

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """Represents a detected chapter."""
    
    title: str
    start_page: int
    start_block_id: int
    content: str = ""
    is_section: bool = False  # True for sections ($$$ marker)
    
    def get_marker(self) -> str:
        """Get the appropriate marker for this chapter."""
        return SECTION_MARKER if self.is_section else CHAPTER_MARKER
    
    def format_title(self) -> str:
        """Format title with markers."""
        marker = self.get_marker()
        return f"\n{marker} {self.title} {marker}\n"


class ChapterDetector:
    """Detects chapters in OCR results."""
    
    def __init__(
        self,
        config: ChapterDetectionConfig,
        book_title: str = "",
        current_chapter_title: str = ""
    ):
        """
        Initialize chapter detector.
        
        Args:
            config: Chapter detection configuration
            book_title: Optional book title for junk filtering
            current_chapter_title: Current chapter title for tracking
        """
        self.config = config
        self.book_title = book_title
        self.current_chapter_title = current_chapter_title
        self.chapter_counter = 1
    
    def detect_chapter(
        self,
        ocr_result: OCRResult,
        text_on_top: bool = False
    ) -> tuple[Optional[Chapter], bool, bool]:
        """
        Detect if a chapter starts on this page.
        
        Args:
            ocr_result: OCR result for the page
            text_on_top: Whether text has already appeared on this page
            
        Returns:
            Tuple of (detected chapter or None, updated text_on_top flag, should_update_text_on_top)
            - First bool: updated text_on_top after processing this page
            - Second bool: whether this block should update text_on_top (for non-junk blocks)
        """
        if not self.config.enabled:
            return None, text_on_top, False
        
        # Get page dimensions
        page_height = ocr_result.page_height
        page_width = ocr_result.page_width
        
        # Calculate chapter detection thresholds
        new_chapter_threshold = page_height * self.config.threshold_percentage / 100
        max_chapter_position = page_height * self.config.max_position_percentage / 100
        
        # Group blocks by block_id to get unique blocks
        seen_blocks = {}
        for block in ocr_result.blocks:
            if block.block_id not in seen_blocks:
                seen_blocks[block.block_id] = block
        
        blocks = sorted(seen_blocks.values(), key=lambda b: (b.y, b.x))
        
        for block in blocks:
            # Skip if block is junk (junk blocks don't affect text_on_top)
            if self._is_block_junk(block, ocr_result.blocks, ocr_result, new_chapter_threshold):
                continue
            
            block_text = ocr_result.get_block_text(block.block_id)
            
            # Check for section (centered text) - sections don't update text_on_top
            section = self._detect_section(
                block, block_text, page_width, page_height, text_on_top, ocr_result
            )
            if section:
                # Section detected: return it but don't update text_on_top
                # (it's just an announcement page)
                return section, text_on_top, False
            
            # Check for chapter with keyword
            if self.config.detect_only_on_header:
                chapter = self._detect_chapter_with_keyword(block, block_text)
                if chapter:
                    self.current_chapter_title = chapter.title
                    self.chapter_counter += 1
                    # Chapter detected, set text_on_top = True
                    return chapter, True, True
                # No chapter but block processed, update text_on_top
                text_on_top = True
                continue
            
            # General chapter detection based on position
            chapter = self._detect_chapter_by_position(
                block,
                block_text,
                page_height,
                new_chapter_threshold,
                max_chapter_position,
                text_on_top,
                ocr_result
            )
            
            if chapter:
                self.current_chapter_title = chapter.title
                self.chapter_counter += 1
                # Chapter detected, set text_on_top = True
                return chapter, True, True
            
            # CRITICAL: Any non-junk block processed must update text_on_top
            # This follows the original script logic (lines 310-312)
            if block.y > new_chapter_threshold and not text_on_top:
                text_on_top = True
            else:
                text_on_top = True
        
        return None, text_on_top, True
    
    def _is_block_junk(
        self,
        block: TextBlock,
        all_blocks: list[TextBlock],
        ocr_result: OCRResult,
        new_chapter_threshold: int
    ) -> bool:
        """
        Determine if a block is junk (page number, header, etc.).
        
        Args:
            block: Block to check
            all_blocks: All blocks on the page
            ocr_result: Complete OCR result
            new_chapter_threshold: Y-position threshold for chapters
            
        Returns:
            True if block is junk
        """
        block_text = ocr_result.get_block_text(block.block_id)
        
        # Check for page numbers
        if re.match(r"^ +[0-9]+ *$", block_text):
            logger.debug(f"Block {block.block_id} is a page number: {block_text}")
            return True
        
        # Check for blank
        if not block_text.strip():
            return True
        
        # Check for book title header
        if self.book_title:
            similarity = calculate_text_similarity(block_text, self.book_title)
            if similarity > 0.9:
                logger.debug(f"Block {block.block_id} matches book title: {similarity:.2f}")
                return True
        
        # Check for chapter title header (repeated at top of pages)
        if self.current_chapter_title:
            similarity = calculate_text_similarity(block_text, self.current_chapter_title)
            # Only junk if at top of page
            if similarity > 0.9 and block.y < new_chapter_threshold:
                logger.debug(f"Block {block.block_id} matches chapter header: {similarity:.2f}")
                return True
        
        # Check for low confidence blocks
        block_words = [
            b for b in all_blocks
            if b.block_id == block.block_id and b.level == 5  # Word level
        ]
        if block_words:
            low_conf_count = sum(1 for w in block_words if w.confidence < 80)
            if low_conf_count / len(block_words) > 0.6:
                logger.debug(f"Block {block.block_id} has low confidence")
                return True
        
        # Check for special characters
        clean_text = block_text.replace(" ", "")
        if clean_text:
            special_chars = r"<>&_()*+=\/{}#@¡¿†‡¶€¥ü∑∫¢@+-…äëïöüñßÄËÏÖÜÑ"
            special_count = sum(1 for c in clean_text if c in special_chars)
            if special_count / len(clean_text) > 0.3:
                logger.debug(f"Block {block.block_id} has too many special chars")
                return True
        
        return False
    
    def _detect_section(
        self,
        block: TextBlock,
        block_text: str,
        page_width: int,
        page_height: int,
        text_on_top: bool,
        ocr_result: OCRResult
    ) -> Optional[Chapter]:
        """Detect section (centered, mid-page text)."""
        # Must have actual letters (not just symbols)
        clean_text = block_text.strip()
        has_letters = any(c.isalpha() for c in clean_text)
        if not has_letters:
            return None
        
        # Must have at least 3 characters (avoid single char or symbols)
        if len(clean_text) < 3:
            return None
        
        # Must be centered (not full width)
        is_centered = (page_width - block.width) > (page_width / 3)
        
        # Must be in middle area (not at top)
        is_mid_page = block.y > (page_height / 5)
        
        # Must be short
        is_short = len(block_text) < 50
        
        # No text above it
        no_text_above = not text_on_top
        
        # Not the word "chapitre"
        not_chapter_keyword = "chapitre" not in block_text.lower()
        
        # Must be the only text on page
        full_page_text = ocr_result.full_text.strip()
        is_only_text = block_text.strip() == full_page_text
        
        if (is_centered and is_mid_page and is_short and 
            no_text_above and not_chapter_keyword and is_only_text):
            logger.info(f"Detected section: {block_text}")
            return Chapter(
                title=block_text.strip(),
                start_page=ocr_result.page_number,
                start_block_id=block.block_id,
                is_section=True
            )
        
        return None
    
    def _detect_chapter_with_keyword(
        self,
        block: TextBlock,
        block_text: str
    ) -> Optional[Chapter]:
        """Detect chapter based on keyword ('Chapitre')."""
        keyword = "Chapitre"
        if keyword in block_text and len(block_text) < len(keyword) + 4:
            title = re.sub(r"[\(\)\[\]\|]", "", block_text)
            logger.info(f"Detected chapter with keyword: {title}")
            return Chapter(
                title=title,
                start_page=block.page_number,
                start_block_id=block.block_id
            )
        return None
    
    def _detect_chapter_by_position(
        self,
        block: TextBlock,
        block_text: str,
        page_height: int,
        new_chapter_threshold: int,
        max_chapter_position: int,
        text_on_top: bool,
        ocr_result: OCRResult
    ) -> Optional[Chapter]:
        """Detect chapter based on position on page."""
        # Must have actual letters (not just symbols)
        clean_text = block_text.strip()
        has_letters = any(c.isalpha() for c in clean_text)
        if not has_letters:
            return None
        
        # Must have at least 3 characters (avoid single char or symbols)
        if len(clean_text) < 3:
            return None
        
        # Must be below threshold but not too far down
        in_valid_range = (
            block.y > (new_chapter_threshold * self.config.min_position_tolerance) and
            block.y < max_chapter_position
        )
        
        # No text above it yet
        no_text_above = not text_on_top
        
        if not (in_valid_range and no_text_above):
            return None
        
        logger.debug(f"Potential chapter at y={block.y}: {block_text}")
        
        # Determine chapter title
        if len(block_text) < 40:
            title = block_text
            
            # Check if title continues on next block
            next_block = self._get_next_block(block.block_id, ocr_result)
            if next_block and abs(block.height - next_block.height) < 5:
                next_text = ocr_result.get_block_text(next_block.block_id)
                title = f"{title} {next_text}".strip()
                logger.debug(f"Chapter title spans two blocks: {title}")
        else:
            # Long text blocks: assign generic chapter number
            # The old script behavior was to mark the first paragraph as "Chapitre N"
            title = "Chapitre"
        
        # Check if this block is the only text on page
        full_page_text = ocr_result.full_text.strip()
        is_only_text = block_text.strip() == full_page_text
        
        # If it's very short and only text, might be a standalone chapter title
        if is_only_text and len(block_text) < 40:
            # Don't include this block in content
            pass
        
        # Default chapter numbering if generic
        if title.lower() == "chapitre":
            title = f"Chapitre {self.chapter_counter}"
        
        # Clean title
        title = re.sub(r"[\(\)\[\]\|]", "", title)
        title = re.sub(r" +", " ", title)
        
        logger.info(f"Detected chapter: {title}")
        return Chapter(
            title=title,
            start_page=block.page_number,
            start_block_id=block.block_id
        )
    
    def _get_next_block(self, current_block_id: int, ocr_result: OCRResult) -> Optional[TextBlock]:
        """Get the next block after current one."""
        blocks = ocr_result.get_blocks_by_level(2)
        for block in blocks:
            if block.block_id == current_block_id + 1:
                return block
        return None
