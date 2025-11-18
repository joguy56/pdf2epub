"""Text post-processing module."""

import logging
import re
from typing import Optional

from pdf2epub.config import TextProcessingConfig
from pdf2epub.utils import (
    ACCENTED_CHARS,
    FOOTNOTE_MARKER,
    HYPHEN_CHARS,
    PAGE_END_MARKER,
    clean_special_quotes,
    compile_accent_pattern,
    remove_blank_lines,
)

logger = logging.getLogger(__name__)


class TextProcessor:
    """Post-processes OCR text to clean up common issues."""
    
    def __init__(self, config: TextProcessingConfig):
        """
        Initialize text processor.
        
        Args:
            config: Text processing configuration
        """
        self.config = config
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for performance."""
        # Double hyphens at line start
        self.pattern_double_hyphen = re.compile(r"\n( *)" + HYPHEN_CHARS + r"+( *)")
        
        # Simple hyphens between words
        self.pattern_word_hyphen = compile_accent_pattern(
            r"([a-zA-Z%s]+) *" + HYPHEN_CHARS + r"+ *([a-zA-Z%s]+)"
        )
        
        # Word breaks across lines
        self.pattern_line_break = compile_accent_pattern(
            r"([a-zA-Z%s]+) *" + HYPHEN_CHARS + r" *\n+ *([a-zA-Z%s]+)"
        )
        
        # Dialog lines
        self.pattern_dialog = compile_accent_pattern(
            r'\n ?(' + HYPHEN_CHARS + r') ?([a-zA-Z%s])(.*)(\n|$)'
        )
        
        # Lettrine (dropped capital)
        self.pattern_lettrine = re.compile(r'\n([a-zA-Z]) ')
        
        # Words concatenated across lines
        self.pattern_concat_lines = compile_accent_pattern(
            r"([a-zA-Z%s,;])\n+([a-zA-z%s])"
        )
        
        # Footnote marker (lines starting with digit and period)
        self.pattern_footnote = re.compile(
            r"(\n1\..*?" + re.escape(PAGE_END_MARKER) + r")",
            flags=re.DOTALL
        )
    
    def process_text(self, text: str | list[str]) -> str:
        """
        Process text through all cleaning steps.
        
        Args:
            text: Either a single string or list of page texts
            
        Returns:
            Cleaned text
        """
        # Convert to list if single string
        if isinstance(text, str):
            pages = [text]
        else:
            pages = text
        
        # Process each page
        processed_pages = []
        for page_text in pages:
            processed = self._process_page(page_text)
            processed_pages.append(processed)
        
        # Join pages
        full_text = "\n".join(processed_pages)
        
        # Apply global processing
        full_text = self._process_full_text(full_text)
        
        logger.info(f"Text processing complete: {len(full_text)} characters")
        return full_text
    
    def _process_page(self, page_text: str) -> str:
        """
        Process a single page of text.
        
        Args:
            page_text: Text from one page
            
        Returns:
            Processed page text
        """
        text = page_text
        
        # Fix hyphenation issues
        if self.config.fix_hyphenation:
            # Double hyphens at line start (dialog)
            text = self.pattern_double_hyphen.sub(r"\n\1-\2", text)
            
            # Simple hyphens between words
            text = self.pattern_word_hyphen.sub(r"\1-\2", text)
            
            # Word breaks across lines
            text = self.pattern_line_break.sub(r"\1\2", text)
        
        # Remove blank lines
        if self.config.remove_blank_lines:
            text = remove_blank_lines(text)
        
        # Fix single quote character
        text = clean_special_quotes(text)
        
        # Fix dialog indentation
        if self.config.fix_dialogs:
            text = self.pattern_dialog.sub(r"\n    \1 \2\3\n", text)
            # Apply twice to catch consecutive dialog lines
            text = self.pattern_dialog.sub(r"\n    \1 \2\3\n", text)
        
        # Fix lettrine (dropped capital at start of paragraph)
        if self.config.fix_lettrine:
            text = self.pattern_lettrine.sub(r'\n\1', text)
        
        # Apply custom filters
        for filter_pattern in self.config.custom_filters:
            try:
                text = re.sub(r"\n.*" + filter_pattern + r".*\n", "\n", text)
            except re.error as e:
                logger.warning(f"Invalid filter pattern '{filter_pattern}': {e}")
        
        # Fix common OCR errors
        text = self._fix_common_ocr_errors(text)
        
        return text
    
    def _process_full_text(self, text: str) -> str:
        """
        Process full text (after pages are joined).
        
        Args:
            text: Full text from all pages
            
        Returns:
            Processed full text
        """
        # Mark footnotes
        text = self.pattern_footnote.sub(
            r'\n' + FOOTNOTE_MARKER + r'\n\1\n' + FOOTNOTE_MARKER + r'\n',
            text
        )
        
        # Remove page end markers
        text = text.replace(PAGE_END_MARKER, '')
        
        # Fix word concatenation across pages
        text = self.pattern_concat_lines.sub(r"\1 \2", text)
        
        # Final blank line cleanup
        text = text.replace("\n\n", "\n")
        
        return text
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR recognition errors.
        
        Args:
            text: Input text
            
        Returns:
            Text with common errors fixed
        """
        # Malformed "Je" (often recognized as "]e")
        text = text.replace(']e', 'Je')
        text = text.replace("]'", "J'")
        
        # Common letter confusions
        replacements = {
            # Add more as needed based on your OCR results
            ' rn ': ' m ',  # "m" recognized as "rn"
            ' cl ': ' d ',  # "d" recognized as "cl"
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def extract_structure(self, text: str) -> dict[str, list[str]]:
        """
        Extract structural elements from text.
        
        Returns dictionary with:
        - chapters: List of chapter titles
        - sections: List of section titles
        - footnotes: List of footnote blocks
        
        Args:
            text: Processed text
            
        Returns:
            Dictionary of structural elements
        """
        from pdf2epub.utils import CHAPTER_MARKER, SECTION_MARKER, FOOTNOTE_MARKER
        
        structure = {
            "chapters": [],
            "sections": [],
            "footnotes": []
        }
        
        # Extract chapters
        chapter_pattern = re.compile(
            re.escape(CHAPTER_MARKER) + r" (.*?) " + re.escape(CHAPTER_MARKER)
        )
        structure["chapters"] = chapter_pattern.findall(text)
        
        # Extract sections
        section_pattern = re.compile(
            re.escape(SECTION_MARKER) + r" (.*?) " + re.escape(SECTION_MARKER)
        )
        structure["sections"] = section_pattern.findall(text)
        
        # Extract footnotes
        footnote_pattern = re.compile(
            re.escape(FOOTNOTE_MARKER) + r"(.*?)" + re.escape(FOOTNOTE_MARKER),
            flags=re.DOTALL
        )
        structure["footnotes"] = footnote_pattern.findall(text)
        
        logger.info(
            f"Extracted structure: {len(structure['chapters'])} chapters, "
            f"{len(structure['sections'])} sections, {len(structure['footnotes'])} footnote blocks"
        )
        
        return structure
    
    def validate_text_quality(self, text: str) -> dict[str, any]:
        """
        Analyze text quality metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "total_chars": len(text),
            "total_words": len(text.split()),
            "total_lines": text.count("\n"),
            "avg_line_length": 0.0,
            "special_char_ratio": 0.0,
            "digit_ratio": 0.0,
        }
        
        if metrics["total_lines"] > 0:
            metrics["avg_line_length"] = metrics["total_chars"] / metrics["total_lines"]
        
        # Calculate character type ratios
        clean_text = text.replace("\n", "").replace(" ", "")
        if clean_text:
            special_chars = sum(1 for c in clean_text if not c.isalnum() and c not in ACCENTED_CHARS)
            metrics["special_char_ratio"] = special_chars / len(clean_text)
            
            digits = sum(1 for c in clean_text if c.isdigit())
            metrics["digit_ratio"] = digits / len(clean_text)
        
        return metrics
