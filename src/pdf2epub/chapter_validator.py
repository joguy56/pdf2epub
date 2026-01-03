"""Chapter validation module to detect OCR numbering errors."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChapterValidationError:
    """Represents a chapter validation error."""
    
    line_number: int
    chapter_text: str
    error_type: str
    expected_number: Optional[int] = None
    found_number: Optional[int] = None
    suggestion: str = ""


class ChapterValidator:
    """Validates chapter numbering and detects OCR errors."""
    
    def __init__(self):
        """Initialize chapter validator."""
        self.errors = []
        self.warnings = []
    
    def validate_file(self, file_path: str) -> tuple[list[ChapterValidationError], list[ChapterValidationError]]:
        """
        Validate chapter numbering in a text file.
        
        Args:
            file_path: Path to the OCR text file
            
        Returns:
            Tuple of (errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        chapters = self._extract_chapters(lines)
        
        if not chapters:
            logger.warning("⚠️  No chapters found in file")
            return self.errors, self.warnings
        
        self._validate_chapter_sequence(chapters)
        self._detect_fake_chapters(chapters)
        self._validate_chapter_format(chapters)
        
        return self.errors, self.warnings
    
    def validate_file_from_text(self, text: str) -> tuple[list[ChapterValidationError], list[ChapterValidationError]]:
        """
        Validate chapter numbering from text content.
        
        Args:
            text: Text content to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        lines = text.split('\n')
        chapters = self._extract_chapters(lines)
        
        if not chapters:
            logger.warning("⚠️  No chapters found in text")
            return self.errors, self.warnings
        
        self._validate_chapter_sequence(chapters)
        self._detect_fake_chapters(chapters)
        self._validate_chapter_format(chapters)
        
        return self.errors, self.warnings
    
    def _extract_chapters(self, lines: list[str]) -> list[tuple[int, str, Optional[int]]]:
        """
        Extract all chapter markers from lines.
        
        Returns:
            List of (line_number, chapter_text, chapter_number)
        """
        chapters = []
        chapter_pattern = re.compile(r'^@@@\s*(.*?)\s*@@@$')
        
        for line_num, line in enumerate(lines, start=1):
            match = chapter_pattern.match(line.strip())
            if match:
                chapter_text = match.group(1)
                chapter_number = self._extract_chapter_number(chapter_text)
                chapters.append((line_num, chapter_text, chapter_number))
        
        logger.info(f"📋 Found {len(chapters)} chapter markers")
        return chapters
    
    def _extract_chapter_number(self, chapter_text: str) -> Optional[int]:
        """
        Extract chapter number from chapter text.
        
        Examples:
            "1. TITRE" -> 1
            "42. ANOTHER TITLE" -> 42
            "Chapitre 5" -> 5
            "SECTION TITLE" -> None
        """
        # Pattern: number followed by period and space
        match = re.match(r'^(\d+)\.\s+', chapter_text)
        if match:
            return int(match.group(1))
        
        # Pattern: "Chapitre" followed by number
        match = re.search(r'[Cc]hapitre\s+(\d+)', chapter_text)
        if match:
            return int(match.group(1))
        
        return None
    
    def _validate_chapter_sequence(self, chapters: list[tuple[int, str, Optional[int]]]):
        """Validate that chapter numbers form a proper sequence."""
        numbered_chapters = [(line, text, num) for line, text, num in chapters if num is not None]
        
        if not numbered_chapters:
            self.warnings.append(ChapterValidationError(
                line_number=0,
                chapter_text="",
                error_type="no_numbered_chapters",
                suggestion="No numbered chapters found - this may be intentional"
            ))
            return
        
        # Extract just the numbers
        numbers = [num for _, _, num in numbered_chapters]
        
        # Check for duplicates
        seen = set()
        for line, text, num in numbered_chapters:
            if num in seen:
                self.errors.append(ChapterValidationError(
                    line_number=line,
                    chapter_text=text,
                    error_type="duplicate_number",
                    found_number=num,
                    suggestion=f"Chapter {num} appears multiple times - possible OCR error"
                ))
            seen.add(num)
        
        # Check for gaps and sequence issues
        sorted_numbers = sorted(numbers)
        expected_start = 1
        
        # Allow starting from number other than 1 (prologue, etc.)
        if sorted_numbers[0] != 1:
            self.warnings.append(ChapterValidationError(
                line_number=numbered_chapters[0][0],
                chapter_text=numbered_chapters[0][1],
                error_type="unexpected_start",
                expected_number=1,
                found_number=sorted_numbers[0],
                suggestion=f"Chapters start at {sorted_numbers[0]} instead of 1 - verify this is intentional"
            ))
            expected_start = sorted_numbers[0]
        
        # Check for gaps in sequence
        for i in range(len(sorted_numbers) - 1):
            current = sorted_numbers[i]
            next_num = sorted_numbers[i + 1]
            
            if next_num != current + 1:
                gap_size = next_num - current - 1
                
                # Find the line with this number
                line, text = next((l, t) for l, t, n in numbered_chapters if n == next_num)
                
                self.errors.append(ChapterValidationError(
                    line_number=line,
                    chapter_text=text,
                    error_type="sequence_gap",
                    expected_number=current + 1,
                    found_number=next_num,
                    suggestion=f"Gap of {gap_size} chapters: {current} → {next_num}. "
                              f"Likely OCR error: check if {next_num} should be {current + 1}"
                ))
        
        # Check for out-of-order chapters
        for i in range(len(numbered_chapters) - 1):
            current_line, current_text, current_num = numbered_chapters[i]
            next_line, next_text, next_num = numbered_chapters[i + 1]
            
            if next_num < current_num:
                self.errors.append(ChapterValidationError(
                    line_number=next_line,
                    chapter_text=next_text,
                    error_type="out_of_order",
                    expected_number=current_num + 1,
                    found_number=next_num,
                    suggestion=f"Chapter {next_num} appears after {current_num} - likely OCR misread"
                ))
    
    def _detect_fake_chapters(self, chapters: list[tuple[int, str, Optional[int]]]):
        """Detect fake chapters created by OCR errors."""
        for line, text, num in chapters:
            # Detect "Chapitre X" without proper number format
            if re.match(r'^[Cc]hapitre\s+\d+$', text.strip()):
                self.warnings.append(ChapterValidationError(
                    line_number=line,
                    chapter_text=text,
                    error_type="suspicious_format",
                    suggestion=f"Chapter marker '{text}' looks like OCR error - should have title"
                ))
            
            # Detect single-word chapters (often OCR errors)
            words = text.strip().split()
            if len(words) == 1 and num is None:
                self.warnings.append(ChapterValidationError(
                    line_number=line,
                    chapter_text=text,
                    error_type="single_word_chapter",
                    suggestion=f"Single-word chapter '{text}' might be OCR error"
                ))
    
    def _validate_chapter_format(self, chapters: list[tuple[int, str, Optional[int]]]):
        """Validate chapter formatting consistency."""
        numbered_count = sum(1 for _, _, num in chapters if num is not None)
        unnumbered_count = len(chapters) - numbered_count
        
        # If we have both numbered and unnumbered, check ratio
        if numbered_count > 0 and unnumbered_count > 0:
            if unnumbered_count > numbered_count * 0.2:  # More than 20% unnumbered
                self.warnings.append(ChapterValidationError(
                    line_number=0,
                    chapter_text="",
                    error_type="mixed_format",
                    suggestion=f"Found {numbered_count} numbered and {unnumbered_count} unnumbered chapters - verify consistency"
                ))
    
    def print_report(self):
        """Print validation report."""
        if not self.errors and not self.warnings:
            logger.info("✅ Chapter validation passed - no issues found")
            return
        
        if self.errors:
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ CHAPTER VALIDATION ERRORS ({len(self.errors)} found)")
            logger.error(f"{'='*80}")
            
            for error in self.errors:
                logger.error(f"\n📍 Line {error.line_number}: {error.chapter_text}")
                logger.error(f"   Type: {error.error_type}")
                
                if error.expected_number and error.found_number:
                    logger.error(f"   Expected: {error.expected_number}, Found: {error.found_number}")
                
                if error.suggestion:
                    logger.error(f"   💡 {error.suggestion}")
        
        if self.warnings:
            logger.warning(f"\n{'='*80}")
            logger.warning(f"⚠️  CHAPTER VALIDATION WARNINGS ({len(self.warnings)} found)")
            logger.warning(f"{'='*80}")
            
            for warning in self.warnings:
                logger.warning(f"\n📍 Line {warning.line_number}: {warning.chapter_text}")
                logger.warning(f"   Type: {warning.error_type}")
                
                if warning.suggestion:
                    logger.warning(f"   💡 {warning.suggestion}")
        
        logger.info(f"\n{'='*80}\n")


def validate_chapters(file_path: str) -> bool:
    """
    Validate chapter numbering in a file.
    
    Args:
        file_path: Path to the OCR text file
        
    Returns:
        True if validation passed (no errors), False otherwise
    """
    validator = ChapterValidator()
    errors, warnings = validator.validate_file(file_path)
    validator.print_report()
    
    return len(errors) == 0
