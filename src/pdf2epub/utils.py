"""Utility functions and constants for pdf2epub."""

import re
from typing import Final

# Accent characters for French text processing
ACCENTED_CHARS: Final[str] = "àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßØøÅåÆæœ"

# Special markers for text structure
CHAPTER_MARKER: Final[str] = "@@@"
SECTION_MARKER: Final[str] = "$$$"
FOOTNOTE_MARKER: Final[str] = "~~~"
PAGE_END_MARKER: Final[str] = "§"

# Regex patterns for text processing
HYPHEN_CHARS: Final[str] = r"[\u2014-]"  # Em-dash and hyphen range


def compile_accent_pattern(pattern: str) -> re.Pattern[str]:
    """
    Compile a regex pattern with accent character support.
    
    Args:
        pattern: Pattern string with %s placeholders for accent characters
        
    Returns:
        Compiled regex pattern
        
    Example:
        >>> pattern = compile_accent_pattern(r"([a-zA-Z%s]+)")
        >>> pattern.match("café")
    """
    # Count %s occurrences and substitute with accent chars
    accent_count = pattern.count("%s")
    if accent_count > 0:
        pattern = pattern % tuple([ACCENTED_CHARS] * accent_count)
    return re.compile(pattern)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Remove trailing/leading whitespace from each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text


def remove_blank_lines(text: str) -> str:
    """
    Remove consecutive blank lines, keeping at most one.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized blank lines
    """
    return re.sub(r"\n\n+", "\n", text)


def clean_special_quotes(text: str) -> str:
    """
    Replace special quote characters with standard ones.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized quotes
    """
    return text.replace("'", "'")


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two text strings.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    from difflib import SequenceMatcher
    
    # Normalize texts before comparison
    text1_normalized = re.sub(r" +", " ", text1.strip())
    text2_normalized = re.sub(r" +", " ", text2.strip())
    
    return SequenceMatcher(None, text1_normalized, text2_normalized).ratio()


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate if a file path is valid and optionally exists.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        True if valid, False otherwise
    """
    import os
    
    if not file_path:
        return False
    
    if must_exist:
        return os.path.isfile(file_path)
    
    # Check if parent directory exists
    parent_dir = os.path.dirname(file_path)
    return os.path.isdir(parent_dir) if parent_dir else True


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Directory path
    """
    import os
    
    os.makedirs(directory, exist_ok=True)


def get_file_base_name(file_path: str) -> str:
    """
    Get base name of file without extension.
    
    Args:
        file_path: Full file path
        
    Returns:
        Base name without extension
        
    Example:
        >>> get_file_base_name("/path/to/book.pdf")
        "book"
    """
    import os
    
    return os.path.splitext(os.path.basename(file_path))[0]


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current += increment
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%)", end="")
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def __enter__(self) -> "ProgressTracker":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.current < self.total:
            print()  # Ensure newline if interrupted
