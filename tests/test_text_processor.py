"""Unit tests for text_processor module."""

import pytest

from pdf2epub.config import TextProcessingConfig
from pdf2epub.text_processor import TextProcessor


@pytest.fixture
def text_processor():
    """Create text processor with default config."""
    config = TextProcessingConfig()
    return TextProcessor(config)


class TestTextProcessor:
    """Test text processing functionality."""
    
    def test_fix_hyphenation(self, text_processor):
        """Test word break fixes across lines."""
        text = "C'est un exem-\nple de texte"
        expected = "C'est un exemple de texte"
        result = text_processor.process_text(text)
        assert "exemple" in result
        assert "exem-\n" not in result
    
    def test_dialog_indentation(self, text_processor):
        """Test dialog line indentation."""
        text = "\n— Bonjour, dit-il.\n— Comment allez-vous ?\n"
        result = text_processor.process_text(text)
        assert "    —" in result
    
    def test_single_quote_replacement(self, text_processor):
        """Test special quote character replacement."""
        text = "C'est l'été"
        result = text_processor.process_text(text)
        assert "'" in result
        assert "'" not in result
    
    def test_lettrine_fix(self, text_processor):
        """Test lettrine (dropped capital) fix."""
        text = "\nC était une fois"
        result = text_processor.process_text(text)
        assert "\nCétait" in result or "C était" not in result
    
    def test_ocr_error_fixes(self, text_processor):
        """Test common OCR error corrections."""
        text = "]e suis content. ]'ai trouvé."
        result = text_processor.process_text(text)
        assert "Je suis" in result
        assert "J'ai" in result
    
    def test_custom_filters(self):
        """Test custom regex filters."""
        config = TextProcessingConfig(custom_filters=["HEADER", "Page [0-9]+"])
        processor = TextProcessor(config)
        
        text = "\nHEADER: Chapter 1\nSome text\nPage 42\nMore text\n"
        result = processor.process_text(text)
        
        assert "HEADER" not in result
        assert "Page 42" not in result
        assert "Some text" in result
        assert "More text" in result
    
    def test_extract_structure(self, text_processor):
        """Test structure extraction."""
        text = """
        Some intro text.
        
        @@@ Chapitre 1 @@@
        Chapter content here.
        
        $$$ Section Title $$$
        Section content.
        
        ~~~ Footnotes ~~~
        1. First footnote.
        ~~~
        """
        
        structure = text_processor.extract_structure(text)
        
        assert len(structure["chapters"]) == 1
        assert "Chapitre 1" in structure["chapters"]
        
        assert len(structure["sections"]) == 1
        assert "Section Title" in structure["sections"]
        
        assert len(structure["footnotes"]) == 1
    
    def test_validate_text_quality(self, text_processor):
        """Test text quality metrics."""
        text = "Ceci est un texte de test.\nAvec plusieurs lignes.\n"
        
        metrics = text_processor.validate_text_quality(text)
        
        assert metrics["total_chars"] > 0
        assert metrics["total_words"] > 0
        assert metrics["total_lines"] > 0
        assert 0 <= metrics["special_char_ratio"] <= 1
        assert 0 <= metrics["digit_ratio"] <= 1
