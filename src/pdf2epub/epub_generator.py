"""EPUB generation module."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from ebooklib import epub

from pdf2epub.utils import CHAPTER_MARKER, SECTION_MARKER, FOOTNOTE_MARKER, PAGE_END_MARKER, get_file_base_name

logger = logging.getLogger(__name__)


class EPUBGeneratorError(Exception):
    """Base exception for EPUB generation errors."""
    pass


class EPUBGenerator:
    """Generates EPUB files from processed text."""
    
    def __init__(
        self,
        title: str,
        author: str,
        language: str = "fr",
        cover_image_path: Optional[Path] = None
    ):
        """
        Initialize EPUB generator.
        
        Args:
            title: Book title
            author: Book author
            language: Book language code
            cover_image_path: Optional path to cover image
        """
        self.title = title
        self.author = author
        self.language = language
        self.cover_image_path = cover_image_path
        
        # Create book
        self.book = epub.EpubBook()
        self._setup_metadata()
    
    def _setup_metadata(self) -> None:
        """Set up EPUB metadata."""
        self.book.set_title(self.title)
        self.book.set_language(self.language)
        self.book.add_author(self.author)
        
        # Generate unique identifier
        timestamp = datetime.now().timestamp()
        title_short = self.title[:8] if len(self.title) > 8 else self.title
        identifier = f"{timestamp}_{title_short.replace(' ', '')}"
        self.book.set_identifier(identifier)
        
        # Add cover if provided
        if self.cover_image_path and self.cover_image_path.exists():
            with open(self.cover_image_path, "rb") as f:
                self.book.set_cover("cover.jpg", f.read())
    
    def generate_from_text(self, text: str, output_path: Path) -> Path:
        """
        Generate EPUB from processed text.
        
        Args:
            text: Processed text with markers
            output_path: Path to save EPUB file
            
        Returns:
            Path to generated EPUB file
            
        Raises:
            EPUBGeneratorError: If generation fails
        """
        try:
            logger.info("Generating EPUB")
            
            # Parse structure
            sections = self._parse_sections(text)
            chapters_list = []
            
            # Process each section
            for section_title, section_content in sections.items():
                # Parse chapters within section
                chapters = self._parse_chapters(section_content)
                logger.debug(f"Section '{section_title}': found {len(chapters)} chapters")
                
                for chapter_title, chapter_content in chapters.items():
                    logger.debug(f"  Creating chapter: {chapter_title}")
                    chapter = self._create_chapter(
                        chapter_title,
                        chapter_content,
                        section_title if section_title != "0" else None
                    )
                    chapters_list.append(chapter)
                    self.book.add_item(chapter)
            
            # Add CSS
            self._add_styles()
            
            # Add navigation
            self.book.add_item(epub.EpubNcx())
            self.book.add_item(epub.EpubNav())
            
            # Create spine
            self.book.spine = ['nav'] + chapters_list
            
            # Write EPUB
            output_path.parent.mkdir(parents=True, exist_ok=True)
            epub.write_epub(str(output_path), self.book)
            
            logger.info(f"EPUB generated: {output_path}")
            return output_path
            
        except Exception as e:
            raise EPUBGeneratorError(f"Failed to generate EPUB: {e}") from e
    
    def _parse_sections(self, text: str) -> dict[str, str]:
        """Parse text into sections."""
        sections = {}
        parts = text.split(SECTION_MARKER)
        
        if not parts[0].strip() or parts[0].strip() == '\n':
            parts.pop(0)
        elif len(parts) % 2 == 1 and len(parts) > 2:
            # First section has no title
            sections['0'] = parts.pop(0)
        
        # Parse section pairs
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                section_title = parts[i].strip()
                section_content = parts[i + 1]
                sections[section_title] = section_content
        
        # If no sections found, treat whole text as one section
        if not sections:
            sections['0'] = text
        
        return sections
    
    def _parse_chapters(self, text: str) -> dict[str, str]:
        """Parse section text into chapters."""
        chapters = {}
        parts = text.split(CHAPTER_MARKER)
        
        # Handle leading content without chapter marker
        if parts and parts[0].strip():
            if not re.match(r'^\s*\n', parts[0]):
                chapters['Introduction'] = parts[0]
        
        # Parse chapter pairs - start from index 1 (after leading content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                chapter_title = parts[i].strip()
                chapter_content = parts[i + 1]
                chapters[chapter_title] = chapter_content
        
        return chapters
    
    def _create_chapter(
        self,
        title: str,
        content: str,
        section_title: Optional[str] = None
    ) -> epub.EpubHtml:
        """Create an EPUB chapter."""
        # Generate file name
        filename = title.replace(' ', '_').replace('?', '').replace('!', '').replace('.', '')
        if not filename:  # Handle empty titles
            filename = "chapter"
        filename = f"{filename}.xhtml"
        
        # Create chapter
        chapter = epub.EpubHtml(title=title, file_name=filename, lang=self.language)
        
        # Build HTML content
        html = '<html><head>'
        if section_title:
            html += f'<h2>{section_title}</h2>'
        html += f'</head><body><h1>{title}</h1>'
        
        # Add paragraphs
        # Skip first empty line and the chapter title line if repeated
        lines = content.split('\n')
        start_index = 0
        
        # Skip leading empty lines and markers
        while start_index < len(lines):
            line_stripped = lines[start_index].strip()
            if not line_stripped or line_stripped in (PAGE_END_MARKER, FOOTNOTE_MARKER):
                start_index += 1
            else:
                break
        
        # Remove title from content if it appears at the start
        # The title might be on its own line or fused with the first paragraph
        logger.debug(f"    Title to skip: '{title}'")
        logger.debug(f"    Total lines in content: {len(lines)}, starting at index {start_index}")
        
        if start_index < len(lines):
            first_line = lines[start_index].strip()
            
            # Check if the line starts with the title
            if first_line == title:
                # Title is on its own line, skip it
                start_index += 1
                logger.debug(f"    ✓ Skipped title on separate line")
            elif first_line.startswith(title):
                # Title is fused with text, remove it from the line
                remaining_text = first_line[len(title):].strip()
                if remaining_text:
                    # Replace the line with just the remaining text
                    lines[start_index] = remaining_text
                    logger.debug(f"    ✓ Removed title prefix, kept: '{remaining_text[:50]}...'")
                else:
                    # Line only contained the title, skip it
                    start_index += 1
                    logger.debug(f"    ✓ Skipped title-only line")
        
        logger.debug(f"    Generating paragraphs from index {start_index} to {len(lines)}")
        
        para_count = 0
        for line in lines[start_index:]:
            line = line.strip()
            # Skip page end markers (§), footnote markers (~~~) and empty lines
            if line and line not in (PAGE_END_MARKER, FOOTNOTE_MARKER):
                html += f'<p>{line}</p>'
                para_count += 1
        
        # Debug log
        p_count = html.count('<p>')
        logger.debug(f"    Generated {p_count} paragraphs (loop counted {para_count}) for chapter '{title}'")
        
        html += '</body></html>'
        chapter.content = html
        
        return chapter
    
    def _add_styles(self) -> None:
        """Add CSS styles to EPUB."""
        default_css = """
        BODY { 
            text-align: justify;
            font-family: Georgia, serif;
            line-height: 1.6;
        }
        h1 {
            text-align: center;
            margin-top: 2em;
            margin-bottom: 1em;
        }
        h2 {
            text-align: center;
            font-style: italic;
            margin-bottom: 1em;
        }
        p {
            text-indent: 1.5em;
            margin: 0.5em 0;
        }
        """
        
        default_css_item = epub.EpubItem(
            uid="style_default",
            file_name="style/default.css",
            media_type="text/css",
            content=default_css
        )
        self.book.add_item(default_css_item)
        
        # Navigation CSS
        nav_css = """
        nav {
            font-family: sans-serif;
        }
        """
        
        nav_css_item = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=nav_css
        )
        self.book.add_item(nav_css_item)
