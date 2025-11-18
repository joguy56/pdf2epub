"""Main pipeline orchestration with parallelization and error recovery."""

import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from pdf2epub.ai_proofreader import create_proofreader
from pdf2epub.chapter_detector import ChapterDetector
from pdf2epub.config import Pdf2EpubConfig
from pdf2epub.epub_generator import EPUBGenerator
from pdf2epub.ocr import EasyOCREngine, TesseractOCR
from pdf2epub.ocr.base import OCRResult
from pdf2epub.pdf_processor import PDFProcessor, PageImage
from pdf2epub.text_processor import TextProcessor
from pdf2epub.utils import PAGE_END_MARKER, get_file_base_name

logger = logging.getLogger(__name__)


@dataclass
class PipelineCheckpoint:
    """Checkpoint data for pipeline resumption."""
    
    pdf_path: str
    stage: str  # "pdf_converted", "ocr_complete", "text_processed"
    processed_pages: int
    total_pages: int
    intermediate_file: Optional[str] = None
    

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ConversionPipeline:
    """Main pipeline for PDF to EPUB conversion."""
    
    def __init__(self, config: Pdf2EpubConfig):
        """
        Initialize conversion pipeline.
        
        Args:
            config: Complete configuration
        """
        self.config = config
        
        # Initialize components
        self.pdf_processor = PDFProcessor(
            config.image_processing,
            config.output
        )
        
        self.ocr_engine = self._create_ocr_engine()
        
        self.chapter_detector = ChapterDetector(
            config.chapter_detection
        )
        
        self.text_processor = TextProcessor(
            config.text_processing
        )
        
        self.ai_proofreader = create_proofreader(config.ai_proofreading)
        
        self.checkpoint_file: Optional[Path] = None
    
    def _create_ocr_engine(self):
        """Create OCR engine based on configuration."""
        if self.config.ocr.engine == "tesseract":
            return TesseractOCR(
                language=self.config.ocr.language,
                confidence_threshold=self.config.ocr.confidence_threshold,
                tesseract_dir=self.config.ocr.tesseract_dir
            )
        else:
            return EasyOCREngine(
                language=self.config.ocr.language,
                confidence_threshold=self.config.ocr.confidence_threshold
            )
    
    def convert(
        self,
        pdf_path: Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        skip_pdf_conversion: bool = False,
        skip_ocr: bool = False
    ) -> Path:
        """
        Run full conversion pipeline.
        
        Args:
            pdf_path: Path to PDF file
            title: Book title (default: filename)
            author: Book author
            skip_pdf_conversion: Skip PDF to image conversion
            skip_ocr: Skip OCR (load from existing text file)
            
        Returns:
            Path to generated EPUB file
            
        Raises:
            PipelineError: If conversion fails
        """
        logger.info(f"Starting conversion pipeline for: {pdf_path}")
        
        # Set defaults
        if not title:
            title = get_file_base_name(pdf_path)
        if not author:
            author = "Unknown Author"
        
        # Setup checkpoint
        if self.config.performance.enable_resume:
            self.checkpoint_file = pdf_path.parent / f".{get_file_base_name(pdf_path)}_checkpoint.json"
        
        try:
            # Stage 1: PDF to images
            page_images = []
            if not skip_pdf_conversion:
                page_images = self._stage_pdf_conversion(pdf_path)
                self._save_checkpoint("pdf_converted", len(page_images))
            
            # Stage 2: OCR
            text = ""
            if not skip_ocr:
                text = self._stage_ocr(page_images if page_images else None, pdf_path, title)
                self._save_checkpoint("ocr_complete", 0)
            else:
                text = self._load_existing_text(pdf_path)
            
            # Stage 3: Text processing
            text = self._stage_text_processing(text)
            self._save_checkpoint("text_processed", 0)
            
            # Stage 4: AI proofreading (optional)
            if self.ai_proofreader:
                text = self._stage_ai_proofreading(text)
            
            # Stage 5: EPUB generation
            epub_path = self._stage_epub_generation(text, pdf_path, title, author)
            
            # Cleanup
            if self.checkpoint_file and self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            
            self.pdf_processor.cleanup()
            
            logger.info(f"✅ Conversion complete: {epub_path}")
            return epub_path
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise PipelineError(f"Conversion failed: {e}") from e
    
    def _stage_pdf_conversion(self, pdf_path: Path) -> list[PageImage]:
        """Stage 1: Convert PDF to images."""
        logger.info("📄 Stage 1: Converting PDF to images")
        
        # Check dependencies
        deps = self.pdf_processor.check_dependencies()
        if not all(deps.values()):
            logger.warning(f"Some dependencies missing: {deps}")
        
        # Convert PDF
        page_images = self.pdf_processor.convert_pdf_to_images(
            pdf_path,
            include_cover=self.config.output.include_cover
        )
        
        # Preprocess images
        page_images = self.pdf_processor.process_all_images(
            page_images,
            parallel=self.config.performance.parallel_processing
        )
        
        logger.info(f"✅ Converted {len(page_images)} pages")
        return page_images
    
    def _stage_ocr(
        self,
        page_images: Optional[list[PageImage]],
        pdf_path: Path,
        title: str
    ) -> str:
        """Stage 2: OCR processing."""
        logger.info("🔍 Stage 2: Running OCR")
        
        # Get images if not provided
        if not page_images:
            temp_dir = Path(self.config.output.temp_directory)
            import os
            image_files = sorted([
                temp_dir / f for f in os.listdir(temp_dir)
                if f.endswith(('.jpg', '.png')) and 'cover' not in f
            ])
            page_images = [
                PageImage(i, f, 0, 0)
                for i, f in enumerate(image_files, 1)
            ]
        
        # Process pages
        if self.config.performance.parallel_processing:
            ocr_results = self._process_pages_parallel(page_images, title)
        else:
            ocr_results = self._process_pages_sequential(page_images, title)
        
        # Combine results
        full_text = self._combine_ocr_results(ocr_results)
        
        # Save intermediate text file
        text_file = pdf_path.parent / f"{get_file_base_name(pdf_path)}_{self.config.ocr.engine}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        logger.info(f"✅ OCR complete: {len(ocr_results)} pages, saved to {text_file}")
        return full_text
    
    def _process_pages_sequential(
        self,
        page_images: list[PageImage],
        title: str
    ) -> list[str]:
        """Process pages sequentially."""
        results = []
        
        with tqdm(total=len(page_images), desc="OCR Progress") as pbar:
            for page_img in page_images:
                if page_img.is_cover:
                    continue
                
                page_text = self._process_single_page(page_img, title)
                results.append(page_text)
                pbar.update(1)
        
        return results
    
    def _process_pages_parallel(
        self,
        page_images: list[PageImage],
        title: str
    ) -> list[str]:
        """Process pages in parallel."""
        # Filter out cover pages and create new indexed list
        pages_to_process = [(idx, page_img) for idx, page_img in enumerate(page_images) if not page_img.is_cover]
        results = [None] * len(pages_to_process)
        
        max_workers = self.config.performance.max_workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_page, page_img, title): new_idx
                for new_idx, (orig_idx, page_img) in enumerate(pages_to_process)
            }
            
            with tqdm(total=len(futures), desc="OCR Progress") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        page_text = future.result()
                        results[idx] = page_text
                    except Exception as e:
                        logger.error(f"Failed to process page {idx + 1}: {e}")
                        results[idx] = ""
                    pbar.update(1)
        
        return [r for r in results if r is not None]
    
    def _process_single_page(self, page_img: PageImage, title: str) -> str:
        """Process a single page with OCR and chapter detection."""
        # Run OCR
        ocr_result = self.ocr_engine.process_image(page_img.file_path)
        
        # Detect chapter (text_on_top starts False for each page)
        chapter, _, _ = self.chapter_detector.detect_chapter(ocr_result, text_on_top=False)
        
        # Build page text
        page_text = ""
        if chapter:
            page_text += chapter.format_title()
        
        # Add content (filtered for junk)
        page_text += self._extract_clean_text(ocr_result, title)
        
        # Add page end marker
        page_text += f"\n{PAGE_END_MARKER}\n"
        
        return page_text
    
    def _extract_clean_text(self, ocr_result: OCRResult, title: str) -> str:
        """Extract clean text from OCR result, filtering junk."""
        from collections import defaultdict
        
        # Group words by block, paragraph, and line
        blocks_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for block in ocr_result.blocks:
            if block.paragraph_id is not None and block.line_id is not None:
                blocks_data[block.block_id][block.paragraph_id][block.line_id].append(block)
        
        clean_lines = []
        page_height = ocr_result.page_height
        
        # Process each block
        for block_id in sorted(blocks_data.keys()):
            block_lines = []
            
            # Reconstruct block text line by line
            for par_id in sorted(blocks_data[block_id].keys()):
                for line_id in sorted(blocks_data[block_id][par_id].keys()):
                    words = blocks_data[block_id][par_id][line_id]
                    # Sort words by x position
                    words.sort(key=lambda w: w.x)
                    line_text = " ".join(w.text for w in words if w.text.strip())
                    if line_text.strip():
                        block_lines.append(line_text)
            
            # Reconstruct full block text
            block_text = " ".join(block_lines)
            
            # Filter junk blocks
            if self._is_text_junk(block_text, blocks_data[block_id], page_height, title):
                continue
            
            # Add valid lines
            clean_lines.extend(block_lines)
        
        return "\n".join(clean_lines)
    
    def _is_text_junk(
        self,
        text: str,
        block_data: dict,
        page_height: int,
        title: str
    ) -> bool:
        """Check if text block is junk."""
        # Page numbers
        if re.match(r"^\s*[0-9]+\s*$", text):
            return True
        
        # Very short text
        if len(text.strip()) < 4:
            return True
        
        # Footer/page number position (bottom 10% of page)
        first_words = next(iter(next(iter(block_data.values())).values()), [])
        if first_words and first_words[0].y > page_height * 0.9:
            return True
        
        return False
    
    def _combine_ocr_results(self, results: list[str]) -> str:
        """Combine OCR results from all pages."""
        return "\n".join(results)
    
    def _stage_text_processing(self, text: str) -> str:
        """Stage 3: Text post-processing."""
        logger.info("✏️  Stage 3: Post-processing text")
        
        processed = self.text_processor.process_text(text)
        
        # Log quality metrics
        metrics = self.text_processor.validate_text_quality(processed)
        logger.info(f"Text metrics: {metrics}")
        
        logger.info("✅ Text processing complete")
        return processed
    
    def _stage_ai_proofreading(self, text: str) -> str:
        """Stage 4: AI proofreading."""
        logger.info("🤖 Stage 4: AI proofreading")
        
        try:
            corrected = self.ai_proofreader.proofread_text(text)
            logger.info("✅ AI proofreading complete")
            return corrected
        except Exception as e:
            logger.error(f"AI proofreading failed: {e}, using original text")
            return text
    
    def _stage_epub_generation(
        self,
        text: str,
        pdf_path: Path,
        title: str,
        author: str
    ) -> Path:
        """Stage 5: Generate EPUB."""
        logger.info("📚 Stage 5: Generating EPUB")
        
        # Get cover image path
        cover_path = None
        if self.config.output.include_cover:
            temp_dir = Path(self.config.output.temp_directory)
            cover_file = temp_dir / "cover.jpg"
            if cover_file.exists():
                cover_path = cover_file
        
        # Create generator
        generator = EPUBGenerator(
            title=title,
            author=author,
            language=self.config.ocr.language[:2],
            cover_image_path=cover_path
        )
        
        # Generate EPUB
        output_path = pdf_path.parent / f"{get_file_base_name(pdf_path)}.epub"
        epub_path = generator.generate_from_text(text, output_path)
        
        logger.info("✅ EPUB generation complete")
        return epub_path
    
    def _load_existing_text(self, pdf_path: Path) -> str:
        """Load text from existing intermediate file."""
        text_file = pdf_path.parent / f"{get_file_base_name(pdf_path)}_{self.config.ocr.engine}.txt"
        
        if not text_file.exists():
            raise PipelineError(f"Intermediate text file not found: {text_file}")
        
        with open(text_file, "r", encoding="utf-8") as f:
            return f.read()
    
    def _save_checkpoint(self, stage: str, processed_pages: int) -> None:
        """Save pipeline checkpoint."""
        if not self.checkpoint_file:
            return
        
        checkpoint = PipelineCheckpoint(
            pdf_path=str(self.checkpoint_file.parent),
            stage=stage,
            processed_pages=processed_pages,
            total_pages=processed_pages
        )
        
        with open(self.checkpoint_file, "w") as f:
            json.dump(asdict(checkpoint), f)
