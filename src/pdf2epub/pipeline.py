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
from pdf2epub.chapter_validator import ChapterValidator
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
            if not skip_ocr:
                text = self._stage_text_processing(text)
                self._save_checkpoint("text_processed", 0)
                
                # Save processed text to file
                text_file = pdf_path.parent / f"{get_file_base_name(pdf_path)}_tesseract.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"✅ Saved post-processed text: {len(text.split())} words")
            
            # Stage 4: AI proofreading (optional)
            if self.ai_proofreader:
                text = self._stage_ai_proofreading(text, pdf_path)
            
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
        
        word_count = len(full_text.split())
        logger.info(f"✅ OCR complete: {len(ocr_results)} pages, {word_count} words")
        return full_text
    
    def _process_pages_sequential(
        self,
        page_images: list[PageImage],
        title: str
    ) -> list[str]:
        """Process pages sequentially in order."""
        results = []
        
        # Get cover OCR mode from config
        cover_ocr_mode = getattr(self.config, '_cover_ocr_mode', 'include')
        
        with tqdm(total=len(page_images), desc="OCR Progress") as pbar:
            for idx, page_img in enumerate(page_images):
                # Skip cover if mode is "skip"
                if page_img.is_cover and cover_ocr_mode == "skip":
                    logger.info(f"⏭️  Skipping OCR on cover page (used as image only)")
                    pbar.update(1)
                    continue
                
                page_text = self._process_single_page(page_img, title, idx + 1)
                results.append(page_text)
                pbar.update(1)
        
        logger.info(f"📄 Processed {len(results)} pages sequentially")
        return results
    
    def _process_pages_parallel(
        self,
        page_images: list[PageImage],
        title: str
    ) -> list[str]:
        """
        Process pages in parallel while preserving page order.
        Uses indexed results to guarantee output order matches input order.
        """
        # Get cover OCR mode from config
        cover_ocr_mode = getattr(self.config, '_cover_ocr_mode', 'include')
        
        # Filter pages based on cover mode
        pages_to_process = []
        for idx, page_img in enumerate(page_images):
            if page_img.is_cover and cover_ocr_mode == "skip":
                logger.info(f"⏭️  Skipping OCR on cover page (used as image only)")
                continue
            pages_to_process.append((idx, page_img))
        
        results = [None] * len(pages_to_process)
        
        max_workers = self.config.performance.max_workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their index
            futures = {
                executor.submit(self._process_single_page, page_img, title, orig_idx + 1): new_idx
                for new_idx, (orig_idx, page_img) in enumerate(pages_to_process)
            }
            
            with tqdm(total=len(futures), desc="OCR Progress") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        page_text = future.result()
                        results[idx] = page_text
                        # Log with page number for clarity
                        logger.debug(f"Completed page {idx + 1}/{len(results)}")
                    except Exception as e:
                        logger.error(f"Failed to process page {idx + 1}: {e}")
                        results[idx] = ""
                    pbar.update(1)
        
        # Verify all pages were processed
        if None in results:
            missing = [i + 1 for i, r in enumerate(results) if r is None]
            logger.warning(f"⚠️  Missing results for pages: {missing}")
        
        # Return results in correct order (filter None values)
        ordered_results = [r for r in results if r is not None]
        logger.info(f"📄 Processed {len(ordered_results)} pages in correct order")
        return ordered_results
    
    def _process_single_page(self, page_img: PageImage, title: str, page_num: Optional[int] = None) -> str:
        """
        Process a single page with OCR and chapter detection.
        
        Args:
            page_img: Page image to process
            title: Book title for junk filtering
            page_num: Optional page number for logging
        
        Returns:
            Extracted and formatted text for this page
        """
        page_label = f"page {page_num}" if page_num else "page"
        logger.debug(f"Processing {page_label}: {page_img.file_path.name}")
        
        # Run OCR
        ocr_result = self.ocr_engine.process_image(page_img.file_path)
        
        # Detect chapter (text_on_top starts False for each page)
        chapter, _, _ = self.chapter_detector.detect_chapter(ocr_result, text_on_top=False)
        
        # Build page text
        page_text = ""
        if chapter:
            page_text += chapter.format_title()
            logger.debug(f"  → Chapter detected on {page_label}: {chapter.title}")
        
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
        
        # ⚠️ CRITICAL: Validate chapter numbering for OCR errors
        logger.info("🔍 Validating chapter numbering...")
        validator = ChapterValidator()
        errors, warnings = validator.validate_file_from_text(processed)
        
        if errors or warnings:
            validator.print_report()
            
            if errors:
                logger.error("❌ CRITICAL: Chapter validation found errors!")
                logger.error("   OCR likely misread chapter numbers (e.g., 17→47, 20→0)")
                logger.error("   These MUST be fixed before AI proofreading to avoid wasted API calls.")
                logger.error("   Please review and correct chapter numbers manually.")
                
                # Check if we're in batch mode (non-interactive)
                batch_mode = getattr(self.config, '_batch_mode', False)
                
                if batch_mode:
                    logger.error("   ⚠️  Running in batch mode - continuing anyway")
                    logger.error("   ⚠️  EPUB may have broken chapter navigation!")
                else:
                    # Interactive mode: ask user
                    try:
                        user_input = input("\n⚠️  Continue anyway? (yes/no): ").strip().lower()
                        if user_input not in ['yes', 'y', 'o', 'oui']:
                            raise PipelineError("Pipeline stopped for chapter validation errors")
                    except (EOFError, KeyboardInterrupt):
                        # If input fails (e.g., in non-interactive context), treat as batch mode
                        logger.error("   ⚠️  Cannot get user input - continuing anyway")
        else:
            logger.info("✅ Chapter validation passed - no issues found")
        
        logger.info("✅ Text processing complete")
        return processed
    
    def _stage_ai_proofreading(self, text: str, pdf_path: Path) -> str:
        """Stage 4: AI proofreading."""
        logger.info("🤖 Stage 4: AI proofreading")
        
        input_words = len(text.split())
        logger.info(f"Input: {input_words} words")
        
        try:
            corrected = self.ai_proofreader.proofread_text(text)
            output_words = len(corrected.split())
            word_diff = output_words - input_words
            logger.info(f"Output: {output_words} words ({word_diff:+d} words, {word_diff/input_words*100:+.1f}%)")
            
            # Save AI-corrected text to separate file
            ai_text_file = pdf_path.parent / f"{get_file_base_name(pdf_path)}_tesseract_ai.txt"
            with open(ai_text_file, "w", encoding="utf-8") as f:
                f.write(corrected)
            logger.info(f"✅ AI proofreading complete, saved to {ai_text_file.name}")
            
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
        logger.info(f"Input text: {len(text.split())} words")
        
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
        # Prefer AI-corrected text if available
        ai_text_file = pdf_path.parent / f"{get_file_base_name(pdf_path)}_tesseract_ai.txt"
        text_file = pdf_path.parent / f"{get_file_base_name(pdf_path)}_tesseract.txt"
        
        if ai_text_file.exists():
            logger.info(f"Loading AI-corrected text from {ai_text_file.name}")
            with open(ai_text_file, "r", encoding="utf-8") as f:
                text = f.read()
                logger.info(f"Loaded {len(text.split())} words")
                return text
        elif text_file.exists():
            logger.info(f"Loading post-processed text from {text_file.name}")
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
                logger.info(f"Loaded {len(text.split())} words")
                return text
        else:
            raise PipelineError(f"Intermediate text file not found: {text_file}")
    
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
