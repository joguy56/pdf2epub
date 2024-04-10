import os
import re
import time
import shutil
import subprocess
from PIL import Image
import easyocr
import platform
from pytesseract import *
import cv2
import imutils
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from pdf2image import pdfinfo_from_path, convert_from_path
from os.path import exists
import argparse
from datetime import datetime
import language_tool_python
import spacy
from ebooklib import epub
import logging
import colorlog
from difflib import SequenceMatcher

## Initialize configuration parser
#config = configparser.ConfigParser()
#config.read('config.ini')  # Assuming the configuration file is named 'config.ini'

# Remove existing loggers to prevent duplicate logs
root_logger = logging.getLogger()
root_logger.handlers = []

# Set up logging
def get_logger():
    logger = logging.getLogger(__name__)
    return logger
logger = get_logger()
logger.setLevel(logging.DEBUG)  # Set the log level

# Set up colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
))
logger.addHandler(handler)

# Create a temporary directory to hold our temporary images.
tempdir = "./tmp"
pd.options.display.max_rows = None


def parse_options():
    """
    Parse command-line arguments.
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Program that converts pdf text file to epub.")

    # Add optional arguments
    parser.add_argument("-i", "--input", help="Nom du fichier d'entrée", required=True)
    parser.add_argument("-O", "--OCR", help="The OCR to use easyOCR|tesseract", required=False, default="tesseract")
    parser.add_argument("--tesseract-dir", help="the directory of tesseract binary", required=False, default='')
    parser.add_argument("-r", "--recognize-only", help="starts directly to the recognition step", action="store_true", required=False)
    parser.add_argument("-g", "--generate-epub-only", help="starts directly to the epub generation", action="store_true", required=False)
    parser.add_argument("-f", "--filter", help="some lines that should be filtered", action="append", required=False)
    parser.add_argument("-l", "--language", help="french (fra - default) or english (eng)", required=False, default='fra')
    parser.add_argument("-a", "--author", help="the author of the book", required=False)
    parser.add_argument("-t", "--title", help="the book's title", required=False, default='')
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true", required=False, default=False)
    parser.add_argument("--no-chap-detection", help="disable chapter detection (enabled by default)", action="store_true", required=False, default=False)
    parser.add_argument("--chap-detect-thres-pct", help="percentage of page height from which a new chapter is beginning", required=False, default=25, type=int)
    parser.add_argument("--no-cover", help="no cover image for this ebook", action='store_true', required=False, default=False)
    parser.add_argument("-x", "--x-margin", help="x cropping margin", default="30", required=False)
    parser.add_argument("-y", "--y-margin", help="y cropping margin", default="50", required=False)

    # Parse command-line arguments
    args = parser.parse_args()

    if args.OCR == 'tesseract':
        if not 'TESSDATA_PREFIX' in os.environ and not args.tesseract_dir:
            raise FileNotFoundError("At least the tesseract directory needs to be known. Use TESSDATA_PREFIX environment variable or the tesseract_dir option")
        elif args.tesseract_dir:
            #os.environ['TESSDATA_PREFIX'] = "C:/Users/jguyot/AppData/Local/Programs/Tesseract-OCR/tessdata"
            os.environ['TESSDATA_PREFIX'] = args.tesseract_dir

    # Log parsed arguments
    logger.info(f"Parsed options: {args}")
    return args


def convert_pdf(args):
    """
    Convert PDF to images.
    """
    # Log creation of temporary directory
    logger.info(f"Creating {tempdir} directory")

    # Get information about the PDF file
    info = pdfinfo_from_path(args.input, userpw=None, poppler_path=None)

    maxPages = info["Pages"]

    # Convert PDF pages to images
    pdf_pages = []
    for page in range(1, maxPages+1, 10):
        pdf_pages += convert_from_path(args.input, dpi=200, first_page=page, last_page=min(page+10-1, maxPages))

    # Log PDF pages
    logger.info(f"PDF pages: {pdf_pages}")

    # Save cover image if not disabled
    if not args.no_cover:
        pdf_pages[0].save(f"{tempdir}/cover.jpg", "JPEG")
        pdf_pages.pop(0)

    for page_enumeration, page in enumerate(pdf_pages, start=1 if args.no_cover else 1):
        filename = f"page_{page_enumeration:03}.jpg"
        page.save(f"{tempdir}/{filename}", "JPEG")

        # Load image
        img = cv2.imread(f"{tempdir}/{filename}", cv2.IMREAD_GRAYSCALE)

        # Apply preprocessing steps
        blur = cv2.GaussianBlur(img, (5, 5), 2)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{tempdir}/{filename}", image)

        # Perform page dewarping
        subprocess.Popen(['page-dewarp', '-oscreen', '-d0', '-f 1.2', f'-x {args.x_margin}', f'-y {args.y_margin}', filename], cwd=tempdir).communicate()


def select_files():
    """
    Select image files for OCR.
    """
    directory = os.fsencode(tempdir)
    file_list = []

    # Iterate through files in temporary directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if 'cover' in filename:
            continue
        if filename.endswith(".png"):
            file_list.append(f"{tempdir}/{filename}")
            logger.info(f"Appending {filename}")
        else:
            processed_img = os.path.basename(filename).rsplit(".", 1)[0] + "_thresh.png"
            if exists(f"{tempdir}/{processed_img}"):
                continue
            logger.info(f"Appending {filename}")
            file_list.append(f"{tempdir}/{filename}")
    return file_list


def is_junk(args, blockid, results, current_chap_title, new_chap_thrs):
    """
    Determine if a block of text is junk.
    """
    blktext = ' '.join([results['text'][i] for i, r in enumerate(results['block_num']) if r == blockid])
    #logger.debug(f"is_junk(): blktext: {blktext}, current_chap_title: {current_chap_title}")
    ratio = SequenceMatcher(a=re.sub(' +', ' ', blktext.strip()), b=re.sub(' +', ' ', args.title.strip())).ratio()
    logger.debug(f"is_junk(): title ratio: {ratio}")
    is_title_header = ratio > 0.9 if args.title else False
    ratio = SequenceMatcher(a=re.sub(' +', ' ', blktext.strip()), b=re.sub(' +', ' ', current_chap_title.strip())).ratio()
    logger.debug(f"is_junk(): chapter ratio: {ratio}")
    is_chap_header = ratio > 0.9 and all([ results['top'][i] < new_chap_thrs for i, r in enumerate(results['block_num']) if r == blockid])
    cmp_set = {c for c in blktext if c != ' '}
    has_special_chars = False
    nb_special_chars = sum(c in cmp_set for c in r"<>&_()*+=\/{}#@¡¿†‡¶€¥ü∑∫¢@+-…äëïöüñßÄËÏÖÜÑ ")
    if nb_special_chars > 0:
        has_special_chars = nb_special_chars / len(cmp_set) >= 0.3
    not_conf_list = [results['conf'][i] < 80 for i, r in enumerate(results['block_num']) if
                     (r == blockid and results['level'][i] == 5)]
    not_conf = float(not_conf_list.count(True)) / float(len(not_conf_list)) > 0.6
    page_number = re.search('^ +[0-9]+ *$', blktext)
    stripped = blktext.strip()
    is_blank = stripped == ""
    logger.debug(f"is_junk(): has_special_chars: {has_special_chars} \t not_conf:{not_conf} \t page_nb:{page_number} \t is_blank: {is_blank} \t is_chap_header:{is_chap_header} \t is_title_header:{is_title_header}")
    return has_special_chars or not_conf or page_number or is_blank or is_chap_header or is_title_header


def make_block_text(blockid, results):
    """
    Create text from a block of text.
    """
    paragraphs = set([results['par_num'][i] for i, r in enumerate(results['block_num']) if r == blockid])
    textlines = []
    for par in paragraphs:
        lines = set([results['line_num'][i] for i, r in enumerate(results['block_num']) if
                     r == blockid and results['par_num'][i] == par])
        for line in lines:
            textlines.append(" ".join([results['text'][i] for i, r in enumerate(results['block_num']) if
                                       r == blockid and results['par_num'][i] == par and results['line_num'][i] == line])[
                              1:])
    return '\n'.join(textlines)


def make_ocr(args):
    """
    Perform OCR on images.
    """
    image_file_list = select_files()
    final_text = []
    chap_number = 1
    current_chapter = ""
         
    for image_file in image_file_list:
        image = Image.open(image_file)
        page_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        logger.info(f"Processing OCR on page {image_file}")
        
        if args.OCR == "easyOCR":
            reader = easyocr.Reader(['fr', 'en'])
            final_text = reader.readtext(page_gray, width_ths=0.7, ycenter_ths=0.5, height_ths=0.7, paragraph=True,
                                          detail=0)
        else:  # tesseract
            results = pytesseract.image_to_data(page_gray, lang=args.language, output_type=Output.DICT)
            results_df = pd.DataFrame.from_dict(results)
            page_width = results_df['width'][0]
            page_height = results_df['height'][0]
            new_chapter_thrs = page_height * args.chap_detect_thres_pct / 100
            logger.debug(f"Page width : {page_width} Page height: {page_height}")
            logger.debug(f"New chapter threshold: {new_chapter_thrs}")

            blocks_idx = [i for i, r in enumerate(results['level']) if r == 2]
            text_on_top = False
            page_text = ''

            # Need to detect all junks on the page
            junk = []
            for i in blocks_idx:
                if is_junk(args, results['block_num'][i], results, current_chapter, new_chapter_thrs):
                    logger.debug(f"Detected that block {results['block_num'][i]}: {results['text'][i]} is junk")
                    junk.append(results['block_num'][i])

            for i in blocks_idx:
                if results['block_num'][i] in junk:
                    continue

                (x, y, w, h) = (results['left'][i], results['top'][i], results['width'][i], results['height'][i])
                cv2.rectangle(page_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                logger.debug(f"block position : {(x, y)}")
                logger.debug(f"block dimension : {(w, h)}")
                
                
                # eventual page numbers
                if page_height - y < 70:
                    continue
                block_num = results['block_num'][i]
                block_text = ' '.join([results['text'][i] for i, r in enumerate(results['block_num']) if r == block_num]).strip()
                page_num = results['page_num'][i]
                full_page_text = ' '.join([results['text'][i] for i, r in enumerate(results['page_num']) if r == page_num and results['block_num'][i] not in junk]).strip()
                logger.debug(f"Block text: {block_text}")
                logger.debug(f"Page text: {full_page_text}")

                # New section detection
                if (page_width - w) > (page_width / 3) and y > (page_height / 5) and not text_on_top and block_text == full_page_text and len(block_text) < 50 and not "chapitre" in block_text.lower():
                    logger.debug(f"Detected a new section")
                    page_text += f'\n$$$ {full_page_text} $$$\n'
                    continue
                    
                
                # chapter detection
                logger.debug(f"no chap detection: {args.no_chap_detection}")
                logger.debug(f"text_on_top: {text_on_top}")
                if not args.no_chap_detection:
                    # the block must not be at top, not be near the end of page, but somewhere in
                    # the middle
                    if y > new_chapter_thrs and y < (0.75 * page_height) and not text_on_top:
                        logger.debug(f"Detected a new chapter")
                        text_on_top = True
                        chap_text = block_text if len(block_text) < 40 else 'Chapitre'
                        if block_text == full_page_text and len(block_text) < 40:
                            continue
                        logger.debug(f"Detecting that text is beginning in middle of the page")
                        if chap_text.lower() == "chapitre":
                            chap_text += f" {chap_number}"
                        page_text += f'\n@@@ {chap_text} @@@\n'
                        current_chapter = chap_text
                        chap_number += 1
                        
                        if len(block_text) < 40: # chapter detected and already inserted as such in page.
                            continue
                    elif y > new_chapter_thrs and not text_on_top:
                        text_on_top = True
                    else:
                        text_on_top = True
                page_text += make_block_text(block_num, results)
            
            if args.debug:
                # Downsize and maintain aspect ratio
                image2 = imutils.resize(page_gray, width=600)
                # After all, we will show the output image 
                cv2.imshow("Image", image2) 
                cv2.waitKey(0) 
            #logger.debug(f"appending : {page_text}")
            final_text.append(page_text)
        # end page marker
        final_text.append("§")

    #logger.info(f"Read text: {final_text}")
    final_text = post_process_text(args, final_text)
    # The recognized text is stored in variable text
    # Finally, write the processed text to the file.
    with open(args.input.rsplit(".", 1)[0] + f'_{args.OCR}.txt', "a", encoding='UTF-8') as output_file:
        output_file.write(final_text)
    return final_text


def post_process_text(args, text):
    """
    Perform post-processing on OCR text.
    """
    lang = language_tool_python.LanguageTool(args.language[:2])
    acc = "àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßØøÅåÆæœ"
    for i, page in enumerate(text):
        # double hyphens at begin of a line
        page = re.sub(r"\n( *)[\u2014-]+( *)", r"\n\1-\2", page)
        # simple hyphens
        page = re.sub(r"([a-zA-Z%s]+) *[\u2014-]+ *([a-zA-Z%s]+)" % (acc, acc), r"\1-\2", page)
        # word cut on two lines
        page = re.sub(r"([a-zA-Z%s]+) *[\u2014-] *\n+ *([a-zA-Z%s]+)" % (acc, acc), r"\1\2", page)
        # blank lines
        page = re.sub(r"\n\n+", "\n", page)
        # single quote character
        page = re.sub('‘', "’", page)
        # dialogs indent
        page = re.sub(r'\n ?([\u2014-]) ?([a-zA-Z%s])(.*)(\n|$)' % acc, r"\n    \1 \2\3\n", page)
        page = re.sub(r'\n ?([\u2014-]) ?([a-zA-Z%s])(.*)(\n|$)' % acc, r"\n    \1 \2\3\n", page)
        # lettrine
        page = re.sub(r'\n([a-zA-Z]) ', r'\n\1', page)

        if args.filter is not None:
            for filt in args.filter:
                page = re.sub("\n.*" + filt + ".*\n", "\n", page)
        # malformed Je
        page = page.replace(']e', 'Je')
        page = page.replace(']’', 'J’')
        # mispells and grammar
        #page = lang.correct(page)

        text[i] = page
    
    # blank lines between pages
    full_text = '\n'.join(text)
    # footnotes
    full_text = re.sub(r"(\n1\..*?§)", r'\n~~~\n\1\n~~~\n', full_text, flags=re.DOTALL) # § stands for textual end page marker
    # no need of end page markers anymore
    full_text = full_text.replace('§', '')
    # deletion of unnecessary blank lines
    full_text = re.sub(r"([a-zA-Z%s,;])\n+([a-zA-z%s])" % (acc, acc), r"\1 \2", full_text)
    full_text = full_text.replace("\n\n", "\n")
    return full_text


def create_epub(args, text):
    """
    Create EPUB file.
    """
    logger.info("Creating epub")
    book = epub.EpubBook()
    if not args.no_cover:
        book.set_cover(f"image.jpg", open(f'{tempdir}/cover.jpg', 'rb').read())
    # Add metadata
    if not args.title:
        title = args.input.rsplit(".", 1)[0]
    else:
        title = args.title
    book.set_title(title)
    book.set_language(args.language[0:-1])
    book.add_author(args.author)

    ct = datetime.now().timestamp()
    title_repl = title if len(title) <= 8 else title[0:8]
    book.set_identifier(f"{str(ct)}_{title_repl.replace(' ', '')}")

    sections = text.split('$$$')
    sections_dict = {}
    if sections[0] == '' or sections[0] == '\n':
        sections.pop(0)
    elif len(sections) % 2 and len(sections) > 2:
        sections_dict['0'] = sections.pop(0)
    only_chapters = len(sections) == 1
    chapters = []
    logger.debug(f"{len(sections)} sections : {sections}")
    for i in range(0, len(sections), 2):
        logger.debug(f"working on section {i}")
        if only_chapters:
            sect_key = '0'
            sect_value = sections[0]
        else:
            sect_key = sections[i]
            sect_value = sections[i + 1]
        sections_dict[sect_key] = sect_value
        
        # Split text into chapters
        chap_elem = sect_value.split('@@@')
        chap_dict = {}
        if re.search('\n+', chap_elem[0]):
            chap_elem.pop(0)
        elif len(sect_value) % 2:
            chap_dict['0'] = chap_elem.pop(0)
        logger.debug(f'{len(chap_elem)} chap_elem: {chap_elem}')
        for i in range(0, len(chap_elem), 2):
            chap_key = chap_elem[i]
            chap_value = chap_elem[i + 1]
            chap_dict[chap_key] = chap_value

        for k, v in chap_dict.items():
            chap = epub.EpubHtml(title=k, file_name=f"{k.replace(' ', '')}.xhtml", lang='hr')
            if not only_chapters:
                chap.content = f'<html><head></head><body><h1>{sect_key}</h1></br><h2>{k}</h2>'
            else:
                chap.content = f'<html><head></head><body><h1>{k}</h1>'
            for line in v.split('\n'):
                chap.content += f"<p>{line}</p>"
            chap.content += "</body></html>"
            logger.debug(f"appending chapter : {chap.content[0:15]}...")
            chapters.append(chap)

    # Add style
    style = '''BODY { text-align: justify;}'''

    default_css = epub.EpubItem(uid="style_default", file_name="style/default.css", media_type="text/css", content=style)
    book.add_item(default_css)

    # Add chapters to the book
    for chap in chapters:
        book.add_item(chap)
    
    # create table of contents
    # - add manual link
    # - add section
    # - add auto created links to chapters

    # book.toc = (epub.Link('intro.xhtml', 'Introduction', 'intro'),
    #             (epub.Section('Languages'),
    #              (c1, c2))
    #             )

    # add navigation files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # define css style
    style = '''
@namespace epub "http://www.idpf.org/2007/ops";

# body {
#     font-family: Cambria, Liberation Serif, Bitstream Vera Serif, Georgia, Times, Times New Roman, serif;
# }

# h2 {
#      text-align: left;
#      text-transform: uppercase;
#      font-weight: 200;     
# }

# ol {
#         list-style-type: none;
# }

# ol > li:first-child {
#         margin-top: 0.3em;
# }


# nav[epub|type~='toc'] > ol > li > ol  {
#     list-style-type:square;
# }


# nav[epub|type~='toc'] > ol > li > ol > li {
#         margin-top: 0.3em;
# }

# '''

    # Add CSS file
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    # Create spine
    book.spine = ['nav', *chapters]

    # Save EPUB file
    epub_file = f"{title}.epub"
    epub.write_epub(epub_file, book, {})

    logger.info(f"EPUB file created: {epub_file}")
    return epub_file


def main():
    """
    Main function.
    """
    start_time = time.time()
    args = parse_options()
    if not args.recognize_only and not args.generate_epub_only :
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)
        os.mkdir(tempdir)
        files = os.listdir()
        for f in files:
            if args.input.split(".")[0] in f and f.split('.')[1] != 'pdf':
                os.unlink(f)
                print(f"deleted {f}")
        convert_pdf(args)
    else:
        print("proceeding directly to the text recognition")
    
    if not args.generate_epub_only:
        text = make_ocr(args)
    else:
        text = ''
        with open(args.input.rsplit(".", 1)[0] + f'_{args.OCR}.txt', "rb") as input_file:
            text = input_file.read().decode("utf-8")
            logger.debug(text)
    epub_file = create_epub(args, text)
    shutil.move(epub_file, args.input.rsplit(".", 1)[0] + ".epub")
    
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration / 60)
    minutes_str = str(minutes) + 'm' if minutes > 0 else ''
    seconds = int(duration % 60)
    seconds_str = str(seconds) + 's' if seconds > 0 else ''
    logger.info(f'Done in {minutes_str}{seconds_str}!')    


if __name__ == "__main__":
    main()
