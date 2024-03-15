#! /usr/bin/python3
import os,re
import time
import shutil
import subprocess
from PIL import Image
import easyocr, platform
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

# Create a LanguageTool object for French
tool = language_tool_python.LanguageTool('fr')
nlp = spacy.load("fr_core_news_sm")


# Create a temporary directory to hold our temporary images.
tempdir="./tmp"
pd.options.display.max_rows = None


#function to resize image in order to append to other images using cv2
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def parse_options():
    # Créez un objet ArgumentParser
    parser = argparse.ArgumentParser(description="Programme de test prenant deux noms de fichiers en ligne de commande.")

    # Ajoutez les arguments optionnels
    parser.add_argument("-i", "--input", help="Nom du fichier d'entrée", required=True)
    parser.add_argument("-O", "--OCR", help="The OCR to use easyOCR|tesseract", required=False, default="tesseract")
    parser.add_argument("-r", "--recognize-only", help="starts directly to the recognition step", action="store_true", required=False)
    parser.add_argument("-f", "--filter", help="some lines that should be filtered", action="append", required=False)
    parser.add_argument("-l", "--language", help="french (fra - default) or english (eng)", required=False, default='fra')
    parser.add_argument("-a", "--author", help="the author of the book", required=False)
    parser.add_argument("-t", "--title", help="the book's titel", required=False)
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true", required=False, default=False)
    parser.add_argument("--no-chap-detection", help="disable chapter detection (enabled by default)", action="store_true", required=False, default=False)
    parser.add_argument("--chap-detect-thres-pct", help="percentage of page height from which a new chapter is beginning", required=False, default=35, type=int)
    parser.add_argument("--no-cover", help="no cover image for this ebook", action='store_true', required=False, default=False)

    # Analysez les arguments de la ligne de commande
    args = parser.parse_args()

    # Affichez les noms de fichiers
    print(f"options: {args}")
    return args


def convert_pdf(args):
        """
        Part #1 : Converting PDF to images
        """
        print (f"created {tempdir} directory")
        
        info = pdfinfo_from_path(args.input, userpw=None, poppler_path=None)

        maxPages = info["Pages"]
        pdf_pages = []
        if platform.system() == "Windows":
            print("calling convert_from_path using poppler")
            for page in range(1, maxPages+1, 10) : 
                pdf_pages += convert_from_path(args.input, dpi=200, first_page=page, last_page = min(page+10-1,maxPages))
            print(f"pdf pages : {pdf_pages}")
        else:
            for page in range(1, maxPages+1, 10) : 
                pdf_pages += convert_from_path(args.input, dpi=200, first_page=page, last_page = min(page+10-1,maxPages))
            print(f"pdf pages : {pdf_pages}")
        # Read in the PDF file at 500 DPI

        if not args.no_cover:
            pdf_pages[0].save(f"{tempdir}\cover.jpg", "JPEG")
            pdf_pages.pop(0)
        # Iterate through all the pages stored above
        
        for page_enumeration, page in enumerate(pdf_pages, start=1 if args.no_cover else 1):
            print(f"creating image for page {page}")
            filename = f"page_{page_enumeration:03}.jpg"
            page.save(f"{tempdir}\{filename}", "JPEG")
            #page.save(f"{tempdir}\original_{filename}", "JPEG")
            
            # Charger l'image 
            img = cv2.imread(f"{tempdir}\{filename}", cv2.IMREAD_GRAYSCALE)

            # Filtre gaussien
            blur = cv2.GaussianBlur(img, (5, 5), 2)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert NumPy array to OpenCV image
            image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # Convert to BGR (optional)

            # Save the image
            cv2.imwrite(f"{tempdir}/{filename}", image)  # Replace "{tempdir}" with your path

            subprocess.Popen(['page-dewarp', '-oscreen', '-d0', '-f 2.0', filename], cwd=tempdir).communicate()


def select_files():
    directory = os.fsencode(tempdir)
    file_list = []
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if 'cover' in filename:
            continue
        if filename.endswith(".png"):
            file_list.append(f"{tempdir}/{filename}")
            print(f"appending {filename}")
        else:
            processed_img=os.path.basename(filename).rsplit(".", 1)[0] + "_thresh.png"
            if exists(f"{tempdir}\{processed_img}"):
                continue
            print(f"appending {filename}")
            file_list.append(f"{tempdir}/{filename}")
    return file_list

def is_junk(args, blockid, results):
    blktext = ' '.join([results['text'][i] for i,r in enumerate(results['block_num']) if r == blockid])
    #print(f"bloc text found : {blktext}")
    # too much special chars and low confidence 
    has_special_chars =  True in [ c in blktext for c in """<>&_()*+=\/{}#@""" ] 
    not_conf_list = [ results['conf'][i] < 80 for i,r in enumerate(results['block_num']) if (r == blockid and results['level'][i] == 5)]
    not_conf = float(not_conf_list.count(True)) / float(len(not_conf_list)) > 0.6
    # page numbers
    page_number = re.search('^ +[0-9]+ *$', blktext)
    # blank blocks
    stripped = blktext.strip()
    is_blank = stripped == ""
    
    print (f"is_junk(): has_special_chars: {has_special_chars} \t not_conf:{not_conf} \t page_nb:{page_number} \t is_blank: {is_blank}")
    #return (has_special_chars and not_conf) or page_number or is_blank
    return not_conf or page_number or is_blank

def make_block_text(blockid, results):
    paragraphs = set([results['par_num'][i] for i, r in enumerate(results['block_num']) if r == blockid])
    textlines = []
    for par in paragraphs:
        lines = set([results['line_num'][i] for i, r in enumerate(results['block_num']) if r == blockid and results['par_num'][i] == par])
        for line in lines:
            textlines.append(" ".join([results['text'][i] for i, r in enumerate(results['block_num']) if r == blockid and results['par_num'][i] == par and results['line_num'][i] == line])[1:])
    #print(f"returning : {textlines}")
    return '\n'.join(textlines)
        
    
def make_ocr(args):
    """
    Part #2 - Recognizing text from the images using OCR
    """
    print("-----------------------------------------------")
    print("---          begin of ocr        --------------")
    print("-----------------------------------------------")

    image_file_list = select_files()
    final_text = [] 
    chap_number = 1
    # Open the file in append mode so that
    # All contents of all images are added to the same file

    # Iterate from 1 to total number of pages
    for image_file in image_file_list:

        # Set filename to recognize text from
        # Again, these files will be:
        # page_1.jpg
        # page_2.jpg
        # ....
        # page_n.jpg
        # transfer image of pdf_file into array
        image = Image.open(image_file)
        page_arr = np.asarray(image)
        page_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        
        print(f"processing OCR on page {image_file}")
        if args.OCR == "easyOCR":
            reader = easyocr.Reader(['fr','en']) # this needs to run only once to load the model into memory
            final_text = reader.readtext(page_gray, width_ths=0.7, ycenter_ths=0.5, height_ths=0.7, paragraph=True, detail=0)
        else: # tesseract
            os.environ['TESSDATA_PREFIX'] = "C:/Users/jguyot/AppData/Local/Programs/Tesseract-OCR/tessdata"
            
            results = pytesseract.image_to_data(page_gray, lang=args.language, output_type=Output.DICT)
            results_df = pd.DataFrame.from_dict(results)
            print(results_df)
            page_width = results_df['width'][0]
            page_height = results_df['height'][0]
            print(f"page_width : {page_width} \t page_height:{page_height}")
            new_chapter_thrs = page_height * args.chap_detect_thres_pct / 100 # if first bloc begins below, it is probably a new chapter
            print(f"new_chapter_theshold: {new_chapter_thrs}")

            blocks_idx = [i for i,r in enumerate(results['level']) if r == 2]
            text_on_top = False
            page_text=''
            
            for i in blocks_idx:
                if is_junk(args, results['block_num'][i], results):
                    print(f"detected that block {results['block_num'][i]} is junk")
                    continue
                                
                (x, y, w, h) = (results['left'][i], results['top'][i], results['width'][i], results['height'][i])
                # draw green lines on boxes
                cv2.rectangle(page_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # eventual page numbers
                if page_height - y < 70:
                    continue
                block_num = results['block_num'][i]
                block_text = ' '.join([ results['text'][i] for i, r in enumerate(results['block_num']) if r == block_num]).strip()
                print(f"block_text: {block_text}")
                
                if not args.no_chap_detection:
                    print(f"bloc pos : {y}, threshold:{new_chapter_thrs}: text_on_top: {text_on_top}")
                    if y > new_chapter_thrs and not text_on_top:
                        print(f"bloc pos : {y}, threshold:{new_chapter_thrs}: detected a new chapter")
                        text_on_top = True
                        if 'chap' in block_text or 'Chap' in block_text:
                            page_text += f'\n@@@ {block_text} @@@\n'
                        else:
                            page_text += f'\n@@@ Chapitre {block_text} @@@\n'
                        chap_number += 1
                        continue
                    elif y <= new_chapter_thrs and not text_on_top:
                        text_on_top = True
                page_text += make_block_text(block_num, results)
            if args.debug:
                # Downsize and maintain aspect ratio
                image2 = imutils.resize(page_gray, width=600)
                # After all, we will show the output image 
                cv2.imshow("Image", image2) 
                cv2.waitKey(0) 
            final_text.append(page_text)
    print(f"read text: {final_text}")
    final_text = post_process_text(args, final_text)
    # The recognized text is stored in variable text
    # Finally, write the processed text to the file.
    with open(args.input.rsplit(".", 1)[0] + f'_{args.OCR}.txt', "a", encoding='UTF-8') as output_file:
        output_file.write(final_text)
    return final_text

def post_process_text(args, text):
    lang = language_tool_python.LanguageTool(args.language[:2])
    acc="àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßØøÅåÆæœ"
    for i, page in enumerate(text):
        # words hyphened to the following line
        page = re.sub(r"([a-zA-Z%s]) *[\u2014-] *\n *([a-zA-Z%s])"%(acc, acc),r"\1\2", page)
        #page = re.sub(r"\n ?([‘’'])", r"    \1", page)
        # blank lines
        page = re.sub(r"\n\n+", "\n", page)
        # isolated page numbers
        #page = re.sub(r"\n[0-9]+\n", "", page)
        # bad quotes
        page = re.sub('‘', "’", page)
        
        # need to play this twice conversations dashes at begin of line and indented.
        page = re.sub(r'\n ?([\u2014-]) ?([a-zA-Z%s])(.*)(\n|$)'% acc, r"\n    \1 \2\3\n", page)
        page = re.sub(r'\n ?([\u2014-]) ?([a-zA-Z%s])(.*)(\n|$)'% acc, r"\n    \1 \2\3\n", page)
        
        
        if args.filter is not None:
            for filt in args.filter:
                page = re.sub("\n.*" + filt + ".*\n", "\n", page)
        
        # mispells and grammar
        #page = lang.correct(page)

        # Update the element in the 'text' list with the modified string
        text[i] = page
        
    # blank lines between pages
    full_text ='\n'.join(text)
    full_text = re.sub(r"([a-zA-Z%s,;])\n *([a-zA-z%s])"%(acc, acc),r"\1 \2", full_text)
    return full_text

def create_epub(args, text):
    print("creating epub")
    book = epub.EpubBook()
    
    if not args.no_cover:
        book.set_cover(f"image.jpg", open(f'{tempdir}\cover.jpg', 'rb').read())
    
    # add metadata
    if not args.title:
        title = args.input.rsplit(".", 1)[0]
    else:
        title = args.title
    book.set_title(title)
    book.set_language(args.language[0:-1])
    book.add_author(args.author)
    
    ct = datetime.now().timestamp()
    book.set_identifier(f"{str(ct)}_{title.replace(' ', ' ')[0-8]}")

    # Step 1: Split the string using the token '@@@'
    elements = text.split('@@@')
    # Step 2: Create a dictionary from the elements
    result_dict = {}
    for i in range(1, len(elements), 2):
        key = elements[i]
        value = elements[i + 1]
        result_dict[key] = value

    # for every chapter
    chapters = []
    for k,v in result_dict.items():
        chap = epub.EpubHtml(title=k, file_name=f"{k.replace(' ', '')}.xhtml", lang='hr')
        chap.content = f'<html><head></head><body><h1>{k}</h1>'
        for line in v.split('\n'):
            chap.content += f"<p>{line}</p>"
        chap.content += "</body></html>"
        chapters.append(chap)

    # defube style
    style = '''BODY { text-align: justify;}'''

    default_css = epub.EpubItem(uid="style_default", file_name="style/default.css", media_type="text/css", content=style)
    book.add_item(default_css)


    # add chapters to the book
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

    # add css file
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    # create spine
    book.spine = ['nav' ] + [ c for c in chapters ]

    # create epub file
    epub_filename = args.input.split('.')[0] + '.epub'
    if os.path.isfile(epub_filename):
        print(f"file {epub_filename} already exists, deleting it")
        os.remove(epub_filename)
    epub.write_epub(epub_filename, book, {})



    
    

def main():
    start_time = time.time()
    args = parse_options()
    if not args.recognize_only :
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
    text = make_ocr(args)
    create_epub(args, text)
    end_time = time.time()
    
    duration = end_time - start_time
    minutes = int(duration / 60)
    minutes_str = str(minutes) + 'm' if minutes > 0 else ''
    seconds = int(duration % 60)
    seconds_str = str(seconds) + 's' if seconds > 0 else ''
    print(f'done in {minutes_str}{seconds_str}')


if __name__ == "__main__":
    main()


