#! /usr/bin/python3

import shutil
import subprocess
from PIL import Image
import os,re
import easyocr, platform
import platform
from pytesseract import *
import cv2
import imutils
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from os.path import exists
import argparse

import language_tool_python
import spacy
import pypub

# Create a LanguageTool object for French
tool = language_tool_python.LanguageTool('fr')
nlp = spacy.load("fr_core_news_sm")


# Create a temporary directory to hold our temporary images.
tempdir="./tmp"
pd.options.display.max_rows = 10


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
        if platform.system() == "Windows":
            print("calling convert_from_path using poppler")
            pdf_pages = convert_from_path(args.input, 500)
            print(f"pdf pages : {pdf_pages}")
        else:
            pdf_pages = convert_from_path(args.input, 500)
        # Read in the PDF file at 500 DPI

        # Iterate through all the pages stored above
        for page_enumeration, page in enumerate(pdf_pages, start=1):
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
    print(f"bloc text found : {blktext}")
    has_special_chars =  True in [ c in blktext for c in """<>&_()*+=\/{}#@""" ] 
    not_conf_list = [ results['conf'][i] < 80 for i,r in enumerate(results['block_num']) if (r == blockid and results['level'][i] == 5)]
    not_conf = float(not_conf_list.count(True)) / float(len(not_conf_list)) > 0.6
    print (f"is_junk(): has_special_chars: {has_special_chars} \t not_conf:{not_conf}")
    return has_special_chars and not_conf
    
def make_ocr(args):
    """
    Part #2 - Recognizing text from the images using OCR
    """
    print("-----------------------------------------------")
    print("---          begin of ocr        --------------")
    print("-----------------------------------------------")

    image_file_list = select_files()
    final_text = []    
    
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
            new_chapter_thrs = page_height * 35 / 100 # if first bloc begins below, it is probably a new chapter
            print(f"new_chapter_theshold: {new_chapter_thrs}")

            blocks_idx = [i for i,r in enumerate(results['level']) if r == 2]
            new_chapter_detected = False
            
            for i in blocks_idx:
                if is_junk(args, results['block_num'][i], results):
                    print(f"detected that block {results['block_num'][i]} is junk, maybe the beginning of new chapter")
                    continue
                                
                (x, y, w, h) = (results['left'][i], results['top'][i], results['width'][i], results['height'][i])
                # draw green lines on boxes
                cv2.rectangle(page_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if y > new_chapter_thrs and not new_chapter_detected:
                    print(f"bloc pos : {y}, threshold:{new_chapter_thrs}: detected a new chapter")
                    new_chapter_detected = True
            
            if args.debug:
                # Downsize and maintain aspect ratio
                image2 = imutils.resize(page_gray, width=600)
                # After all, we will show the output image 
                cv2.imshow("Image", image2) 
                cv2.waitKey(0) 
            #text = [str(((pytesseract.image_to_string(image, lang=args.language))))]
        #print(f"read text: {text}")
        #post_process_text(args, text)


def post_process_text(args, text):
    lang = language_tool_python.LanguageTool(args.language[:2])
    with open(args.input.rsplit(".", 1)[0] + f'_{args.OCR}.txt', "a", encoding='UTF-8') as output_file:
        acc="àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßØøÅåÆæœ"
        for i, page in enumerate(text):
            # words hyphened to the following line
            page = re.sub(r"([a-zA-Z%s]) *[\u2014-] *\n+([a-zA-Z%s])"%(acc, acc),r"\1\2", page)
            # unnecessary new lines
            page = re.sub(r"([a-zA-Z%s,;])\n+([a-z%s])"%(acc, acc),r"\1 \2", page)
            #txt = re.sub(r"\n ?([‘’'])", r"    \1", txt)
            # blank lines
            #txt = re.sub(r"\n\n", "\n", txt)
            # isolated page numbers
            page = re.sub(r"\n[0-9]+\n", "", page)
            # bad quotes
            page = re.sub('‘', "’", page)
            # weird chars before hyphens or long dashes
            page = re.sub(r'[\.~,] ?([\u2014-])', r"\1", page)
            
            # need to play this twice convesations dashes at begin of line and indented.
            page = re.sub(r'\n ?([\u2014-]) ?([a-zA-Z%s])(.*)\n'% acc, r"\n    \1 \2\3\n", page)
            page = re.sub(r'\n ?([\u2014-]) ?([a-zA-Z%s])(.*)\n'% acc, r"\n    \1 \2\3\n", page)
            
            
            if args.filter is not None:
                for filt in args.filter:
                    page = re.sub("\n.*" + filt + ".*\n", "\n", page)
            
            # mispells and grammar
            page = lang.correct(page)

            # Update the element in the 'text' list with the modified string
            text[i] = page
            
        # The recognized text is stored in variable text
        # Finally, write the processed text to the file.
        output_file.write('\n'.join(text))

def create_epub(args):
    if not args.title:
        output_epub = pypub.Epub(args.input.rsplit(".", 1)[0])
    else:
        output_epub = pypub.Epub(args.title)
    
    
    

def main():
    args = parse_options()
    if not args.recognize_only :
        shutil.rmtree(tempdir)
        os.mkdir(tempdir)
        convert_pdf(args)
    else:
        print("proceeding directly to the text recognition")
    make_ocr(args)
    create_epub(args)
    


if __name__ == "__main__":
    main()


