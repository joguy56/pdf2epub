from cv2 import namedWindow

from page_dewarp.cli import ArgParser
from page_dewarp.image import WarpedImage
from page_dewarp.options import cfg
from page_dewarp.pdf import save_pdf

# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101


def dewarp(filename, x = 50, y = 20, focal = 1.2):
    
    if cfg.debug_lvl_opt.DEBUG_LEVEL > 0 and cfg.debug_out_opt.DEBUG_OUTPUT != "file":
        namedWindow("Dewarp")
        
    # overloaded options
    cfg.debug_out_opt.DEBUG_OUTPUT = 0
    cfg.camera_opts.FOCAL_LENGTH = focal
    cfg.image_opts.PAGE_MARGIN_X = x
    cfg.image_opts.PAGE_MARGIN_Y = y    

    outfiles = []
    
    processed_img = WarpedImage(filename)
    if processed_img.written:
        outfiles.append(processed_img.outfile)
        print(f"  wrote {processed_img.outfile}", end="\n\n")

    if cfg.pdf_opts.CONVERT_TO_PDF:
        save_pdf(outfiles)
