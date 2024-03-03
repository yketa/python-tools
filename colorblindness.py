#! /usr/bin/env python

"""
This scripts aims at simulating various sorts of colorblindness by converting
images and documents to what they would like to someone with a moderate degree
of this condition.

Necessary packages:
- numpy,
- matplotlib,
- fitz,
- img2pdf,
- colorspacious.

Useful ressources:
- https://davidmathlogic.com/colorblind
- https://colorspacious.readthedocs.io/en/latest/tutorial.html
- https://doi.org/10.1109/TVCG.2009.113
- https://gist.github.com/mwaskom/b35f6ebc2d4b340b4f64a4e28e778486
"""

from numpy import array
from matplotlib.pyplot import imread, imsave
import fitz
from img2pdf import convert
from colorspacious import cspace_convert

from tempfile import TemporaryDirectory
from os.path import join

# CONVERSION CLASSES

def CVD(arr, type='deuteranomaly', severity=50):
    """
    Convert RGB array to simulate color vision deficiency (CVD).

    (see https://colorspacious.readthedocs.io/en/latest/tutorial.html)

    Parameters
    ----------
    arr : float or int array-like
        Input RGB image array.
    type : string
        CVD type. (default: deuteranomaly)
    severity : float
        Severity (between 0 and 100).

    Returns
    -------
    arrCVD : float Numpy array
        Output RGB image array.
    """

    cvd_space = {"name": "sRGB1+CVD", "cvd_type": type, "severity": severity}
    arrCVD = cspace_convert(array(arr)[:, :, :3], cvd_space, "sRGB1")

    # corrections to RGB
    arrCVD[arrCVD > 1] = 1
    arrCVD[arrCVD < 0] = 0

    return arrCVD

# SCRIPT

if __name__ == '__main__':

    # ARGUMENTS

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'FILE',
        help='Input file.',
        type=str)

    parser.add_argument(
        '-c', '--cvd-type',
        help='Color vision deficiency (CVD) type.',
        default='deuteranomaly',
        choices=['deuteranomaly', 'protanomaly', 'tritanomaly'],
        type=str)
    parser.add_argument(
        '-s', '--severity',
        help='Severity (between 0 and 100).',
        default=50,
        type=float)

    parser.add_argument(
        '-o', '--output',
        help='Output document name.',
        default='out',
        type=str)
    parser.add_argument(
        '-d', '--dpi',
        help='DPI of output file.',
        default=200,
        type=int)

    args = parser.parse_args()

    # IMAGES

    try:            # image file
        img = (array(imread(args.FILE)),)
        if (img[0] > 1).any(): img = (img[0]/255,)
    except OSError: # .pdf document
        with fitz.open(args.FILE) as doc:
            img = list(map(
                lambda n:
                    (lambda page:
                        (lambda pixmap: list(map(
                            lambda y: list(map(
                                lambda x: array(pixmap.pixel(x, y))/255,
                                range(pixmap.width))),
                            range(pixmap.height))))(
                        page.get_pixmap()))(
                    doc.load_page(n)),
                range(doc.page_count)))

    imgCVD = tuple( # conversion
        CVD(_, type=args.cvd_type, severity=args.severity) for _ in img)

    # SAVE

    if len(imgCVD) == 1:    # save image
        imsave('%s.jpg' % args.output, imgCVD[0], dpi=args.dpi)
    else:                   # save.pdf document
        with TemporaryDirectory() as tmp_dir:
            tmp_template = join(tmp_dir, '%04i.jpg')
            for index in range(len(imgCVD)):
                imsave(tmp_template % index, imgCVD[index],
                    dpi=args.dpi)
            with open('%s.pdf' % args.output, 'wb') as pdf_out:
                pdf_out.write(convert(
                    [tmp_template % index for index in range(len(imgCVD))]))

