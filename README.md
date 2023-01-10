# img2table

`img2table` is a table identification and extraction Python Library based on [OpenCV](https://opencv.org/) image 
processing that supports most common image file formats as well as PDF files.

It also provides implementations for several OCR services and tools in order to parse table contents.

## Installation
```python
pip install img2table
```

## Supported file formats

### Images

Images are loaded using the `opencv-python` library, supported formats are listed below.

<blockquote>
<ul>
<li>Windows bitmaps - <em>.bmp, </em>.dib</li>
<li>JPEG files - <em>.jpeg, </em>.jpg, *.jpe</li>
<li>JPEG 2000 files - *.jp2</li>
<li>Portable Network Graphics - *.png</li>
<li>WebP - *.webp</li>
<li>Portable image format - <em>.pbm, </em>.pgm, <em>.ppm </em>.pxm, *.pnm</li>
<li>PFM files - *.pfm</li>
<li>Sun rasters - <em>.sr, </em>.ras</li>
<li>TIFF files - <em>.tiff, </em>.tif</li>
<li>OpenEXR Image files - *.exr</li>
<li>Radiance HDR - <em>.hdr, </em>.pic</li>
<li>Raster and Vector geospatial data supported by GDAL<br>
<cite><a href="https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56">OpenCV: Image file reading and writing</a></cite></li>
</ul>
</blockquote>

### PDF 

Searchable and non-searchable PDF files are supported.

## Usage

### Documents

#### Images
Images are instantiated as follows :
```python
from img2table.document import Image

image = Image(src, dpi=200)
```
>**src**: *file path, bytes or `io.BytesIO` object*<br>
>**dpi**: *estimated image dpi (default 200)*

#### PDF
PDF files are instantiated as follows :
```python
from img2table.document import PDF

pdf = PDF(src, dpi=300, pages=[0, 2])
```
>**src**: *file path, bytes or `io.BytesIO` object*<br>
>**dpi**: *dpi used for conversion of PDF pages to images (default 300)*<br>
>**pages**: *list of PDF page indexes to be processed (default `None`: all pages are processed)*

### OCR
`img2table` provides an interface for several OCR services and tools in order to parse table content.

#### Tesseract
Tesseract is instantiated as such :
```python
from img2table.ocr import TesseractOCR

ocr = TesseractOCR(n_threads=1, lang="eng")
```
>**n_threads**: *number of concurrent Tesseract threads (default 1)*<br>
>**lang**: *lang parameter used in Tesseract (default "eng")*<br>


*Usage of [Tesseract-OCR](https://tesseract-ocr.github.io/tessdoc/) requires prior installation. Check documentation for instructions relative to your platform.*
