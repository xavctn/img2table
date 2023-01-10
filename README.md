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
<li>Windows bitmaps - <em>.bmp, </em>.dib (always supported)</li>
<li>JPEG files - <em>.jpeg, </em>.jpg, *.jpe (see the Note section)</li>
<li>JPEG 2000 files - *.jp2 (see the Note section)</li>
<li>Portable Network Graphics - *.png (see the Note section)</li>
<li>WebP - *.webp (see the Note section)</li>
<li>Portable image format - <em>.pbm, </em>.pgm, <em>.ppm </em>.pxm, *.pnm (always supported)</li>
<li>PFM files - *.pfm (see the Note section)</li>
<li>Sun rasters - <em>.sr, </em>.ras (always supported)</li>
<li>TIFF files - <em>.tiff, </em>.tif (see the Note section)</li>
<li>OpenEXR Image files - *.exr (see the Note section)</li>
<li>Radiance HDR - <em>.hdr, </em>.pic (always supported)</li>
<li>Raster and Vector geospatial data supported by GDAL (see the Note section)<br>
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

image = Image(src, dpi=300)
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
>**pages**: *list of PDF page indexes to be processed (default None: all pages are processed)*
