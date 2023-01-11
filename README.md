# img2table

`img2table` is a simple, easy to use, table identification and extraction Python Library based on [OpenCV](https://opencv.org/) image 
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

> <span style="color:grey; font-weight: bold">Parameters</span>
><dl>
>    <dt>src : str, <code>pathlib.Path</code>, bytes or <code>io.BytesIO</code>, required</dt>
>    <dd style="font-style: italic;">Image source</dd>
>    <dt>dpi : int, optional, default <code>200</code></dt>
>    <dd style="font-style: italic;">Estimated image dpi, used to adapt OpenCV algorithm parameters</dd>
></dl>

#### PDF
PDF files are instantiated as follows :
```python
from img2table.document import PDF

pdf = PDF(src, dpi=300, pages=[0, 2])
```

> <span style="color:grey; font-weight: bold">Parameters</span>
><dl>
>    <dt>src : str, <code>pathlib.Path</code>, bytes or <code>io.BytesIO</code>, required</dt>
>    <dd style="font-style: italic;">PDF source</dd>
>    <dt>dpi : int, optional, default <code>300</code></dt>
>    <dd style="font-style: italic;">Dpi used for conversion of PDF pages to images</dd>
>    <dt>pages : list, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">List of PDF page indexes to be processed. If None, all pages are processed</dd>
></dl>

### OCR
`img2table` provides an interface for several OCR services and tools in order to parse table content.

#### Tesseract
Tesseract is instantiated as such :
```python
from img2table.ocr import TesseractOCR

ocr = TesseractOCR(n_threads=1, lang="eng")
```

> <span style="color:grey; font-weight: bold">Parameters</span>
><dl>
>    <dt>n_threads : int, optional, default <code>1</code></dt>
>    <dd style="font-style: italic;">Number of concurrent Tesseract threads used</dd>
>    <dt>lang : str, optional, default <code>"eng"</code></dt>
>    <dd style="font-style: italic;">Lang parameter used in Tesseract for text extraction</dd>
></dl>


*Usage of [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) requires prior installation. 
Check [documentation](https://tesseract-ocr.github.io/tessdoc/) for instructions.*

### Table extraction

Multiple tables can be extracted at once from a PDF page/ an image using the `extract_tables` method of a document.

```python
from img2table.ocr import TesseractOCR
from img2table.document import Image

# Instantiation of OCR
ocr = TesseractOCR(n_threads=1, lang="eng")

# Instantiation of document, either an image or a PDF
doc = Image(src, dpi=200)

# Table extraction
extracted_tables = doc.extract_tables(ocr=ocr,
                                      implicit_rows=True,
                                      min_confidence=50)
```
> <span style="color:grey; font-weight: bold">Parameters</span>
><dl>
>    <dt>ocr : OCRInstance, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">OCR instance used to parse document text. If None, cells content will not be extracted</dd>
>    <dt>implicit_rows : bool, optional, default <code>True</code></dt>
>    <dd style="font-style: italic;">Boolean indicating if implicit rows should be identified - check related <a href="/examples/Implicit_rows.ipynb" target="_self">example</a></dd>
>    <dt>min_confidence : int, optional, default <code>50</code></dt>
>    <dd style="font-style: italic;">Minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)</dd>
></dl>

#### Method return

The [`ExtractedTable`](/src/img2table/tables/objects/extraction.py#L23) class is used to model extracted tables from documents.

> <span style="color:grey; font-weight: bold">Attributes</span>
><dl>
>    <dt>bbox : <code><a href="/src/img2table/tables/objects/extraction.py#L9" target="_self">BBox</a></code></dt>
>    <dd style="font-style: italic;">Table bounding box</dd>
>    <dt>title : str</dt>
>    <dd style="font-style: italic;">Extracted title of the table</dd>
>    <dt>content : <code>OrderedDict</code></dt>
>    <dd style="font-style: italic;">Dict with with row index as key and list of <code><a href="/src/img2table/tables/objects/extraction.py#L17" target="_self">TableCell</a></code> objects as values</dd>
>    <dt>df : <code>pd.DataFrame</code></dt>
>    <dd style="font-style: italic;">Pandas DataFrame representation of the table</dd>
></dl>


## Examples

Several Jupyter notebooks with examples are available :
<ul style="list-style-type: circle">
<li>
<a style="font-weight: bold" href="/examples/Image.ipynb" target="_self">Images</a>: library usage for images
</li>
<li>
<a style="²" href="/examples/PDF.ipynb" target="_self">PDF</a>: library usage for PDF files
</li>
<li>
<a style="font-weight: bold" href="/examples/Implicit_rows.ipynb" target="_self">Implicit rows</a>: illustrated effect 
of the parameter <code>implicit_rows</code> of the <code>extract_tables</code> method
</li>
</ul>

## FYI

<ul style="list-style-type: circle">
<li>
If possible (i.e for searchable PDF), PDF text will be extracted directly from the file and the OCR service/tool will not be called.
</li>
</ul>