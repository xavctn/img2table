# img2table

`img2table` is a simple, easy to use, table identification and extraction Python Library based on [OpenCV](https://opencv.org/) image 
processing that supports most common image file formats as well as PDF files.

Thanks to its design, it provides a practical and lighter alternative to Neural Networks based solutions, especially for usage on CPU.

## Table of contents
* [Installation](#installation)
* [Features](#features)
* [Supported file formats](#supported-file-formats)
   * [Images](#images-formats)
   * [PDF](#pdf-formats)
* [Usage](#usage)
   * [Documents](#documents)
      * [Images](#images-doc)
      * [PDF](#pdf-doc)
   * [OCR](#ocr)
      * [Tesseract](#tesseract)
      * [PaddleOCR](#paddle)
      * [Google Vision](#vision)
      * [AWS Textract](#textract)
      * [Azure Cognitive Services](#azure)
   * [Table extraction](#table-extract)
   * [Excel export](#xlsx)
* [Examples](#examples)
* [Caveats / FYI](#fyi)


## Installation <a name="installation"></a>
The library can be installed via pip.
```python
# Standard installation, supporting Tesseract
pip install img2table

# For usage with Paddle OCR (Python <= 3.10 only)
pip install img2table[paddle]
# For usage with Paddle OCR - GPU (CUDA 9 / CUDA 10) (Python <= 3.10 only)
pip install img2table[paddle-gpu]

# For usage with Google Vision OCR
pip install img2table[gcp]

# For usage with AWS Textract OCR
pip install img2table[aws]

# For usage with Azure Cognitive Services OCR
pip install img2table[azure]
```

## Features <a name="features"></a>

* Table identification for images and PDF files, including bounding boxes at the table cell level
* Handling of complex table structures such as merged cells
* Handling of implicit rows - see [example](/examples/Implicit_rows.ipynb)
* Table content extraction by providing support for OCR services / tools
* Extracted tables are returned as a simple object, including a Pandas DataFrame representation
* Export extracted tables to an Excel file, preserving their original structure

## Supported file formats <a name="supported-file-formats"></a>

### Images <a name="images-formats"></a>

Images are loaded using the `opencv-python` library, supported formats are listed below.

<details>
<summary>Supported image formats</summary>
<br>

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
<cite><a href="https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56">OpenCV: Image file reading and writing</a></cite></li>
</ul>
</blockquote>
</details>
<br>
Multi-page images are not supported.

---

### PDF <a name="pdf-formats"></a>

Both native and scanned PDF files are supported.

## Usage <a name="usage"></a>

### Documents <a name="documents"></a>

#### Images <a name="images-doc"></a>
Images are instantiated as follows :
```python
from img2table.document import Image

image = Image(src, 
              detect_rotation=False)
```

> <h4>Parameters</h4>
><dl>
>    <dt>src : str, <code>pathlib.Path</code>, bytes or <code>io.BytesIO</code>, required</dt>
>    <dd style="font-style: italic;">Image source</dd>
>    <dt>detect_rotation : bool, optional, default <code>False</code></dt>
>    <dd style="font-style: italic;">Detect and correct skew/rotation of the image</dd>
></dl>
<br>
The implemented method to handle skewed/rotated images supports skew angles up to 45° and is
based on the publication by <a href="https://www.mdpi.com/2079-9292/9/1/55">Huang, 2020</a>.<br>
Setting the <code>detect_rotation</code> parameter to <code>True</code>, image coordinates and bounding boxes returned by other 
methods might not correspond to the original image.

#### PDF <a name="pdf-doc"></a>
PDF files are instantiated as follows :
```python
from img2table.document import PDF

pdf = PDF(src, pages=[0, 2])
```

> <h4>Parameters</h4>
><dl>
>    <dt>src : str, <code>pathlib.Path</code>, bytes or <code>io.BytesIO</code>, required</dt>
>    <dd style="font-style: italic;">PDF source</dd>
>    <dt>pages : list, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">List of PDF page indexes to be processed. If None, all pages are processed</dd>
></dl>

PDF pages are converted to images with a 200 DPI for table identification.

---

### OCR <a name="ocr"></a>

`img2table` provides an interface for several OCR services and tools in order to parse table content.<br>
If possible (i.e for searchable PDF), PDF text will be extracted directly from the file and the OCR service/tool will not be called.

<details>
<summary>Tesseract<a name="tesseract"></a></summary>
<br>

```python
from img2table.ocr import TesseractOCR

ocr = TesseractOCR(n_threads=1, 
                   lang="eng", 
                   psm=11,
                   tessdata_dir="...")
```

> <h4>Parameters</h4>
><dl>
>    <dt>n_threads : int, optional, default <code>1</code></dt>
>    <dd style="font-style: italic;">Number of concurrent threads used to call Tesseract</dd>
>    <dt>lang : str, optional, default <code>"eng"</code></dt>
>    <dd style="font-style: italic;">Lang parameter used in Tesseract for text extraction</dd>
>    <dt>psm : int, optional, default <code>11</code></dt>
>    <dd style="font-style: italic;">PSM parameter used in Tesseract, run <code>tesseract --help-psm</code> for details</dd>
>    <dt>tessdata_dir : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">Directory containing Tesseract traineddata files. If None, the <code>TESSDATA_PREFIX</code> env variable is used.</dd>
></dl>


*Usage of [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) requires prior installation. 
Check [documentation](https://tesseract-ocr.github.io/tessdoc/) for instructions.*
<br>
</details>

<details>
<summary>PaddleOCR<a name="paddle"></a></summary>
<br>

*Available for Python versions <= 3.10*

<a href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR</a> is an open-source OCR based on Deep Learning models.<br>
At first use, relevant languages models will be downloaded.

```python
from img2table.ocr import PaddleOCR

ocr = PaddleOCR(lang="en")
```

> <h4>Parameters</h4>
><dl>
>    <dt>lang : str, optional, default <code>"en"</code></dt>
>    <dd style="font-style: italic;">Lang parameter used in Paddle for text extraction, check <a href="https://github.com/Mushroomcat9998/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md#5-support-languages-and-abbreviations">documentation for available languages</a></dd>
></dl>

*Released in version 0.0.13*
<br>
</details>

<details>
<summary>Google Vision<a name="vision"></a></summary>
<br>

Authentication to GCP can be done by setting the standard `GOOGLE_APPLICATION_CREDENTIALS` environment variable.<br>
If this variable is missing, an API key should be provided via the `api_key` parameter.

```python
from img2table.ocr import VisionOCR

ocr = VisionOCR(api_key="api_key", timeout=15)
```

> <h4>Parameters</h4>
><dl>
>    <dt>api_key : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">Google Vision API key</dd>
>    <dt>timeout : int, optional, default <code>15</code></dt>
>    <dd style="font-style: italic;">API requests timeout, in seconds</dd>
></dl>
<br>
</details>

<details>
<summary>AWS Textract<a name="textract"></a></summary>
<br>

When using AWS Textract, the DetectDocumentText API is exclusively called.

Authentication to AWS can be done by passing credentials to the `TextractOCR` class.<br>
If credentials are not provided, authentication is done using environment variables or configuration files. 
Check `boto3` [documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) for more details.

```python
from img2table.ocr import TextractOCR

ocr = TextractOCR(aws_access_key_id="***",
                  aws_secret_access_key="***",
                  aws_session_token="***",
                  region="eu-west-1")
```

> <h4>Parameters</h4>
><dl>
>    <dt>aws_access_key_id : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">AWS access key id</dd>
>    <dt>aws_secret_access_key : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">AWS secret access key</dd>
>    <dt>aws_session_token : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">AWS temporary session token</dd>
>    <dt>region : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">AWS server region</dd>
></dl>
<br>
</details>

<details>
<summary>Azure Cognitive Services<a name="azure"></a></summary>
<br>

```python
from img2table.ocr import AzureOCR

ocr = AzureOCR(endpoint="abc.azure.com",
               subscription_key="***")
```

> <h4>Parameters</h4>
><dl>
>    <dt>endpoint : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">Azure Cognitive Services endpoint. If None, inferred from the <code>COMPUTER_VISION_ENDPOINT</code> environment variable.</dd>
>    <dt>subscription_key : str, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">Azure Cognitive Services subscription key. If None, inferred from the <code>COMPUTER_VISION_SUBSCRIPTION_KEY</code> environment variable.</dd>
></dl>
<br>
</details>

---

### Table extraction <a name="table-extract"></a>

Multiple tables can be extracted at once from a PDF page/ an image using the `extract_tables` method of a document.

```python
from img2table.ocr import TesseractOCR
from img2table.document import Image

# Instantiation of OCR
ocr = TesseractOCR(n_threads=1, lang="eng")

# Instantiation of document, either an image or a PDF
doc = Image(src)

# Table extraction
extracted_tables = doc.extract_tables(ocr=ocr,
                                      implicit_rows=True,
                                      borderless_tables=False,
                                      min_confidence=50)
```
> <h4>Parameters</h4>
><dl>
>    <dt>ocr : OCRInstance, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">OCR instance used to parse document text. If None, cells content will not be extracted</dd>
>    <dt>implicit_rows : bool, optional, default <code>True</code></dt>
>    <dd style="font-style: italic;">Boolean indicating if implicit rows should be identified - check related <a href="/examples/Implicit_rows.ipynb" target="_self">example</a></dd>
>    <dt>borderless_tables : bool, optional, default <code>False</code></dt>
>    <dd style="font-style: italic;">Boolean indicating if <a href="/examples/borderless.ipynb" target="_self">borderless tables</a> are extracted. It requires to provide an OCR to the method in order to be performed - <b>feature in alpha version</b></dd>
>    <dt>min_confidence : int, optional, default <code>50</code></dt>
>    <dd style="font-style: italic;">Minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)</dd>
></dl>

*Borderless table extraction released in version 0.0.14*

#### Method return

The [`ExtractedTable`](/src/img2table/tables/objects/extraction.py#L35) class is used to model extracted tables from documents.

> <h4>Attributes</h4>
><dl>
>    <dt>bbox : <code><a href="/src/img2table/tables/objects/extraction.py#L12" target="_self">BBox</a></code></dt>
>    <dd style="font-style: italic;">Table bounding box</dd>
>    <dt>title : str</dt>
>    <dd style="font-style: italic;">Extracted title of the table</dd>
>    <dt>content : <code>OrderedDict</code></dt>
>    <dd style="font-style: italic;">Dict with with row indexes as keys and list of <code><a href="/src/img2table/tables/objects/extraction.py#L20" target="_self">TableCell</a></code> objects as values</dd>
>    <dt>df : <code>pd.DataFrame</code></dt>
>    <dd style="font-style: italic;">Pandas DataFrame representation of the table</dd>
></dl>

<h5 style="color:grey">Images</h5>

`extract_tables` method from the `Image` class returns a list of `ExtractedTable` objects. 
```Python
output = [ExtractedTable(...), ExtractedTable(...), ...]
```

<h5 style="color:grey">PDF</h5>

`extract_tables` method from the `PDF` class returns an `OrderedDict` object with page indexes as keys and lists of `ExtractedTable` objects. 
```Python
output = {
    0: [ExtractedTable(...), ...],
    1: [],
    ...
    last_page: [ExtractedTable(...), ...]
}
```


### Excel export <a name="xlsx"></a>

Tables extracted from a document can be exported to a xlsx file. The resulting file is composed of one worksheet per extracted table.<br>
Method arguments are mostly common with the `extract_tables` method.

```python
from img2table.ocr import TesseractOCR
from img2table.document import Image

# Instantiation of OCR
ocr = TesseractOCR(n_threads=1, lang="eng")

# Instantiation of document, either an image or a PDF
doc = Image(src)

# Extraction of tables and creation of an xlsx file containing tables
doc.to_xlsx(dest=dest,
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=False,
            min_confidence=50)
```
> <h4>Parameters</h4>
><dl>
>    <dt>dest : str, <code>pathlib.Path</code> or <code>io.BytesIO</code>, required</dt>
>    <dd style="font-style: italic;">Destination for xlsx file</dd>
>    <dt>ocr : OCRInstance, optional, default <code>None</code></dt>
>    <dd style="font-style: italic;">OCR instance used to parse document text. If None, cells content will not be extracted</dd>
>    <dt>implicit_rows : bool, optional, default <code>True</code></dt>
>    <dd style="font-style: italic;">Boolean indicating if implicit rows should be identified - check related <a href="/examples/Implicit_rows.ipynb" target="_self">example</a></dd>
>    <dt>borderless_tables : bool, optional, default <code>False</code></dt>
>    <dd style="font-style: italic;">Boolean indicating if <a href="/examples/borderless.ipynb" target="_self">borderless tables</a> are extracted. It requires to provide an OCR to the method in order to be performed - <b>feature in alpha version</b></dd>
>    <dt>min_confidence : int, optional, default <code>50</code></dt>
>    <dd style="font-style: italic;">Minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)</dd>
></dl>
> <h4>Returns</h4>
> If a <code>io.BytesIO</code> buffer is passed as dest arg, it is returned containing xlsx data



## Examples <a name="examples"></a>

Several Jupyter notebooks with examples are available :
<ul>
<li>
<a href="/examples/Basic_usage.ipynb" target="_self">Basic usage</a>: generic library usage, including examples with images, PDF and OCRs
</li>
<li>
<a href="/examples/borderless.ipynb" target="_self">Borderless tables</a>: specific examples dedicated to the extraction of borderless tables
</li>
<li>
<a href="/examples/Implicit_rows.ipynb" target="_self">Implicit rows</a>: illustrated effect 
of the parameter <code>implicit_rows</code> of the <code>extract_tables</code> method
</li>
</ul>

## Caveats / FYI <a name="fyi"></a>

<ul>
<li>
For table extraction, results are highly dependent on OCR quality. By design, tables where no OCR data 
can be found are not returned.
</li>
<li>
The library is tailored for usage on documents with white/light background. 
Effectiveness can not be guaranteed on other type of documents. 
</li>
<li>
Borderless tables extraction is still in alpha stage and might be inconsistent on complex cases.
Improvements to the algorithm will be released in future versions.
</li>
</ul>
