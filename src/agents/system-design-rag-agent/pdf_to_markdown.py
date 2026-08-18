import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from typing import Optional, Union

from pydantic import BaseModel, ValidationError
import pymupdf
import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from docling.datamodel.settings import settings as docling_settings
docling_settings.inference.compile_torch_models = False

from docling.datamodel.document import ConversionResult
from docling_core.types.doc.common.annotations import DescriptionAnnotation
from docling_core.types.doc.document import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.items.node import NodeItem


@contextmanager
def time_perf(task_name="Task"):
    print(f"[{task_name}] starting...")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{task_name}] took {end - start:.4f} seconds")


_CODE_BREAK_BEFORE_RE = re.compile(
    # !!! Only works for Python or SQL code !!!
    # keywords only count when a colon follows within a reasonable distance
    # otherwise they're just as likely to be ordinary English words (a SQL comment saying "else" or "while").
    # "def" has a negative lookbehind for "async" so "async def" breaks as one unit
    # instead of the plain "def" branch also matching the space inside it.
    r"(?<!async)[ \t]+(?=def\b(?=[^\n]{0,60}[(:]))"
    r"|[ \t]+(?=async\s+def\b(?=[^\n]{0,60}[(:]))"
    r"|[ \t]+(?=class\b(?=[^\n]{0,60}[(:]))"
    r"|[ \t]+(?=(?:elif|else|except|finally|if|while|for|try|with)\b(?=[^\n]{0,80}:))"
    r"|[ \t]+(?=(?:import|from|return|raise|break|continue|pass)\b)"
    r"|[ \t]+(?=#)"
    r"|[ \t]+(?=--)"
    r"|[ \t]+(?=(?:SELECT|UPDATE|INSERT|DELETE|BEGIN|COMMIT|ROLLBACK|CREATE|WITH|ALTER|DROP|TABLE)\b)"
)
_SENTENCE_END_RE = re.compile(r"[.!?:;\"')\]}]$")


def _get_image_link(pdf_stem: str, image_hash: str):
    return f"images/{pdf_stem}/{image_hash}.png"


_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
_OLLAMA_VLM_MODEL = "qwen2.5vl:7b"
_IMAGES_DIR = Path("data/images")
_ANALYSIS_CONFIDENCE_THRESHOLD = 0.5


def _crop_picture(pdf_file_path: Path, item: PictureItem) -> Optional[bytes]:
    if not item.prov:
        return None

    prov = item.prov[0]
    with pymupdf.open(str(pdf_file_path)) as doc:
        page = doc[prov.page_no - 1]
        bbox = prov.bbox.to_top_left_origin(page_height=page.rect.height)
        rect = pymupdf.Rect(bbox.l, bbox.t, bbox.r, bbox.b)
        pix = page.get_pixmap(clip=rect, dpi=150)
        
        return pix.tobytes("png")


def _gather_node_item_context(document: DoclingDocument, item: PictureItem) -> str:
    """
    Gather surrounding text in the same parent to make context for the given docling's PictureItem.
    """
    if item.parent is None:
        return ""

    parent = item.parent.resolve(document)
    texts = [
        sibling.text.strip()
        for ref in parent.children
        if isinstance(sibling := ref.resolve(document), TextItem)
        and sibling.text.strip()
    ]
    return "\n".join(texts)


class VlmPictureAnalysisResult(BaseModel):
    title: str
    caption: str
    confidence: float
    

def _describe_image_vlm(image_bytes: bytes, context: str, pdf_stem: str) -> Optional[tuple[str, str, str]]:
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a PDF expert. Your goal is to analyze an image cropped from a PDF document, and write a short "
        "title and a 1-2 paragraph caption for it. The caption should describe the key visual elements inside the image"
        "and how they connect to each other - not just the theme.\n\n"
        "Also score your confidence, from 0.0 to 1.0, that this image is a meaningful diagram "
        "or chart that adds real information beyond the surrounding text - as opposed to a "
        "decorative icon, logo, stock photo, or purely illustrative graphic with no real content.\n\n"
        f"Here's surrounding document context:\n{context}\n\n"
        "Respond only in English, with no other language mixed in.\n\n"
        'Respond with only valid JSON of the form {"title": "...", "caption": "...", "confidence": 0.0}.'
    )

    try:
        response = requests.post(
            _OLLAMA_URL,
            json={
                "model": _OLLAMA_VLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        raw_result = json.loads(response.json()["choices"][0]["message"]["content"])
        analysis_result = VlmPictureAnalysisResult.model_validate(raw_result)
        title = analysis_result.title
        caption = analysis_result.caption
        confidence = float(analysis_result.confidence)
    except (json.JSONDecodeError, ValidationError):
        return None

    if confidence < _ANALYSIS_CONFIDENCE_THRESHOLD or not title:
        return None

    image_dir = _IMAGES_DIR / pdf_stem
    image_path = image_dir / f"{image_hash}.png"
    if not image_path.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

    return title, caption, image_hash


VLM_ENRICHED_PICTURE_PROVENANCES = {
    "title": "vlm_enriched_picture_title",
    "caption": "vlm_enriched_picture_caption",
    "hash": "vlm_enriched_picture_hash",
}

def _enrich_pictures(document: DoclingDocument, pdf_file_path: Path) -> None:
    pictures = [item for item, _ in document.iterate_items() if isinstance(item, PictureItem)]
    pdf_stem = pdf_file_path.stem

    if not pictures:
        print(f"[pictures] {pdf_file_path.name}: no pictures found, skipping VLM enrichment")
        return

    print(f"[pictures] {pdf_file_path.name}: captioning {len(pictures)} picture(s) via {_OLLAMA_VLM_MODEL}...")

    def _describe(item: PictureItem) -> tuple[PictureItem, Optional[tuple[str, str, str]]]:
        crop = _crop_picture(pdf_file_path, item)
        if crop is None:
            return item, None
        return item, _describe_image_vlm(crop, _gather_node_item_context(document, item), pdf_stem)

    enriched = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (item, result) in enumerate(pool.map(_describe, pictures), start=1):
            if result is None:
                print(f"[pictures] ({i}/{len(pictures)}) skipped - no crop or below confidence threshold")
                continue
            title, caption, image_hash = result
            item.annotations.append(DescriptionAnnotation(text=title, provenance=VLM_ENRICHED_PICTURE_PROVENANCES["title"]))
            item.annotations.append(DescriptionAnnotation(text=caption, provenance=VLM_ENRICHED_PICTURE_PROVENANCES["caption"]))
            item.annotations.append(DescriptionAnnotation(text=image_hash, provenance=VLM_ENRICHED_PICTURE_PROVENANCES["hash"]))
            enriched += 1
            print(f"[pictures] ({i}/{len(pictures)}) captioned: {title}")

    print(f"[pictures] {pdf_file_path.name}: {enriched}/{len(pictures)} picture(s) captioned")
    

def _normalize_code_block(code: str) -> str:
    """
    Flatten single-line code block back into multiple lines. Does NOT reconstruct indentation.
    """
    code = re.sub(r";[ \t]*(?!$)", ";\n", code)
    code = _CODE_BREAK_BEFORE_RE.sub("\n", code)
    return code
    

def _render_markdown(document: DoclingDocument, source: Path) -> str:
    blocks: list[str] = []
    prev_was_list_item = False
    prev_item: NodeItem | None = None
    code_accumulator: list[str] = []
    image_count = 0
    table_count = 0
    pdf_stem = source.stem

    has_title_item = any(isinstance(item, TitleItem) for item, _ in document.iterate_items())
    heading_floor = 2 if has_title_item else 1

    for item, _level in document.iterate_items():
        rendered: Optional[str] = None
        is_list_item = False

        # We strip the interactive "Test Your Knowledge" quiz followed by a "Quick
        # Reference" CTA card. It's always the last thing in the document, so
        # once we hit it we're done.
        item_text = getattr(item, "text", None)
        if item_text and item_text.strip().lower().startswith("test your knowledge"):
            break

        if isinstance(item, TitleItem):
            text = item.text.strip()
            rendered = f"# {text}" if text else None
                
        elif isinstance(item, SectionHeaderItem):
            text = item.text.strip()
            if text:
                level = min(max(item.level, heading_floor), 6)
                rendered = f"{'#' * level} {text}"
                
        elif isinstance(item, ListItem):
            text = item.text.strip()
            if text:
                marker = (item.marker or "-").strip() or "-"
                rendered = f"{marker} {text}"
                is_list_item = True
                
        elif isinstance(item, TableItem):
            # TODO: Add reclassification for tables. docling can sometimes YAML as tables. 
            # Can look into table summarization to improve chunking for larger and complex tables
            table_md = item.export_to_markdown(doc=document).strip()
            rendered = table_md if table_md else None
                
        elif isinstance(item, CodeItem):
            code = item.text.strip()
            if code:
                same_parent = (
                    isinstance(prev_item, CodeItem)
                    and item.parent is not None
                    and prev_item.parent is not None
                    and item.parent.cref == prev_item.parent.cref
                )

                continues = same_parent and len(code_accumulator) > 0

                if continues:
                    code_accumulator.append(code)
                    joined = " ".join(code_accumulator)
                    blocks[-1] = f"```\n{_normalize_code_block(joined)}\n```"
                else:
                    code_accumulator = [code]
                    rendered = f"```\n{_normalize_code_block(code)}\n```"
        
        
        elif isinstance(item, PictureItem):
            title = next(
                (
                    getattr(a, "text", None)
                    for a in item.annotations
                    if getattr(a, "provenance", None) == VLM_ENRICHED_PICTURE_PROVENANCES["title"] and getattr(a, "text", None)
                ),
                None,
            )
            caption = next(
                (
                    getattr(a, "text", None)
                    for a in item.annotations
                    if getattr(a, "provenance", None) == VLM_ENRICHED_PICTURE_PROVENANCES["caption"] and getattr(a, "text", None)
                ),
                None,
            )
            image_hash = next(
                (
                    getattr(a, "text", None)
                    for a in item.annotations
                    if getattr(a, "provenance", None) == VLM_ENRICHED_PICTURE_PROVENANCES["hash"] and getattr(a, "text", None)
                ),
                None,
            )

            if title and caption and image_hash:
                image_count += 1
                rendered = f"![{title}]({_get_image_link(pdf_stem=pdf_stem, image_hash=image_hash)})\n**Figure {image_count}: {caption.strip()}**"
            else:
                rendered = None
        
        elif isinstance(item, TextItem):
            if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                continue
                        
            text = item.text.strip()
            if text:      
                prev_sibling_was_picture = (
                    prev_item
                    and isinstance(prev_item, PictureItem)
                    and item.parent is not None
                    and (
                        # docling sometimes nests a diagram's stray label as a
                        # child of the PictureItem itself, not as a sibling
                        # under the picture's own parent - check both.
                        item.parent.cref == prev_item.self_ref
                        or (prev_item.parent is not None and item.parent.cref == prev_item.parent.cref)
                    )
                    and not _SENTENCE_END_RE.search(text) # if text fragment
                )
                if prev_sibling_was_picture:
                    continue
                
                print(item)
                
                if item.formatting and item.formatting.bold:
                    text = f"**{text}**"
                elif item.formatting and item.formatting.italic:
                    text = f"*{text}*"
                if item.hyperlink:
                    text = f"[{text}]({item.hyperlink})"
                
                same_parent = (
                    isinstance(prev_item, TextItem) 
                    and prev_item.label in (DocItemLabel.TEXT)
                    and item.parent is not None
                    and prev_item.parent is not None
                    and item.parent.cref == prev_item.parent.cref
                )
                
                continues = (
                    same_parent
                    and blocks
                    and not _SENTENCE_END_RE.search(blocks[-1])
                )   
                if continues:
                    blocks[-1] = f"{blocks[-1]} {text}"
                    rendered = None
                else:
                    rendered = text
        
        if rendered is None:
            continue
        
        prev_item = item

        if blocks:
            blocks.append("\n" if (is_list_item and prev_was_list_item) else "\n\n")
            
        blocks.append(rendered)
        prev_was_list_item = is_list_item

    return "".join(blocks)



def _matching_font_cells(prov, page) -> list[tuple[object, object]]:
    """
    Match a text item's bbox back to the raw page cells it was built from.
    """
    if page.size is None:
        return []
    
    matches = []
    for cell in page.cells:
        if not (hasattr(cell, "font_name") and cell.font_name):
            continue
        
        bbox = cell.rect.to_bounding_box().to_bottom_left_origin(page_height=page.size.height)
        if prov.bbox.intersection_area_with(bbox) > 0:
            matches.append((cell, bbox))
            
    return matches


def _resolve_heading_levels(result: ConversionResult) -> None:
    """
    docling's layout model correctly identifies SectionHeaderItem nodes but
    assigns every one the same flat level (always 1).

    We derive real levels from glyph size instead of text heuristics:
    match each heading's bbox back to the raw page cells
    and take the median matched-cell height as a font-size proxy. Larger text is a
    higher-level (more senior) heading. Heights are rounded to the nearest
    0.5pt before clustering so near-identical glyph heights collapse into one
    size bucket instead of each heading getting its own level; distinct
    buckets are then rank-mapped to levels 1..N, so however many distinct
    heading sizes the document actually uses, they nest correctly - no
    hardcoded assumption about how many heading levels exist.

    Levels assigned here are structural input for DoclingDocument._hierarchize(),
    which reparents each heading/its following content under the correct ancestor 
    based on these levels.
    """
    doc = result.document
    heading_items = [item for item, _ in doc.iterate_items() if isinstance(item, SectionHeaderItem)]
    if not heading_items:
        return

    heights_by_id: dict[int, float] = {}
    for item in heading_items:
        if not item.prov:
            continue
        
        prov = item.prov[0]
        page_no = prov.page_no
        if page_no - 1 >= len(result.pages):
            continue
        
        page = result.pages[page_no - 1]

        heights = [bbox.height for _, bbox in _matching_font_cells(prov, page)]
        if heights:
            heights_by_id[id(item)] = round(statistics.median(heights) * 2) / 2

    if not heights_by_id:
        return

    distinct_sizes = sorted(set(heights_by_id.values()), reverse=True)
    level_by_size = {size: rank + 1 for rank, size in enumerate(distinct_sizes)}
    smallest_level = len(distinct_sizes)

    for item in heading_items:
        size = heights_by_id.get(id(item))
        item.level = level_by_size[size] if size is not None else smallest_level
    
    # Let docling's own tree-structuring methods reparent everything under the right ancestor
    # and fix up self_ref/children indices afterward.
    result.document._hierarchize()
    result.document._normalize_references()


def _convert_pdf_to_docling(
    pdf_file_path: Path, 
    do_picture_enrichment: Optional[bool] = False, 
) -> DoclingDocument:
    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = False
    pipeline_options.generate_parsed_pages = True
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    # Needed for figuring heading levels 
    pipeline_options.generate_parsed_pages = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    print(f"[docling] converting {pdf_file_path.name}...")
    result = converter.convert(source=pdf_file_path)
    print(f"[docling] converted {pdf_file_path.name}: {len(result.pages)} page(s)")

    _resolve_heading_levels(result)
    
    if do_picture_enrichment:
        _enrich_pictures(result.document, pdf_file_path)
        
    return result.document


def convert_pdf_to_markdown(pdf_file_path: Path) -> str:
    with time_perf("convert_pdf_to_markdown"):
        document = _convert_pdf_to_docling(pdf_file_path)
        
        print(f"[docling] rendering markdown for {pdf_file_path.name}...")
        return _render_markdown(document, pdf_file_path)
            
