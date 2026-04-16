import os
import io
import base64

from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)


def generate_image_description(pil_img) -> str:
    """
    Generate a detailed semantic description of an image for vector search.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            [
                "Describe this image in detail for semantic search. "
                "If the image shows card variants, product tiers, charts, "
                "labels, legends, or visual comparisons, mention them clearly.",
                pil_img,
            ]
        )

        return response.text.strip() if response.text else "Image description unavailable"

    except Exception as e:
        return f"Image description failed: {str(e)}"


def parse_document(file_path: str) -> list[dict]:
    """
    Parse a PDF into typed multimodal chunks using Docling.

    Each returned chunk is a dict:
      {
        content: semantic text (image description / table text / paragraph)
        content_type: "text" | "table" | "image"
        metadata: {section, page_number, source_file, position, ...}
        image_path: filepath for image chunks (None otherwise)
      }
    """

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    for item in doc.iterate_items():
        node = item[0] if isinstance(item, tuple) else item
        label = str(getattr(node, "label", "")).lower()

        if label in ("page_header", "page_footer"):
            continue

        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None

        position = None
        if prov and getattr(prov[0], "bbox", None):
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        def _meta(content_type: str, element_type: str):
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,
            }

        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append({
                    "content": text,
                    "content_type": "text",
                    "metadata": _meta("text", label),
                    "image_path": None,
                })

        elif "table" in label:
            table_text = ""

            if hasattr(node, "export_to_dataframe"):
                try:
                    df = node.export_to_dataframe()
                    if df is not None and not df.empty:
                        rows = []
                        headers = [str(c).strip() for c in df.columns]
                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {str(v).strip()}"
                                for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                rows.append(" | ".join(pairs))
                        table_text = "\n".join(rows)
                except Exception:
                    pass

            if not table_text:
                table_text = getattr(node, "text", "")

            if table_text.strip():
                parsed_chunks.append({
                    "content": table_text.strip(),
                    "content_type": "table",
                    "metadata": _meta("table", "table"),
                    "image_path": None,
                })

        elif "picture" in label or "figure" in label or label == "chart":
            caption = getattr(node, "text", "") or ""
            pil_img = None
            image_path = None
            description = None

            try:
                if hasattr(node, "get_image"):
                    pil_img = node.get_image(doc)
                elif hasattr(node, "image") and node.image:
                    pil_img = getattr(node.image, "pil_image", None)

                if pil_img:
                    filename = f"{source_file}_p{page_no}_img{len(parsed_chunks)}.png"
                    image_path = os.path.join(IMAGE_DIR, filename)
                    pil_img.save(image_path, format="PNG")

                    description = generate_image_description(pil_img)

            except Exception as e:
                description = f"Image processing failed: {str(e)}"

            content = (
                description
                or caption.strip()
                or f"[Image on page {page_no}]"
            )

            parsed_chunks.append({
                "content": content,          
                "content_type": "image",
                "metadata": _meta("image", "picture"),
                "image_path": image_path,    
            })

        else:
            text = getattr(node, "text", "")
            if text and text.strip():
                parsed_chunks.append({
                    "content": text.strip(),
                    "content_type": "text",
                    "metadata": _meta("text", label),
                    "image_path": None,
                })

    return parsed_chunks