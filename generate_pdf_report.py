#!/usr/bin/env python3
"""
Generate PDF version of the project report.
Converts PROJECT_REPORT_FINAL.md → PROJECT_REPORT_FINAL.pdf
Uses markdown + weasyprint for high-quality PDF output.
"""

import re
import os
import base64
from pathlib import Path

def load_image_as_base64(img_path: Path) -> str | None:
    """Load image file and return base64 data URI."""
    if not img_path.exists():
        return None
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = img_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"


def preprocess_markdown(text: str, base_dir: Path) -> str:
    """
    Pre-process markdown before conversion:
    - Embed images as base64
    - Convert LaTeX math to readable text/HTML
    - Strip YAML frontmatter
    """

    # Remove YAML frontmatter (between --- ... ---)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # Convert display math blocks $$...$$ to styled div
    def replace_display_math(m):
        formula = m.group(1).strip()
        # Clean up LaTeX to readable text
        formula = formula.replace(r"\frac{1}{2}", "½")
        formula = formula.replace(r"\frac{1}{n}", "1/n")
        formula = formula.replace(r"\sum_{i}", "Σᵢ")
        formula = formula.replace(r"\sum_{i=1}^{n}", "Σᵢ₌₁ⁿ")
        formula = formula.replace(r"\sum(y_i - \hat{y}_i)^2", "Σ(yᵢ - ŷᵢ)²")
        formula = formula.replace(r"\sum(y_i - \bar{y})^2", "Σ(yᵢ - ȳ)²")
        formula = formula.replace(r"\|w\|^2", "∥w∥²")
        formula = formula.replace(r"\xi_i + \xi_i^*", "ξᵢ + ξᵢ*")
        formula = formula.replace(r"\xi_i", "ξᵢ")
        formula = formula.replace(r"\xi_i^*", "ξᵢ*")
        formula = formula.replace(r"\varepsilon", "ε")
        formula = formula.replace(r"\phi(x_i)", "φ(xᵢ)")
        formula = formula.replace(r"\exp", "exp")
        formula = formula.replace(r"\gamma", "γ")
        formula = formula.replace(r"\|x_i - x_j\|^2", "∥xᵢ - xⱼ∥²")
        formula = formula.replace(r"V_k", "V_k")
        formula = formula.replace(r"\cdot", "·")
        formula = formula.replace(r"\hat{y}_i", "ŷᵢ")
        formula = formula.replace(r"\bar{y}", "ȳ")
        formula = formula.replace(r"y_i", "yᵢ")
        formula = formula.replace(r"R^2", "R²")
        formula = formula.replace(r"TP_k", "TPₖ")
        formula = formula.replace(r"FN_k", "FNₖ")
        formula = formula.replace(r"\frac{1}{K}", "1/K")
        formula = formula.replace(r"\frac{TP_k}{TP_k + FN_k}", "TPₖ/(TPₖ + FNₖ)")
        formula = formula.replace(r"\sum_{k=1}^{K}", "Σₖ₌₁ᴷ")
        formula = formula.replace("{", "").replace("}", "")
        formula = formula.replace("\\\\", " ")
        return f'<div class="math-block">{formula}</div>'

    text = re.sub(r"\$\$(.*?)\$\$", replace_display_math, text, flags=re.DOTALL)

    # Convert inline math $...$ to styled span
    def replace_inline_math(m):
        formula = m.group(1)
        formula = formula.replace(r"\hat{y}", "ŷ").replace(r"\bar{y}", "ȳ")
        formula = formula.replace("^2", "²").replace("_i", "ᵢ").replace("_k", "ₖ")
        return f'<span class="math-inline">{formula}</span>'

    text = re.sub(r"\$([^$\n]+)\$", replace_inline_math, text)

    # Embed images
    def replace_image(m):
        alt = m.group(1)
        src = m.group(2)
        img_path = base_dir / src
        b64 = load_image_as_base64(img_path)
        if b64:
            return f'<figure><img src="{b64}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'
        else:
            return f'<p><em>[Abbildung nicht gefunden: {src}]</em></p>'

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, text)

    return text


def build_html(md_content: str) -> str:
    """Convert markdown to full HTML document."""
    import markdown as md_lib

    extensions = ["tables", "fenced_code", "attr_list", "toc"]
    try:
        html_body = md_lib.markdown(
            md_content,
            extensions=extensions,
        )
    except Exception as e:
        print(f"Warning: Markdown conversion issue: {e}")
        html_body = f"<pre>{md_content}</pre>"

    css = """
    @page {
        size: A4;
        margin: 2.5cm 2.5cm 2.5cm 2.5cm;
        @bottom-center {
            content: counter(page) " / " counter(pages);
            font-size: 9pt;
            color: #666;
        }
    }

    body {
        font-family: "Linux Libertine", "Georgia", "Times New Roman", serif;
        font-size: 10.5pt;
        line-height: 1.55;
        color: #1a1a1a;
        max-width: 100%;
    }

    h1 {
        font-size: 16pt;
        font-weight: bold;
        color: #1a237e;
        border-bottom: 2px solid #1a237e;
        padding-bottom: 4pt;
        margin-top: 24pt;
        margin-bottom: 10pt;
    }

    h2 {
        font-size: 13pt;
        font-weight: bold;
        color: #283593;
        border-bottom: 1px solid #c5cae9;
        padding-bottom: 2pt;
        margin-top: 18pt;
        margin-bottom: 8pt;
    }

    h3 {
        font-size: 11.5pt;
        font-weight: bold;
        color: #3949ab;
        margin-top: 14pt;
        margin-bottom: 6pt;
    }

    h4 {
        font-size: 10.5pt;
        font-weight: bold;
        font-style: italic;
        color: #3f51b5;
        margin-top: 10pt;
        margin-bottom: 4pt;
    }

    p {
        margin-bottom: 8pt;
        text-align: justify;
        hyphens: auto;
        orphans: 3;
        widows: 3;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10pt 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }

    th {
        background-color: #e8eaf6;
        color: #1a237e;
        font-weight: bold;
        padding: 5pt 8pt;
        border: 1px solid #9fa8da;
        text-align: left;
    }

    td {
        padding: 4pt 8pt;
        border: 1px solid #c5cae9;
    }

    tr:nth-child(even) td {
        background-color: #f8f9ff;
    }

    tr:nth-child(odd) td {
        background-color: #ffffff;
    }

    code {
        font-family: "Courier New", "DejaVu Sans Mono", monospace;
        font-size: 9pt;
        background-color: #f5f5f5;
        padding: 1pt 3pt;
        border-radius: 2pt;
        color: #c62828;
    }

    pre {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-left: 3px solid #3f51b5;
        padding: 8pt 12pt;
        font-size: 8.5pt;
        line-height: 1.4;
        overflow-x: auto;
        page-break-inside: avoid;
    }

    pre code {
        background: none;
        padding: 0;
        color: #1a1a1a;
    }

    blockquote {
        border-left: 3px solid #9fa8da;
        padding: 6pt 12pt;
        margin: 8pt 0;
        color: #444;
        background-color: #f8f9ff;
        font-style: italic;
    }

    ul, ol {
        margin: 8pt 0;
        padding-left: 20pt;
    }

    li {
        margin-bottom: 3pt;
        line-height: 1.5;
    }

    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 8pt auto;
    }

    figure {
        text-align: center;
        margin: 14pt 0;
        page-break-inside: avoid;
    }

    figcaption {
        font-size: 9pt;
        color: #555;
        font-style: italic;
        margin-top: 6pt;
        text-align: center;
    }

    .math-block {
        font-family: "Linux Libertine", "Georgia", serif;
        font-style: italic;
        text-align: center;
        background-color: #f8f9ff;
        border: 1px solid #e8eaf6;
        padding: 8pt 12pt;
        margin: 10pt 40pt;
        border-radius: 4pt;
        font-size: 10pt;
    }

    .math-inline {
        font-style: italic;
        font-family: "Linux Libertine", "Georgia", serif;
    }

    hr {
        border: none;
        border-top: 1px solid #c5cae9;
        margin: 16pt 0;
    }

    /* Cover page styling */
    .cover-title {
        font-size: 20pt;
        font-weight: bold;
        color: #1a237e;
        text-align: center;
        margin-bottom: 10pt;
    }

    .cover-subtitle {
        font-size: 13pt;
        color: #283593;
        text-align: center;
        font-style: italic;
    }

    strong {
        font-weight: bold;
    }

    em {
        font-style: italic;
    }

    a {
        color: #1565c0;
    }

    @media print {
        h1, h2 { page-break-after: avoid; }
        table, figure, pre { page-break-inside: avoid; }
    }
    """

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Projektbericht – Nelson Pinheiro – NIR Spektroskopie und ML</title>
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""


def generate_pdf(md_path: Path, pdf_path: Path):
    """Main function: markdown → HTML → PDF."""
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        print("ERROR: weasyprint not installed. Run: pip install weasyprint")
        return False

    print(f"Reading: {md_path}")
    md_text = md_path.read_text(encoding="utf-8")

    print("Pre-processing markdown (embedding images, converting math)...")
    md_processed = preprocess_markdown(md_text, md_path.parent)

    print("Converting markdown to HTML...")
    html_content = build_html(md_processed)

    # Save HTML for inspection
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML saved: {html_path}")

    print(f"Generating PDF: {pdf_path}")
    font_config = FontConfiguration()
    html_doc = HTML(string=html_content, base_url=str(md_path.parent))
    html_doc.write_pdf(
        pdf_path,
        font_config=font_config,
        presentational_hints=True,
    )
    print(f"✓ PDF saved: {pdf_path}")
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"  File size: {size_mb:.2f} MB")
    return True


if __name__ == "__main__":
    base = Path(__file__).parent
    md_file = base / "PROJECT_REPORT_FINAL.md"
    pdf_file = base / "PROJECT_REPORT_FINAL.pdf"

    success = generate_pdf(md_file, pdf_file)
    if success:
        print("\n✓ Report generation complete!")
        print(f"  Markdown: {md_file}")
        print(f"  PDF:      {pdf_file}")
    else:
        print("\n✗ PDF generation failed")
