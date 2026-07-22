#!/usr/bin/env python3
"""Convert final_project_report.md to PDF using markdown + xhtml2pdf."""

import os
import markdown
from xhtml2pdf import pisa

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(PROJECT_ROOT, "final_project_report.md")
PDF_PATH = os.path.join(PROJECT_ROOT, "final_project_report.pdf")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Read markdown
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

# Convert relative image paths to absolute paths
md_text = md_text.replace("](results/", f"]({RESULTS_DIR}/")

# Convert markdown to HTML
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "md_in_html"],
)

# Wrap in HTML document with print-ready CSS
html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: letter;
    margin: 0.75in 0.85in;
  }}
  body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.4;
    color: #111;
  }}
  h1 {{
    font-size: 16pt;
    text-align: center;
    margin-bottom: 4px;
  }}
  h2 {{
    font-size: 13pt;
    margin-top: 14pt;
    border-bottom: 1px solid #cccccc;
    padding-bottom: 2px;
  }}
  h3 {{
    font-size: 11.5pt;
    margin-top: 10pt;
  }}
  p {{
    text-align: justify;
    margin: 5px 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 9pt;
    margin: 8px 0;
  }}
  th, td {{
    border: 1px solid #666666;
    padding: 3px 5px;
    text-align: left;
  }}
  th {{
    background-color: #e8e8e8;
    font-weight: bold;
  }}
  img {{
    max-width: 480px;
    display: block;
    margin: 8px auto;
  }}
  code {{
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1px 3px;
  }}
  pre {{
    background-color: #f4f4f4;
    padding: 6px;
    font-size: 8.5pt;
    border: 1px solid #dddddd;
  }}
  hr {{
    border: none;
    border-top: 1px solid #cccccc;
    margin: 12px 0;
  }}
  ul, ol {{
    margin: 4px 0;
    padding-left: 20px;
  }}
  li {{
    margin: 2px 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Generate PDF
with open(PDF_PATH, "wb") as pdf_file:
    status = pisa.CreatePDF(html_doc, dest=pdf_file)

if status.err:
    print(f"Error generating PDF: {status.err}")
else:
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"PDF generated: {PDF_PATH}")
    print(f"Size: {size_kb:.0f} KB")
    
    # Quick verification: count pages (approximate from file size)
    if size_kb < 100:
        print("WARNING: PDF seems too small - images may not have rendered!")
    else:
        print("PDF appears to contain embedded images (size > 100 KB)")
