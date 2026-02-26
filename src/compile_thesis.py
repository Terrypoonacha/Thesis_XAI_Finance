import os
import markdown
from xhtml2pdf import pisa
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PDF = PROJECT_ROOT / "Thesis_Final.pdf"

# Order of chapters
CHAPTERS = [
    "Chapter1_Introduction.md",
    "Chapter2_LiteratureReview.md",
    "Chapter3_Methodology.md",
    "Chapter4_CaseStudy.md",
    "Chapter5_Evaluation.md",
    "Appendix_Technical.md"
]

def compile_thesis():
    print("Compiling Thesis PDF...")
    
    # 1. Combine Markdown
    full_text = "# Balancing Innovation and Compliance: An Agentic XAI Framework\n\n"
    full_text += "**Master's Thesis**\n\n"
    full_text += "---\n\n"
    
    for chapter_file in CHAPTERS:
        path = REPORTS_DIR / chapter_file
        if path.exists():
            print(f"Adding {chapter_file}...")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Add page break before each chapter (CSS handling needed, but for now just separation)
                full_text += f"\n\n<div style='page-break-before: always;'></div>\n\n"
                full_text += content
        else:
            print(f"Warning: {chapter_file} not found.")

    # 2. Convert to HTML
    print("Converting to HTML...")
    html_content = markdown.markdown(full_text, extensions=['extra', 'codehilite'])
    
    # Add simple CSS for styling
    css_style = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 12pt; line-height: 1.5; }
        h1 { color: #2c3e50; font-size: 24pt; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-top: 20px; }
        h2 { color: #34495e; font-size: 18pt; margin-top: 15px; }
        h3 { color: #7f8c8d; font-size: 14pt; margin-top: 10px; }
        code { background-color: #f4f4f4; padding: 2px 5px; font-family: Courier New, monospace; }
        pre { background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }
        img { max-width: 100%; height: auto; }
        .page-break { page-break-before: always; }
    </style>
    """
    
    final_html = f"<html><head>{css_style}</head><body>{html_content}</body></html>"
    
    # 3. Write PDF
    print(f"Writing PDF to {OUTPUT_PDF}...")
    with open(OUTPUT_PDF, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
        
    if pisa_status.err:
        print("Error creating PDF")
    else:
        print("Thesis PDF compiled successfully!")

if __name__ == "__main__":
    compile_thesis()
