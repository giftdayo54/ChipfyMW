"""PDF district report. Only supported metrics are listed; unavailable ones say so."""
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from . import config, adaptation

def build_pdf(district: str, analysis_date: str, metrics: list, sources: list, limitations: list,
              recommendations: dict) -> bytes:
    """metrics: list of (name, display_string, source_or_valid_time)."""
    st = getSampleStyleSheet(); buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"M-CHIP report - {district}")
    f = [Paragraph(f"M-CHIP District Hazard Report: {district}", st["Title"]),
         Paragraph(f"Analysis date: {analysis_date}", st["Normal"]), Spacer(1, 8),
         Paragraph(config.DISCLAIMER, st["Italic"]), Spacer(1, 10),
         Paragraph("Indicators", st["Heading2"]),
         Table([["Indicator", "Value", "Source / validity"]] + [list(map(str, m)) for m in metrics])]
    for title, rows in (("Data sources", sources), ("Limitations", limitations)):
        f += [Spacer(1, 10), Paragraph(title, st["Heading2"])] + [Paragraph(f"- {r}", st["Normal"]) for r in rows]
    f += [Spacer(1, 10), Paragraph("Suggested actions", st["Heading2"]),
          Paragraph(recommendations["notice"], st["Italic"])]
    f += [Paragraph(f"- [{k}] {v}", st["Normal"]) for k, v in recommendations["items"]]
    doc.build(f)
    return buf.getvalue()
