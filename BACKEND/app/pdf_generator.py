from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf(report_text, filename="research_report.pdf"):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    elements = []

    for line in report_text.split("\n"):

        if line.strip():

            paragraph = Paragraph(
                line,
                styles["BodyText"]
            )

            elements.append(paragraph)

            elements.append(
                Spacer(1, 12)
            )

    doc.build(elements)

    return filename
