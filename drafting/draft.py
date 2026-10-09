
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer,Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_RIGHT

doc = SimpleDocTemplate("lawyer_document.pdf")

styles = getSampleStyleSheet()

story = []
'''
story.append(Paragraph("LEGAL DOCUMENT", styles["Title"]))

story.append(Spacer(1, 20))

story.append(Paragraph(
    "This is a sample legal document prepared by an advocate.",
    styles["Normal"]
))
story.append(Spacer(100,20))

story.append(Paragraph(
    "this is for the demo"
))
'''

data = [[
    Paragraph("place",styles["Normal"]),
    Paragraph("name",styles["Normal"])
]]
table = Table(data,colWidths = [300,100])

table.setStyle([
    ("ALIGN",(0,0),(0,0),"LEFT"),
    ("ALIGN",(1,0),(1,0),"RIGHT")
])

story.append(table)

doc.build(story)
