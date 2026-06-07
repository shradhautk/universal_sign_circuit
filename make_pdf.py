"""
Generate a polished PDF for Anthropic interpretability scientist application.

Run in Terminal:
    cd ~/Desktop/universal_sign_circuit
    pip3 install reportlab --break-system-packages
    python3 make_pdf.py

Output: ~/Desktop/universal_sign_circuit/sign_circuit_findings.pdf
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = os.path.expanduser("~/Desktop/universal_sign_circuit/sign_circuit_findings.pdf")

def make_styles():
    styles = {}
    styles['title'] = ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18,
        leading=22, textColor=colors.HexColor('#1a1a2e'), spaceAfter=6)
    styles['subtitle'] = ParagraphStyle('subtitle', fontName='Helvetica', fontSize=11,
        leading=14, textColor=colors.HexColor('#444444'), spaceAfter=4)
    styles['callout'] = ParagraphStyle('callout', fontName='Helvetica-BoldOblique', fontSize=11,
        leading=16, textColor=colors.HexColor('#1a1a2e'), leftIndent=18, rightIndent=18,
        spaceAfter=10, spaceBefore=6, backColor=colors.HexColor('#f0f4ff'))
    styles['h1'] = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=13,
        leading=16, textColor=colors.HexColor('#1a1a2e'), spaceBefore=16, spaceAfter=6)
    styles['h2'] = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11,
        leading=14, textColor=colors.HexColor('#333333'), spaceBefore=10, spaceAfter=4)
    styles['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=10,
        leading=14, textColor=colors.HexColor('#222222'), spaceAfter=6, alignment=TA_JUSTIFY)
    styles['bullet'] = ParagraphStyle('bullet', fontName='Helvetica', fontSize=10,
        leading=14, textColor=colors.HexColor('#222222'), leftIndent=16, spaceAfter=3)
    styles['caption'] = ParagraphStyle('caption', fontName='Helvetica-Oblique', fontSize=9,
        leading=12, textColor=colors.HexColor('#666666'), spaceAfter=8, alignment=TA_CENTER)
    return styles

def make_table(data, col_widths):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,0), 9),
        ('ALIGN',        (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',     (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f7f9ff')]),
        ('GRID',         (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    return t

def build():
    s = make_styles()
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=0.9*inch, bottomMargin=0.9*inch,
        title="Universal Sign Decision Circuit in Large Language Models",
        author="Shradha")
    story = []

    # Header
    story.append(Paragraph("Universal Sign Decision Circuit in Large Language Models", s['title']))
    story.append(Paragraph("Mechanistic Interpretability Research", s['subtitle']))
    story.append(Paragraph("Models: Qwen2.5-72B | LLaMA-3.3-70B | Method: Teacher Forcing + Attention Weights + Logit Lens", s['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a1a2e'), spaceAfter=12))

    # Key Finding
    story.append(Paragraph("Key Finding", s['h1']))
    story.append(Paragraph(
        "Layers 76-80 in Qwen2.5-72B (L77-79 in LLaMA-3.3-70B) form a universal sign decision "
        "circuit for mathematical formula generation. This circuit fires in the same narrow layer "
        "window across four distinct mathematical domains, two model families, and two analysis "
        "methods — independent of recursion depth, problem size, or task structure.", s['callout']))
    story.append(Paragraph(
        "Sign errors in LLM mathematical reasoning are not random. They originate from a single "
        "localized mechanism that either correctly activates the minus pathway or defaults to plus "
        "via pretraining frequency bias. The error and the correct decision are made in the same "
        "place by the same mechanism.", s['body']))

    # Experiments
    story.append(Paragraph("Experiments", s['h1']))
    story.append(Paragraph(
        "Five experiments across four mathematical domains and two model families, "
        "using both attention weight analysis and the logit lens:", s['body']))
    exp_data = [
        ["Experiment", "Model", "Domain", "Method", "Decision Layer"],
        ["EXP1",       "LLaMA-3.3-70B", "5x5 cofactor expansion",          "Attention weights", "L78-79"],
        ["EXP1b",      "LLaMA-3.3-70B", "Integration by parts (12 levels)","Attention weights", "L77-79"],
        ["EXP2",       "Qwen2.5-72B",   "5x5 cofactor expansion",          "Attention weights", "L78-79"],
        ["EXP3",       "Qwen2.5-72B",   "4x4 cofactor expansion",          "Attention weights", "L77-78"],
        ["EXP3-Gauss", "Qwen2.5-72B",   "5x5 Gaussian elimination (RREF)", "Logit lens",        "L76-80"],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(exp_data, [0.9*inch, 1.5*inch, 2.1*inch, 1.4*inch, 1.0*inch]))
    story.append(Paragraph("Four domains. Two models. Two analysis methods. Same layer window every time.", s['caption']))

    # Methodology
    story.append(Paragraph("Methodology", s['h1']))
    story.append(Paragraph("Step 1 — Teacher Forcing", s['h2']))
    story.append(Paragraph(
        "Model responses to linear algebra problems are collected. For each response, the correct "
        "solution steps are injected via teacher forcing: the model is fed the correct prefix and "
        "its hidden states are captured at every sign token position. This isolates the sign "
        "decision mechanism from earlier reasoning errors — the question is not whether the model "
        "gets the answer right, but where and how it commits to a sign token.", s['body']))

    story.append(Paragraph("Step 2 — Attention Weight Analysis (EXP1-EXP3)", s['h2']))
    story.append(Paragraph(
        "At each sign token position p, attention weights from all heads at all layers are extracted. "
        "The analysis identifies which layers show the sharpest attention shifts correlated with sign "
        "polarity, revealing L76-L79 as the consistent decision window.", s['body']))

    story.append(Paragraph("Step 3 — Logit Lens (EXP3-Gauss)", s['h2']))
    story.append(Paragraph(
        "The hidden state h[p-1] is projected through the final LayerNorm and lm_head at each of "
        "the 80 layers to obtain per-layer token probabilities P(+) and P(-). This directly reveals "
        "when the model commits to a sign:", s['body']))
    story.append(Paragraph("- Before L66: both P(+) and P(-) are essentially zero", s['bullet']))
    story.append(Paragraph("- At L76-L80: the decision crystallizes, reaching P approaching 1.0", s['bullet']))
    story.append(Paragraph("- L76-L80 is the first and only place where sign probability becomes meaningful", s['bullet']))

    # Error taxonomy
    story.append(PageBreak())
    story.append(Paragraph("Sign Error Taxonomy", s['h1']))
    story.append(Paragraph(
        "Two mechanistically distinct failure modes were identified, both localized to L76-L80:", s['body']))

    story.append(Paragraph("Type A — Total Activation Failure (majority of errors)", s['h2']))
    story.append(Paragraph("- P(-) never rises above 1e-5 at any of the 80 layers", s['bullet']))
    story.append(Paragraph("- P(+) ramps from ~0 at L1-L66 to 0.9999+ at L76-L80", s['bullet']))
    story.append(Paragraph("- Matrix context failed to activate the subtraction pathway entirely", s['bullet']))
    story.append(Paragraph("- The model commits confidently to + in the same layer window as correct decisions", s['bullet']))

    story.append(Paragraph("Type B — Weak Activation + Override (minority of errors)", s['h2']))
    story.append(Paragraph("- L76: P(-) = 0.515, P(+) = 0.484 — circuit briefly preferred minus", s['bullet']))
    story.append(Paragraph("- L77-L80: P(+) climbs back to 0.783, P(-) = 0.217", s['bullet']))
    story.append(Paragraph("- Subtraction signal fires weakly at L76 but is overridden within the same window", s['bullet']))

    # Mechanistic interpretation
    story.append(Paragraph("Mechanistic Interpretation", s['h1']))
    story.append(Paragraph("Pretraining frequency bias", s['h2']))
    story.append(Paragraph(
        "+ appears more often than - in mathematical text. The circuit's function is to push against "
        "this prior when subtraction is required. When it fails (Type A) or fires weakly (Type B), "
        "the prior dominates and + is output.", s['body']))

    story.append(Paragraph("Depth-invariance", s['h2']))
    story.append(Paragraph(
        "Cofactor expansion has 3 recursion levels, integration by parts has 12, and Gaussian "
        "elimination is iterative with no fixed depth. The sign decision always crystallizes in "
        "the same layer window. The circuit encodes a general mathematical sign-token prior, "
        "not a task-specific mechanism.", s['body']))

    # Summary
    story.append(Paragraph("Summary", s['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#aaaaaa'), spaceAfter=8))
    story.append(Paragraph(
        "L76-L80 in Qwen2.5-72B (and L77-79 in LLaMA-3.3-70B) is the universal sign decision "
        "circuit for mathematical formula generation. It fires in the same layer window across "
        "cofactor expansion, integration by parts, and Gaussian elimination — regardless of "
        "recursion depth, problem size, or model family. Sign errors are not failures of a "
        "downstream correction mechanism; they are failures of this circuit to correctly activate "
        "the minus pathway, causing the model to default to + via pretraining frequency bias. "
        "The error and the correct decision are made in the same place by the same mechanism.", s['callout']))

    # Dataset note
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=6))
    story.append(Paragraph(
        "Model responses were generated using a curated linear algebra benchmark covering "
        "determinant, rank, nullity, and matrix operation problems across multiple matrix sizes. "
        "The benchmark and full dataset are part of ongoing research under separate publication. "
        "Raw model outputs and problem data are available for review upon request.", s['caption']))

    doc.build(story)
    print(f"Done. PDF saved to: {OUTPUT}")

if __name__ == "__main__":
    build()
