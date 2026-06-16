# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""  # v4
Behavioral Portfolio Optimizer — Streamlit Dashboard
Full version with: live market data, manual input, CSV upload,
custom structured product composer, and extended optimizer (5+ securities).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.graph_objects as go

def generate_pdf_report(constraint_label, nd_res, dr_res, p3_return, p3_std,
                         nd_labels, nd_weights, nd_colors,
                         dr_labels, dr_weights, dr_colors,
                         der_label_sel, H_val, _alpha, use_es, _L,
                         data_mode, names_in, grid_lbl, lam_summary,
                         p0_stats=None, p0_labels=None, p0_weights=None, p0_colors=None,
                         fig_plotly=None, fig_png=None):
    """Generate a PDF summary report of optimisation results."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, Image as RLImage,
                                     PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
    from reportlab.platypus.frames import Frame
    import io, datetime

    # ── Page number canvas callback ───────────────────────────────────────────
    class _NumberedDoc(SimpleDocTemplate):
        def handle_pageEnd(self):
            self.canv.saveState()
            self.canv.setFont('Helvetica', 8)
            self.canv.setFillColor(colors.grey)
            self.canv.drawCentredString(
                A4[0] / 2, 1.2*cm,
                f"Page {self.canv.getPageNumber()} — Beyond Mean-Variance Portfolio Optimiser | Jeddou (2026)"
            )
            self.canv.restoreState()
            super().handle_pageEnd()

    buf = io.BytesIO()
    doc = _NumberedDoc(buf, pagesize=A4,
                       leftMargin=2*cm, rightMargin=2*cm,
                       topMargin=2*cm, bottomMargin=2.5*cm)

    styles = getSampleStyleSheet()
    navy   = colors.HexColor('#0d1a2e')
    blue   = colors.HexColor('#1a6bbf')
    green  = colors.HexColor('#10b981')
    gold   = colors.HexColor('#f59e0b')
    coral  = colors.HexColor('#e76f51')
    light  = colors.HexColor('#c0c8d8')
    white  = colors.white

    title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                  textColor=navy, fontSize=16, spaceAfter=4)
    h1_style    = ParagraphStyle('H1', parent=styles['Heading1'],
                                  textColor=blue, fontSize=12, spaceAfter=4)
    h2_style    = ParagraphStyle('H2', parent=styles['Heading2'],
                                  textColor=navy, fontSize=10, spaceAfter=2)
    body_style  = ParagraphStyle('Body2', parent=styles['Normal'],
                                  fontSize=9, spaceAfter=3)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
                                    fontSize=8, textColor=colors.grey, spaceAfter=2)

    def section_header(text, color=blue):
        return [
            Paragraph(_pdf_safe(text), ParagraphStyle('SH', parent=styles['Heading2'],
                       textColor=white, backColor=color, fontSize=10,
                       spaceBefore=8, spaceAfter=4, leftIndent=-6, rightIndent=-6,
                       borderPadding=(4,6,4,6))),
        ]

    def weights_table(labels, weights, colors_list, title):
        from reportlab.platypus import Flowable
        from reportlab.lib.units import cm as _cm

        class BarFlowable(Flowable):
            def __init__(self, fraction, bar_color, width=5*cm, height=0.35*cm):
                Flowable.__init__(self)
                self.fraction = max(0.0, min(1.0, fraction))
                self.bar_color = bar_color
                self.width = width
                self.height = height
            def draw(self):
                # Background track
                self.canv.setFillColor(colors.HexColor('#e9ecef'))
                self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
                # Filled portion
                if self.fraction > 0:
                    self.canv.setFillColor(self.bar_color)
                    self.canv.rect(0, 0, self.width * self.fraction, self.height, fill=1, stroke=0)

        # Header row
        header = [Paragraph(f'<b>{_pdf_safe(title)}</b>', ParagraphStyle('h', parent=styles["Normal"], fontSize=8, textColor=white)),
                  Paragraph('<b>Weight</b>', ParagraphStyle('h', parent=styles["Normal"], fontSize=8, textColor=white)),
                  Paragraph('<b>Allocation</b>', ParagraphStyle('h', parent=styles["Normal"], fontSize=8, textColor=white))]
        rows = [header]

        _bar_colors = [colors.HexColor('#e63946'), colors.HexColor('#f4a261'),
                       colors.HexColor('#e9c46a'), colors.HexColor('#2a9d8f'),
                       colors.HexColor('#264653'), colors.HexColor('#023e8a'),
                       colors.HexColor('#e76f51'), colors.HexColor('#457b9d')]

        for i, (lbl, w) in enumerate(zip(labels, weights)):
            _wf = float(w) / 100.0 if float(w) > 1.0 else float(w)
            _wf = max(0.0, min(1.0, _wf))
            _col = _bar_colors[i % len(_bar_colors)]
            rows.append([
                Paragraph(_pdf_safe(lbl), ParagraphStyle('c', parent=styles["Normal"], fontSize=8)),
                Paragraph(f'{_wf*100:.1f}%', ParagraphStyle('c', parent=styles["Normal"], fontSize=8, alignment=1)),
                BarFlowable(_wf, _col, width=5*cm, height=0.3*cm),
            ])

        t = Table(rows, colWidths=[6*cm, 2*cm, 5.5*cm], rowHeights=[0.55*cm] + [0.5*cm]*len(labels))
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        return t

    def metrics_table(res, is_interp=False, interp_ret=None, interp_std=None, gain=None):
        if is_interp:
            data = [
                ['Metric', 'Value'],
                ['Expected return (interpolated)', f'{interp_ret:.2f}%'],
                ['Std deviation', f'{interp_std:.2f}%'],
                ['Return vs Portfolio (1)', f'{gain:+.2f} pp'],
            ]
        else:
            data = [
                ['Metric', 'Value'],
                ['Expected return', f"{res['expected_return']*100:.2f}%"],
                ['Std deviation',   f"{res['std_dev']*100:.2f}%"],
                ['Skewness',        f"{res['skewness']:.3f}"],
                ['Realised ES E[r|r<H]' if use_es else 'Realised P(r<H)',  f"{res['shortfall_stat']*100:.2f}%"],
            ]
        t = Table(data, colWidths=[8*cm, 4*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy),
            ('TEXTCOLOR',  (0,0), (-1,0), white),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ]))
        return t

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph('Portfolio Optimisation Report', title_style))
    story.append(Paragraph(
        'Beyond Mean-Variance: Portfolio Optimiser with Derivatives &amp; Structured Products — A Mental Accounts Framework',
        ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))
    story.append(Paragraph(
        f'Generated: {datetime.datetime.now().strftime("%d %B %Y, %H:%M")} | '
        f'Constraint: {_pdf_safe(constraint_label)} | Data: {_pdf_safe(data_mode.split("(")[0].strip())}',
        caption_style))
    story.append(HRFlowable(width='100%', thickness=1, color=blue, spaceAfter=8))

    # ── Simulation parameters ─────────────────────────────────────────────────
    story += section_header('Simulation Parameters', navy)
    params = [
        ['Parameter', 'Value'],
        ['Data source', data_mode.split('(')[0].strip()],
        ['Securities', ', '.join(names_in)],
        ['Derivative', der_label_sel if der_label_sel else 'None'],
        ['Constraint type', 'Expected Shortfall' if use_es else 'Value-at-Risk (VaR)'],
        ['Threshold H', f'{H_val:.0%}'],
        ['Max shortfall prob (α)' if not use_es else 'Min ES (L)', f'{_alpha:.0%}' if not use_es else f'{_L:.0%}'],
        ['Implied risk-aversion λ', lam_summary],
        ['Grid resolution', grid_lbl.split('(')[0].strip()],
    ]
    t = Table(params, colWidths=[7*cm, 7*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('TEXTCOLOR',  (0,0), (-1,0), white),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # ── Optimisation chart ────────────────────────────────────────────────────
    _chart_bytes = fig_png if fig_png is not None else (
        fig_plotly.to_image(format='png', width=700, height=400, scale=2)
        if fig_plotly is not None else None
    )
    if _chart_bytes is not None:
        try:
            _img_buf = io.BytesIO(_chart_bytes)
            _rl_img = RLImage(_img_buf, width=17*cm, height=9.7*cm)
            story += section_header('Optimisation Chart — Efficient Frontiers', navy)
            story.append(_rl_img)
            story.append(Paragraph(
                'Purple dashed: MV frontier (Markowitz) · Blue: Behavioural frontier without derivatives · '
                'Gold squares: Behavioural frontier with derivatives · '
                'Purple circle: Portfolio (0) · Green diamond: Portfolio (1) · Orange square: Portfolio (2) · Coral star: Portfolio (3)',
                caption_style))
            story.append(Spacer(1, 8))
        except Exception as _chart_err:
            story.append(Paragraph(f'Chart export unavailable: {_pdf_safe(str(_chart_err))}', caption_style))

    # ── Page break — portfolios start on page 2 ──────────────────────────────
    story.append(PageBreak())

    # ── Summary — resulting portfolios ───────────────────────────────────────
    _p1_ref = nd_res['expected_return'] * 100 if nd_res else None
    def _dpp(ret_pct):
        if _p1_ref is None or ret_pct is None:
            return '\u2014'
        return f'{ret_pct - _p1_ref:+.2f} pp'
    _sum = [['Portfolio', 'Exp. return', 'Std dev', 'Skewness', 'Gap vs (1)']]
    if p0_stats:
        _sum.append(['Portfolio (0) \u2014 Markowitz MV optimum',
                     f"{p0_stats['expected_return']*100:.2f}%", f"{p0_stats['std_dev']*100:.2f}%",
                     '0.000', _dpp(p0_stats['expected_return']*100)])
    if nd_res:
        _sum.append(['Portfolio (1) \u2014 Behavioural, no derivative',
                     f"{nd_res['expected_return']*100:.2f}%", f"{nd_res['std_dev']*100:.2f}%",
                     f"{nd_res['skewness']:.3f}", '\u2014'])
    if dr_res and der_label_sel:
        _sum.append([f'Portfolio (2) \u2014 with {der_label_sel}',
                     f"{dr_res['expected_return']*100:.2f}%", f"{dr_res['std_dev']*100:.2f}%",
                     f"{dr_res['skewness']:.3f}", _dpp(dr_res['expected_return']*100)])
    if p3_return is not None:
        _sum.append([f'Portfolio (3) \u2014 same variance as (1), with {der_label_sel}',
                     f"{p3_return:.2f}%", f"{p3_std:.2f}%" if p3_std is not None else '\u2014',
                     '\u2014', _dpp(p3_return)])
    if len(_sum) > 1:
        story += section_header('Summary \u2014 resulting portfolios', navy)
        _stab = Table(_sum, colWidths=[7.2*cm, 2.6*cm, 2.2*cm, 2.2*cm, 2.3*cm])
        _stab.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy),
            ('TEXTCOLOR', (0,0), (-1,0), white),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(_stab)
        story.append(Paragraph(
            'Gap vs (1) = expected-return gap relative to Portfolio (1). '
            'Portfolio (0) is a Gaussian mean-variance construct (skewness 0). '
            'Portfolio (3) is interpolated from the derivative frontier (indicative only).',
            caption_style))
        story.append(Spacer(1, 8))

    # ── Portfolio (0) — Markowitz MV optimum ─────────────────────────────────
    if p0_stats:
        story += section_header('Portfolio (0) \u2014 Markowitz mean-variance optimum (no derivative)',
                                colors.HexColor('#a855f7'))
        story.append(Paragraph(
            "Minimum-variance portfolio at Portfolio (1)'s expected return \u2014 coincides with "
            "Portfolio (1) when it is mean-variance efficient (the MVT/MAT equivalence). "
            "Gaussian construct: skewness 0; tail probability from the normal model.",
            caption_style))
        story.append(metrics_table(p0_stats))
        if p0_labels and p0_weights:
            story.append(Spacer(1, 4))
            story.append(weights_table(p0_labels, p0_weights, p0_colors, 'Security'))
        story.append(Spacer(1, 8))

    # ── Portfolio (1) ─────────────────────────────────────────────────────────
    if nd_res:
        story += section_header('Portfolio (1) — Behavioural optimum without derivatives', green)
        story.append(metrics_table(nd_res))
        story.append(Spacer(1, 4))
        story.append(weights_table(nd_labels, nd_weights, nd_colors, 'Security'))
        story.append(Spacer(1, 8))

    # ── Portfolio (2) ─────────────────────────────────────────────────────────
    if dr_res and der_label_sel:
        story += section_header(f'Portfolio (2) — Optimum with {der_label_sel}', gold)
        if nd_res:
            delta = (dr_res['expected_return'] - nd_res['expected_return']) * 100
            story.append(Paragraph(
                f'Return vs Portfolio (1): <b>{delta:+.2f} pp</b> at same constraint (H={H_val:.0%}, alpha={_alpha:.0%})',
                body_style))
        story.append(metrics_table(dr_res))
        story.append(Spacer(1, 4))
        story.append(weights_table(dr_labels, dr_weights, dr_colors, 'Security / Instrument'))
        story.append(Spacer(1, 8))

    # ── Portfolio (3) ─────────────────────────────────────────────────────────
    if p3_return is not None and nd_res:
        story += section_header(f'Portfolio (3) — Same variance as Portfolio (1), with {der_label_sel}', coral)
        story.append(Paragraph(
            'Interpolated from the derivative frontier — indicative only. '
            'Portfolio (3) is not directly optimised: its return is obtained by interpolating '
            'the derivative frontier at the same standard deviation as Portfolio (1). '
            'No exact weight vector exists — the allocation below is the derivative frontier '
            'allocation from Portfolio (2) provided for reference only.',
            caption_style))
        gain = p3_return - nd_res['expected_return'] * 100
        story.append(metrics_table(None, is_interp=True,
                                   interp_ret=p3_return, interp_std=p3_std, gain=gain))
        story.append(Spacer(1, 4))
        if dr_labels and dr_weights:
            story.append(Paragraph(
                '<i>Reference allocation (from Portfolio (2) derivative frontier — indicative only):</i>',
                ParagraphStyle('ref', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
            story.append(Spacer(1, 2))
            story.append(weights_table(dr_labels, dr_weights, dr_colors, 'Security / Instrument'))
        story.append(Spacer(1, 8))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        'This report is for educational and research purposes only. '
        'It does not constitute financial advice or investment recommendations. '
        'Built on Das &amp; Statman (2009), Das, Markowitz, Scheid &amp; Statman (2010) JFQA, and Jeddou (2012) USI Lugano.',
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey)))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ── Shared helpers for the scalable + backtest PDF reports ────────────────────
def _pdf_safe(s):
    """Replace glyphs the base PDF font can't render with ASCII equivalents, and
    XML-escape &, <, > so ReportLab's Paragraph parser doesn't treat them as markup
    (e.g. 'P(r < H)' or a '<=' from a '≤' would otherwise open an unclosed tag).
    Intended markup such as <b>/<br/> is added by callers *outside* this function."""
    if s is None:
        return ""
    s = str(s).replace("&", "&amp;")
    for a, b in (("\u2264", "<="), ("\u2265", ">="), ("\u2014", "-"), ("\u2013", "-"),
                 ("\u2022", "-"), ("\u00b7", "|"), ("\u26a0", "(!)"), ("\u2713", "(ok)"),
                 ("\u2605", "*"), ("\u03b1", "alpha"), ("\u03c3", "sigma"), ("\u03b2", "beta"),
                 ("\u00b2", "^2"), ("\u2009", " "), ("\u00a0", " "), ("\u2192", "->"),
                 ("\u00d7", "x")):
        s = s.replace(a, b)
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _md_bold(s):
    """Convert simple **bold** markdown to PDF-safe <b>bold</b>."""
    parts = _pdf_safe(s).split("**")
    return "".join(p if i % 2 == 0 else f"<b>{p}</b>" for i, p in enumerate(parts))


def _styled_pdf(title, subtitle, blocks,
                footer_note="Beyond Mean-Variance Portfolio Optimiser | Jeddou (2026)"):
    """Generic styled report builder, matching the look of the grid PDF.

    blocks: list of dicts, each one of:
      {"type":"section","text":...,"color":"blue|green|gold|navy"}
      {"type":"para","text": HTML-ish string}
      {"type":"caption","text":...}
      {"type":"spacer","h":points}
      {"type":"metrics","items":[(label,value), ...]}
      {"type":"table","data":[[...], ...],"header":bool,"col_widths":[...]}
      {"type":"weights","rows":[(label, fraction, hexcolor), ...]}
      {"type":"image","png":bytes,"width":cm,"caption":...}
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable,
                                     Image as RLImage, Flowable)
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.utils import ImageReader
    import io as _io

    navy = colors.HexColor('#0d1a2e'); blue = colors.HexColor('#1a6bbf')
    green = colors.HexColor('#10b981'); gold = colors.HexColor('#f59e0b')
    light = colors.HexColor('#5b6b86'); white = colors.white
    _accent = {'blue': blue, 'green': green, 'gold': gold, 'navy': navy}

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T2', parent=styles['Title'], textColor=navy,
                                 fontSize=16, spaceAfter=2)
    sub_style = ParagraphStyle('S2', parent=styles['Normal'], textColor=light,
                               fontSize=9, spaceAfter=8)
    body_style = ParagraphStyle('B2', parent=styles['Normal'], fontSize=9,
                                spaceAfter=4, leading=13)
    cap_style = ParagraphStyle('C2', parent=styles['Normal'], fontSize=8,
                               textColor=colors.grey, spaceAfter=3, leading=11)

    class _Doc(SimpleDocTemplate):
        def handle_pageEnd(self):
            self.canv.saveState(); self.canv.setFont('Helvetica', 8)
            self.canv.setFillColor(colors.grey)
            self.canv.drawCentredString(A4[0] / 2, 1.2 * cm,
                                        f"Page {self.canv.getPageNumber()} - {footer_note}")
            self.canv.restoreState(); super().handle_pageEnd()

    class _Bar(Flowable):
        def __init__(self, frac, c, width=6.5 * cm, height=0.34 * cm):
            Flowable.__init__(self)
            self.frac = max(0.0, min(1.0, frac)); self.c = c
            self.width = width; self.height = height

        def draw(self):
            self.canv.setFillColor(colors.HexColor('#e9ecef'))
            self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
            if self.frac > 0:
                self.canv.setFillColor(self.c)
                self.canv.rect(0, 0, self.width * self.frac, self.height, fill=1, stroke=0)

    def _section(text, color):
        return Paragraph(_pdf_safe(text), ParagraphStyle(
            'SH2', parent=styles['Heading2'], textColor=white, backColor=color,
            fontSize=10, spaceBefore=8, spaceAfter=4, leftIndent=-6, rightIndent=-6,
            borderPadding=(4, 6, 4, 6)))

    buf = _io.BytesIO()
    doc = _Doc(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
               topMargin=2 * cm, bottomMargin=2.5 * cm)
    story = [Paragraph(_pdf_safe(title), title_style)]
    if subtitle:
        story.append(Paragraph(_pdf_safe(subtitle), sub_style))
    story.append(HRFlowable(width="100%", thickness=1.4, color=blue, spaceAfter=8))

    for blk in blocks:
        t = blk.get("type")
        if t == "section":
            story.append(_section(blk["text"], _accent.get(blk.get("color", "blue"), blue)))
        elif t == "para":
            story.append(Paragraph(blk["text"], body_style))
        elif t == "caption":
            story.append(Paragraph(blk["text"], cap_style))
        elif t == "spacer":
            story.append(Spacer(1, blk.get("h", 6)))
        elif t == "metrics":
            items = blk["items"]
            cells = [[Paragraph(
                f'<font color="#1a6bbf" size="9"><b>{_pdf_safe(lab)}</b></font><br/>'
                f'<font size="14"><b>{_pdf_safe(val)}</b></font>',
                ParagraphStyle('M2', parent=styles['Normal'], alignment=TA_CENTER, leading=16))
                for lab, val in items]]
            tbl = Table(cells, colWidths=[(17.0 / len(items)) * cm for _ in items])
            tbl.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7e2')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7e2')),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f4f7fb')),
                ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
            story.append(tbl); story.append(Spacer(1, 6))
        elif t == "table":
            data = blk["data"]; header = blk.get("header", True)
            tdata = []
            for ri, row in enumerate(data):
                if header and ri == 0:
                    tdata.append([Paragraph(f'<b><font color="white">{_pdf_safe(c)}</font></b>',
                                            body_style) for c in row])
                else:
                    tdata.append([Paragraph(_pdf_safe(c), body_style) for c in row])
            tbl = Table(tdata, colWidths=blk.get("col_widths"))
            ts = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#d0d7e2')),
                  ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                  ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                  ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                  ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#f4f7fb')])]
            if header:
                ts.append(('BACKGROUND', (0, 0), (-1, 0), blue))
            tbl.setStyle(TableStyle(ts)); story.append(tbl); story.append(Spacer(1, 6))
        elif t == "weights":
            wdata = []
            for lab, frac, c in blk["rows"]:
                try:
                    _bc = colors.HexColor(c)
                except Exception:
                    _bc = blue
                wdata.append([Paragraph(f'<font color="{c}"><b>{_pdf_safe(lab)}</b></font>', body_style),
                              _Bar(float(frac), _bc),
                              Paragraph(f'<b>{float(frac) * 100:.1f}%</b>', body_style)])
            tbl = Table(wdata, colWidths=[5 * cm, 7 * cm, 2 * cm])
            tbl.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                     ('TOPPADDING', (0, 0), (-1, -1), 2),
                                     ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
            story.append(tbl); story.append(Spacer(1, 6))
        elif t == "image" and blk.get("png"):
            try:
                ir = ImageReader(_io.BytesIO(blk["png"])); iw, ih = ir.getSize()
                w = blk.get("width", 17 * cm); h = w * ih / float(iw)
                story.append(RLImage(_io.BytesIO(blk["png"]), width=w, height=h))
                if blk.get("caption"):
                    story.append(Paragraph(_pdf_safe(blk["caption"]), cap_style))
                story.append(Spacer(1, 6))
            except Exception:
                pass

    doc.build(story)
    return buf.getvalue()


def generate_mc_pdf_report(meta, weight_rows, der_lines=None, fig_png=None):
    """Build the Scalable (Monte-Carlo + CVaR) optimiser results PDF."""
    blocks = [
        {"type": "section", "text": "Scalable Monte-Carlo + CVaR optimum", "color": "gold"},
        {"type": "metrics", "items": [
            ("Expected return", meta["er"]), ("Realised CVaR", meta["es"]),
            ("Volatility", meta["vol"]), ("Skewness", meta["skew"])]},
        {"type": "para", "text": meta["summary_html"]},
    ]
    if der_lines:
        blocks.append({"type": "section", "text": "Derivative pricing used", "color": "blue"})
        for dl in der_lines:
            blocks.append({"type": "para", "text": dl})
    if fig_png:
        blocks.append({"type": "section", "text": "Return / tail-risk frontier", "color": "blue"})
        blocks.append({"type": "image", "png": fig_png, "caption": meta.get("chart_caption", "")})
    blocks.append({"type": "section", "text": "Portfolio weights", "color": "blue"})
    blocks.append({"type": "weights", "rows": weight_rows})
    if meta.get("footer_caption"):
        blocks.append({"type": "caption", "text": meta["footer_caption"]})
    return _styled_pdf("Scalable Portfolio Optimiser - Results",
                       meta.get("subtitle", ""), blocks,
                       footer_note="Scalable MC + CVaR Optimiser | Beyond Mean-Variance | Jeddou (2026)")


def generate_backtest_pdf_report(meta, summary_rows, ab_rows=None,
                                 verdict_lines=None, fig_png=None):
    """Build the Out-of-Sample Backtest results PDF."""
    blocks = [{"type": "para", "text": meta["period_html"]}]
    if fig_png:
        blocks.append({"type": "section", "text": "Out-of-sample portfolio paths", "color": "green"})
        blocks.append({"type": "image", "png": fig_png, "caption": meta.get("chart_caption", "")})
    blocks.append({"type": "section", "text": "Expected vs realised", "color": "blue"})
    blocks.append({"type": "table", "data": summary_rows, "header": True,
                   "col_widths": None})
    if ab_rows:
        blocks.append({"type": "section", "text": "Alpha / beta vs benchmark", "color": "blue"})
        blocks.append({"type": "table", "data": ab_rows, "header": True, "col_widths": None})
    if verdict_lines:
        blocks.append({"type": "section", "text": "Verdict", "color": "green"})
        for v in verdict_lines:
            blocks.append({"type": "para", "text": v})
    if meta.get("footer_caption"):
        blocks.append({"type": "caption", "text": meta["footer_caption"]})
    return _styled_pdf("Out-of-Sample Backtest - Results",
                       meta.get("subtitle", ""), blocks,
                       footer_note="Out-of-Sample Backtest | Beyond Mean-Variance | Jeddou (2026)")


from scipy.optimize import minimize
from io import StringIO
from datetime import date, timedelta
from behavioral_portfolio_optimizer import (
    build_state_space, assign_probabilities, optimize_portfolio,
    compute_structured_payoff, bs_call, bs_put
)
from turbo_optimizer import optimize_portfolio_turbo
from es_rigorous import optimize_portfolio_es_rigorous
from core.pricing import (
    _SYN_UNDERLYING, _COMPONENT_PRESETS, preset_components, build_der_config,
    _bt_legs, _leg_value, mtm_gross_path, _mc_leg_intrinsic, _mc_leg_value_vec,
    mc_der_returns,
)
from core.scenario import (
    mc_generate_scenarios, mc_build_matrix, mc_max_return_cvar, mc_min_cvar,
    mc_frontier, mc_realised_es, mc_analytical_es, mc_gmv_weights,
    _mc_psd_cholesky, _mc_cvar_rows,
)
from core.grid import (
    run_opt, build_frontier, mv_frontier_at_return, implied_lambda,
)
from core.backtest import _bt_portfolio_path, _bt_metrics, _capm_alpha_beta
from core import risk_profile as _rp
from core import ratios as _rt
from core import gdrive as _gd
from core.markets import (
    corr_to_cov, clean_returns, parse_csv, fetch_tickers, fetch_close_prices,
    stats_from_prices, fetch_ticker_info, fetch_ticker_history, fetch_ticker_financials,
)
try:                                    # newer statement fetchers; tolerate a not-yet-reloaded module
    from core.markets import fetch_ticker_cashflow, fetch_ticker_balance
except Exception:
    fetch_ticker_cashflow = fetch_ticker_balance = None
from scipy.stats import norm as _norm
from scipy.optimize import brentq as _brentq

# mv_frontier_at_return -> moved to core/grid.py


# implied_lambda -> moved to core/grid.py

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beyond Mean-Variance Portfolio Optimiser",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main{background:#0d1117}.block-container{padding-top:1.5rem;padding-left:2.5rem !important;padding-right:2.5rem !important}
h1{color:#fff;font-size:1.6rem}h2,h3{color:#c0c8d8}
.info-box{background:#1a1a2e;border:none;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;color:#ffffff !important}
.warn-box{background:#1a1200;border:1px solid #f59e0b;border-radius:6px;padding:.5rem 1rem;color:#f59e0b;font-size:.82rem;margin-top:.3rem}
.ok-box{background:rgba(16,185,129,.12);border:1px solid #2f8f68;border-radius:6px;padding:.5rem 1rem;color:#86e0b0;font-size:.82rem;margin-top:.3rem}

    /* Larger tab labels */
    button[data-baseweb="tab"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stMain"] > div > div.block-container {
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }
.section-header{border:1px solid #30363d;background:linear-gradient(165deg,#1b2330,#161b22);padding:.4rem .8rem;border-radius:12px;margin-top:1.2rem;margin-bottom:.5rem;color:#E3C77E;font-weight:600;font-size:1.05rem;letter-spacing:.02em;text-align:center;overflow:hidden}
#sh1 ~ #sh1{display:none !important}
/* Hide any section-header or h2 that leaks into main content area */
section[data-testid="stMain"] .section-header {display:none !important}
section[data-testid="stMain"] h2:has(+ hr) {display:none !important}

    .sidebar-divider{border:none;border-top:2px solid #2a3a4a;margin:1rem 0}
    section[data-testid="stSidebar"] div.stButton > button,section[data-testid="stSidebar"] div.stButton > button[kind="primary"]{background:linear-gradient(180deg,#5aabff 0%,#2d7dd2 100%) !important;border:none !important;border-bottom:3px solid #1a5fa0 !important;border-radius:8px !important;color:#ffffff !important;font-size:1.05rem !important;font-weight:700 !important;padding:.6rem 1rem !important;box-shadow:0 4px 8px rgba(0,0,0,0.5) !important;text-shadow:0 1px 2px rgba(0,0,0,0.3) !important;width:100% !important}
    section[data-testid="stSidebar"] div.stButton > button:hover{background:linear-gradient(180deg,#6bbfff 0%,#3a8de0 100%) !important;box-shadow:0 6px 14px rgba(0,0,0,0.6) !important;transform:translateY(-1px) !important}
    section[data-testid="stSidebar"] div.stButton > button:active{background:linear-gradient(180deg,#2d7dd2 0%,#1a5fa0 100%) !important;border-bottom:1px solid #1a5fa0 !important;transform:translateY(1px) !important}
/* ── Dark sidebar theme (HTML boxes keep their own colours) ─────────────── */
section[data-testid="stSidebar"]{background:#0d1117 !important}
section[data-testid="stSidebar"] hr{border-color:#2a3a4a !important}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [role="radiogroup"] label *{color:#e7ecf4 !important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea{background:#161b22 !important;color:#e7ecf4 !important;-webkit-text-fill-color:#e7ecf4 !important}
section[data-testid="stSidebar"] [data-baseweb="input"],section[data-testid="stSidebar"] [data-baseweb="select"]>div,section[data-testid="stSidebar"] [data-baseweb="base-input"]{background:#161b22 !important;border-color:#30363d !important}
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:#e7ecf4 !important}
section[data-testid="stSidebar"] [data-testid="stExpander"]{background:#161b22 !important;border-color:#30363d !important}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,section[data-testid="stSidebar"] [data-testid="stExpander"] p{color:#e7ecf4 !important}
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"]{color:#ffffff !important}
/* Frames around each numbered section (the container that holds that section's header) */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec1hdr):not(:has(#sec2hdr)),section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec2hdr):not(:has(#sec1hdr)),section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec3hdr):not(:has(#sec1hdr)),section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec4hdr):not(:has(#sec1hdr)){border:1px solid #30363d;border-radius:12px;background:linear-gradient(165deg,#1b2330,#161b22);padding:.55rem .65rem}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec1hdr):not(:has(#sec2hdr)) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child [data-testid="stMarkdownContainer"],section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec2hdr):not(:has(#sec1hdr)) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child [data-testid="stMarkdownContainer"],section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec3hdr):not(:has(#sec1hdr)) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child [data-testid="stMarkdownContainer"],section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(#sec4hdr):not(:has(#sec1hdr)) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child [data-testid="stMarkdownContainer"]{margin-bottom:0 !important}
/* Gold bottom buttons: Run optimiser + Reset/New simulation */
section[data-testid="stSidebar"] .st-key-run_opt_btn div.stButton > button,section[data-testid="stSidebar"] .st-key-reset_sim_btn div.stButton > button,section[data-testid="stSidebar"] .st-key-fetch_btn div.stButton > button,section[data-testid="stSidebar"] .st-key-sample_csv_btn div.stDownloadButton > button,section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{background:linear-gradient(180deg,#E3C77E 0%,#C9A24B 100%) !important;border:none !important;border-bottom:3px solid #9a7b2e !important;color:#0d1117 !important;text-shadow:none !important}
section[data-testid="stSidebar"] .st-key-run_opt_btn div.stButton > button:hover,section[data-testid="stSidebar"] .st-key-reset_sim_btn div.stButton > button:hover,section[data-testid="stSidebar"] .st-key-fetch_btn div.stButton > button:hover,section[data-testid="stSidebar"] .st-key-sample_csv_btn div.stDownloadButton > button:hover,section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover{background:linear-gradient(180deg,#edd596 0%,#d4b15e 100%) !important}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.05) !important;border:1px solid #34527a !important;align-items:center !important;text-align:center !important}
section[data-testid="stSidebar"] .st-key-sample_csv_btn div.stDownloadButton{display:flex !important;justify-content:center !important}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *{color:#c0c8d8 !important}
/* ── Dark main content: convert hard-coded light panels to the frame shade ─────── */
section[data-testid="stMain"] [style*="background: rgb(255, 255, 255)"][style*="border-radius: 50%"]{background:#E3C77E !important}
section[data-testid="stMain"] [style*="background: rgb(255, 255, 255)"]:not([style*="border-radius: 50%"]),section[data-testid="stMain"] [style*="background: rgb(255, 251, 234)"],section[data-testid="stMain"] [style*="background: rgb(240, 244, 255)"]{background:#1b2330 !important;border-color:#30363d !important}
section[data-testid="stMain"] [style*="color: rgb(17, 17, 17)"],section[data-testid="stMain"] [style*="color: rgb(51, 51, 51)"],section[data-testid="stMain"] [style*="color: rgb(85, 85, 85)"],section[data-testid="stMain"] [style*="color: rgb(26, 58, 107)"],section[data-testid="stMain"] [style*="color: rgb(26, 58, 92)"]{color:#e7ecf4 !important}
/* remove Streamlit's default table cell borders inside the How-to / Output-chart info boxes */
section[data-testid="stMain"] .info-box table,section[data-testid="stMain"] .info-box table th,section[data-testid="stMain"] .info-box table td,section[data-testid="stMain"] .info-box table tr{border:none !important}
/* Gold "Tools" back button (top of page) */
.st-key-_nav_back div.stButton > button{background:linear-gradient(180deg,#E3C77E 0%,#C9A24B 100%) !important;border:none !important;border-bottom:3px solid #9a7b2e !important;color:#0d1117 !important;text-shadow:none !important}
.st-key-_nav_back div.stButton > button:hover{background:linear-gradient(180deg,#edd596 0%,#d4b15e 100%) !important}
section[data-testid="stSidebar"] .st-key-_nav_back div.stButton > button{background:linear-gradient(180deg,#E3C77E 0%,#C9A24B 100%) !important;border-bottom:3px solid #9a7b2e !important;color:#0d1117 !important}
section[data-testid="stSidebar"] .st-key-_nav_back div.stButton > button:hover{background:linear-gradient(180deg,#edd596 0%,#d4b15e 100%) !important}
.st-key-_nav_back div.stButton > button{text-transform:uppercase !important}
/* Glossary "Ask AI" button — double width */
.st-key-gloss_ask div.stButton > button,.st-key-gloss_clear div.stButton > button{min-width:194px !important}
.st-key-gloss_ask div.stButton{display:flex !important;justify-content:center !important}
.st-key-gloss_clear{margin-top:1.1rem !important}
/* Dark text on gold primary buttons in main area (PDF export, run buttons) */
section[data-testid="stMain"] button[kind="primary"],section[data-testid="stMain"] button[kind="primary"] *,section[data-testid="stMain"] button[kind="primaryFormSubmit"],section[data-testid="stMain"] button[kind="primaryFormSubmit"] *{color:#0d1117 !important}
/* Frameless Summary table (the #0d1a2e box) */
section[data-testid="stMain"] [style*="background: rgb(13, 26, 46)"] table,section[data-testid="stMain"] [style*="background: rgb(13, 26, 46)"] table td,section[data-testid="stMain"] [style*="background: rgb(13, 26, 46)"] table th,section[data-testid="stMain"] [style*="background: rgb(13, 26, 46)"] table tr{border:none !important}
/* Half-width, centered PDF export button */
.st-key-pdf_download div.stDownloadButton > button{width:50% !important}
.st-key-pdf_download div.stDownloadButton{display:flex !important;justify-content:center !important}
/* Half-width, centered Run/Export buttons on Backtest & Scalable pages */
.st-key-bt_run div.stButton > button,.st-key-mc_run div.stButton > button,.st-key-bt_pdf_download div.stDownloadButton > button{width:50% !important}
.st-key-bt_run div.stButton,.st-key-mc_run div.stButton,.st-key-bt_pdf_download div.stDownloadButton{display:flex !important;justify-content:center !important}
/* Portfolio result frames — match the Optimisation Parameters section frame */
section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]{border-color:#30363d !important}
section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.pf-frame-marker):not(:has([data-testid="stVerticalBlockBorderWrapper"] .pf-frame-marker)){background:#1b2330 !important}
section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.pf-frame-marker) [data-baseweb="input"],section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.pf-frame-marker) [data-baseweb="base-input"],section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.pf-frame-marker) [data-baseweb="select"]>div{background-color:#0d1117 !important}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# ── Static explanations dictionary (no API cost) ─────────────────────────────
EXPLANATIONS = {
    # ── Objectives & feasibility (Grid 3D landscape) ─────────────────────────
    "Downside-aware utility (mean − κ·CVaR)": (
        "A downside-aware utility scores a portfolio as its expected return minus a penalty on tail "
        "risk: U = E[r] − κ·CVaR, where CVaR (the α-conditional VaR) is the average return in the worst "
        "α-tail and κ ≥ 0 sets how strongly a deep tail is penalised. Unlike mean-variance — which "
        "penalises symmetric variance and so taxes upside and downside alike — it charges only for the "
        "downside, so it rewards the asymmetric, insured payoffs that derivatives add. In this app it is "
        "one of the objectives you can plot on the Grid optimiser's 3D landscape; κ is calibrated so the "
        "behavioural optimum sits near the peak. It is a lens for ranking portfolios by return-versus-tail, "
        "not the engine's hard rule — the optimiser still enforces the explicit downside limit "
        "(P(r<H) ≤ α for VaR, or the ES floor E[r|r<H] ≥ L)."
    ),
    "Constrained optimum (downside-feasible region)": (
        "The optimiser maximises expected return subject to a downside limit — P(r<H) ≤ α (VaR) or "
        "E[r|r<H] ≥ L (Expected Shortfall). Portfolios that satisfy the limit form the feasible region; "
        "those that breach it are infeasible. The optimum is the best portfolio inside that region, so it "
        "typically sits on the constraint boundary — the riskiest allocation still allowed — rather than at "
        "the unconstrained peak of a smooth objective surface. On the Grid optimiser's 3D landscape the "
        "infeasible region is greyed out and the feasible region is coloured, so the marked optimum sits at "
        "the top edge of the coloured zone: the greyed peak beyond it scores higher but breaks the limit, "
        "which is exactly what optimising under a downside constraint means."
    ),
    "Drawdown": (
        "Drawdown is how far a portfolio has fallen from its own running peak (its high-water mark) — a "
        "cumulative, path-dependent measure of the loss you are currently sitting in. A <i>drawdown breach</i> "
        "is when that decline passes a chosen loss limit <b>H</b>. Because it accumulates many small losses, it "
        "catches a slow grind lower that a single-period return test would miss, which makes it the most "
        "intuitive &ldquo;am I down more than I can stomach?&rdquo; reading. In the Live Portfolio tool the "
        "drawdown is plotted over time against your H limit, and the share of days spent beyond H is compared "
        "to your shortfall tolerance &alpha;."
    ),
    "Horizon-return": (
        "A horizon-return is the portfolio's return measured over a rolling window of a chosen length (the "
        "horizon — e.g. 1, 3, 6 or 12 months). A <i>horizon-return breach</i> is a window whose return falls "
        "below your loss limit <b>H</b>; keeping the breach frequency at or under <b>&alpha;</b> is exactly the "
        "mental-accounting constraint <b>P(r &lt; H) &le; &alpha;</b> that the Grid and Scalable optimisers "
        "enforce. Unlike drawdown it is measured from a fixed window start rather than the running peak, so a "
        "slow grind can stay within limit on the horizon view while still showing as a drawdown. Short horizons "
        "react to single shocks; long horizons need more history. The Live Portfolio tool plots it against H "
        "with the same green-to-red shading."
    ),
    "Stress testing": (
        "Stress testing asks how a portfolio would behave under <i>adverse</i> conditions, rather than how it has "
        "behaved on average — the forward-looking complement to realised risk metrics. Three common forms: "
        "<b>historical scenario replay</b> applies the actual returns from a past crisis (e.g. the 2008 financial "
        "crisis, the 2020 COVID crash, the 2022 selloff) to today's weights; an <b>instantaneous shock</b> moves "
        "the market by a set amount — propagated through the portfolio's <b>&beta;</b> — or shocks individual "
        "holdings; and a <b>parametric stress</b> scales volatilities and pushes correlations toward 1 (capturing "
        "how diversification tends to break down in a crash), then re-derives Value at Risk. In the Live Portfolio "
        "tool each result is judged against your loss limit <b>H</b>, so you can see which scenarios would breach "
        "the protection your risk profile set."
    ),
    # ── Derivatives ───────────────────────────────────────────────────────────
    "Put option": (
        "A put option gives the holder the right to sell the underlying asset at a fixed strike price. "
        "Its payoff is max(K - S_T, 0), increasing in value when the underlying falls below the strike. "
        "In a behavioural portfolio, a long put acts as downside insurance — it reduces the probability "
        "of breaching the mental-account threshold H, allowing the optimizer to allocate more to "
        "high-return risky assets while satisfying the shortfall constraint."
    ),
    "Call option": (
        "A call option gives the holder the right to buy the underlying asset at a fixed strike price. "
        "Its payoff is max(S_T - K, 0), increasing in value when the underlying rises above the strike. "
        "In a behavioural portfolio, a long call provides leveraged upside participation with limited "
        "downside. It adds positive skewness to the return distribution, which can raise expected return "
        "for a given mental-account constraint level."
    ),
    "Safety collar (long put + short call)": (
        "A safety collar combines a long put (downside protection) and a short call (capping upside). "
        "The short call premium offsets part of the put cost, making the hedge cheaper. "
        "The result is a return profile bounded on both sides: losses are limited below the put strike, "
        "but gains are also capped above the call strike. "
        "Useful when the investor wants cheap downside protection and is willing to sacrifice extreme upside."
    ),
    "Aggressive collar (long call + short put)": (
        "An aggressive collar combines a long call (upside participation) and a short put (accepting downside risk). "
        "The short put premium finances the call, making upside exposure cheaper. "
        "Unlike the safety collar, this structure increases rather than reduces downside risk — "
        "it suits investors with a strong upward view who are comfortable bearing more tail risk "
        "in exchange for leveraged upside."
    ),
    "Straddle (long call + long put)": (
        "A straddle combines a long call and a long put at the same strike, profiting when the "
        "underlying moves significantly in either direction. "
        "It is a bet on high volatility regardless of direction. "
        "In a behavioural portfolio context, a straddle adds fat tails and positive excess kurtosis "
        "to the return distribution — it performs well in extreme market moves and can help "
        "satisfy mental-account constraints when large moves are expected."
    ),
    "Strangle (long call + long put, diff strikes)": (
        "A strangle is similar to a straddle but uses different strikes for the call and put, "
        "making it cheaper since both options are out-of-the-money. "
        "It profits from large moves in either direction but requires a bigger move than a straddle to break even. "
        "In a behavioural portfolio, it provides asymmetric tail protection at lower cost than a straddle, "
        "useful when extreme but not moderate moves are expected."
    ),
    "Capital-guaranteed note — uncapped": (
        "A capital-guaranteed note (CGN) is a structured product that returns your invested capital "
        "(plus a floor F) at maturity, while providing participation in the upside of an underlying asset. "
        "The uncapped version places no ceiling on that upside participation. "
        "In a behavioural portfolio its role is structural rather than return-maximising: the capital "
        "guarantee directly satisfies the mental-account downside constraint (the chance of finishing "
        "below the safety threshold H stays within the tolerance α), so the optimizer can hold the note "
        "without breaching that limit. But the protection is not free — its cost is embedded in the "
        "note's price. Once the CGN is priced correctly, the expected-return gain over a portfolio of "
        "primary securities alone is typically modest: the CGN buys downside safety, not a large return premium."
    ),
    "Capital-guaranteed note — capped": (
        "A capped CGN is identical to the uncapped version but limits upside participation above a cap level. "
        "The cap reduces the cost of the product (the issuer saves on the call spread), making it cheaper "
        "than the uncapped version. "
        "The trade-off is sacrificed upside beyond the cap. "
        "In a behavioural portfolio it offers the same structural benefit as the uncapped note — the "
        "capital guarantee satisfies the mental-account downside constraint — at lower cost, though with "
        "a smaller expected-return contribution when the underlying performs strongly. As with the uncapped "
        "note, that contribution is modest once the product is priced correctly."
    ),
    "Barrier-M note": (
        "A barrier-M note pays the absolute value of the underlying return when that return stays "
        "within a corridor [-M, +M], and pays zero outside it. "
        "It profits from moderate moves in either direction but loses value in extreme moves — "
        "the opposite of a straddle. "
        "In a behavioural portfolio it is useful when low-volatility environments are expected, "
        "providing income from small fluctuations while the mental-account constraint limits tail risk."
    ),
    "Bull call spread (long call + short higher call)": (
        "A bull call spread is long a call at a lower strike and short a call at a higher strike. "
        "It gives bullish upside between the two strikes at a lower cost than a plain call, but caps "
        "the gain above the upper strike. In a behavioural portfolio it offers cheaper, capped "
        "participation in the underlying's rise while keeping the downside premium small."
    ),
    "Bear put spread (long put + short lower put)": (
        "A bear put spread is long a put at a higher strike and short a put at a lower strike. "
        "It provides downside protection (or a bearish bet) between the strikes at a lower cost than a "
        "single put, with the protection capped below the lower strike. It is a cheaper way to satisfy "
        "a mental-account downside constraint when only moderate declines are expected."
    ),
    "Long butterfly (calls)": (
        "A long call butterfly is long one call at a low strike, short two calls at a middle strike, and "
        "long one call at a high strike. It is a very low-cost bet that the underlying finishes near the "
        "middle strike (a 'pin'), paying its maximum there and expiring worthless in the wings. It adds "
        "a sharply peaked, positively skewed payoff to the portfolio at minimal premium."
    ),
    "Call condor": (
        "A call condor is long a call at a low strike, short two calls at two middle strikes, and long a "
        "call at a high strike. Like a butterfly but with a flat plateau of maximum payoff between the "
        "inner strikes, it bets on the underlying staying within a range, at low cost and with capped "
        "risk on both sides."
    ),
    "Reverse convertible (bond − short put)": (
        "A reverse convertible combines a zero-coupon bond with a short put on the underlying. It pays a "
        "high fixed coupon (the bond plus the put premium received) in exchange for taking the downside "
        "below the put strike — the mirror image of a capital-guaranteed note. Upside is capped at the "
        "coupon; principal is at risk if the underlying falls. A premium reflects the issuer's markup."
    ),
    "Discount certificate (capped underlying)": (
        "A discount certificate gives exposure to the underlying purchased at a discount (financed by "
        "selling a call), in exchange for capping the upside at the call strike. The discount provides a "
        "partial downside buffer. It is built from a synthetic long underlying plus a short call, and an "
        "optional premium reflects the issuer's markup."
    ),
    "Outperformance certificate (geared upside)": (
        "An outperformance (participation) certificate gives full exposure to the underlying plus extra "
        "geared participation above a strike (an additional long call), so it outperforms the underlying "
        "on the upside while retaining full downside. It is built from a synthetic long underlying plus "
        "an extra long call, with an optional premium for the issuer's markup."
    ),
    # ── Risk measures ─────────────────────────────────────────────────────────
    "Value at Risk (VaR)": (
        "Value at Risk (VaR) at level α is the return threshold H such that losses exceed H with "
        "probability at most α. For example, a 5% VaR of -10% means there is a 5% chance of "
        "losing more than 10%. "
        "In this app, the VaR constraint is the mental-account threshold: "
        "P(portfolio return < H) ≤ α. The optimizer finds the highest expected return portfolio "
        "satisfying this constraint."
    ),
    "Expected Shortfall (ES)": (
        "Expected Shortfall (ES), also called Conditional VaR or CVaR, measures the average loss "
        "in the worst α% of scenarios. Unlike VaR which only gives a threshold, ES captures the "
        "severity of losses beyond that threshold. "
        "In this app, the ES constraint requires E[return | return < H] ≥ L — "
        "the average loss in the tail must not be worse than L. "
        "ES is considered a more complete risk measure than VaR as it is coherent and convex."
    ),
    "Shortfall probability": (
        "The shortfall probability is P(portfolio return < H) — the probability that the portfolio "
        "return falls below the mental-account threshold H. "
        "It is the key output of the VaR constraint mode. "
        "The optimizer ensures this probability stays at or below α. "
        "A result of 4.4% with α=5% means the constraint is satisfied with 0.6% margin."
    ),
    "Skewness": (
        "Skewness measures the asymmetry of a return distribution. "
        "Positive skewness means occasional large gains and frequent small losses — "
        "preferred by investors. Negative skewness means occasional large losses. "
        "Derivatives like calls and CGNs add positive skewness to portfolio returns, "
        "which is one reason behavioral portfolios including them can outperform "
        "mean-variance portfolios that ignore higher moments."
    ),
    "Excess kurtosis": (
        "Excess kurtosis measures the fatness of the tails of a return distribution relative to "
        "a normal distribution. Positive excess kurtosis (leptokurtosis) means fatter tails — "
        "more extreme events than a normal distribution would predict. "
        "This is why mean-variance theory, which assumes normality, is insufficient for portfolios "
        "containing derivatives: options have highly non-normal payoff distributions."
    ),
    # ── Portfolio theory ──────────────────────────────────────────────────────
    "Mean-variance efficient frontier": (
        "The mean-variance efficient frontier, introduced by Markowitz (1952), is the set of portfolios "
        "that maximise expected return for a given level of variance (risk). "
        "It is the foundation of modern portfolio theory. "
        "However it assumes normally distributed returns and ignores higher moments — "
        "making it inadequate for portfolios containing derivatives. "
        "This app shows the MV frontier alongside behavioral frontiers to illustrate this limitation."
    ),
    "Markowitz optimization": (
        "Markowitz optimization solves: max w'μ - (λ/2) w'Σw subject to sum(w)=1, w≥0. "
        "The parameter λ is the risk-aversion coefficient. Higher λ penalises variance more, "
        "producing lower-risk portfolios. "
        "A key result in this app is that for any mental-account constraint (H, α), "
        "there exists an implied λ such that Markowitz and behavioral optimization yield "
        "identical portfolios — when no derivatives are present."
    ),
    "Mental accounting": (
        "Mental accounting, introduced by Richard Thaler (1985, 1999), describes how people "
        "mentally divide their money into separate 'accounts' according to its source or intended "
        "purpose, and then make decisions account-by-account rather than across their wealth as a "
        "whole. A windfall is spent more freely than salary; a dedicated 'holiday fund' is preserved "
        "even while carrying expensive debt. Money is treated as non-fungible, even though "
        "economically a dollar is a dollar."
        "<br><br>"
        "In <b>behavioural portfolio theory</b> (Shefrin &amp; Statman, 2000) this becomes a design "
        "principle: instead of holding one mean-variance portfolio, the investor holds a layered "
        "<b>pyramid</b> of sub-portfolios, each tied to a goal with its own risk tolerance — a "
        "<b>safety layer</b> that must not be lost, an <b>income layer</b>, and an "
        "<b>aspirational layer</b> reaching for upside."
        "<br><br>"
        "Das &amp; Statman (2009) formalise each mental account as a <b>downside constraint</b>: the "
        "investor sets a threshold return H and a maximum acceptable probability α of finishing below "
        "it, and the optimiser maximises expected return subject to that shortfall limit. Framed this "
        "way, derivatives become natural tools — a protective put defends the safety layer's "
        "threshold, while calls and structured notes add asymmetric upside to the aspirational layer. "
        "This is precisely why a behavioural portfolio that embraces mental accounts can dominate a "
        "single mean-variance portfolio that ignores them."
    ),
    "Optimum mental-accounting portfolio": (
        "The optimum mental-accounting portfolio is the best portfolio under a behavioural "
        "(mental-account) constraint: it maximises expected return subject to a downside limit — "
        "the probability of finishing below a threshold H must stay within α (VaR form), or the "
        "average shortfall must stay above a floor L (ES / CVaR form). "
        "<br><br>"
        "The Grid optimiser's 3D landscape makes it visual: each point is a candidate portfolio "
        "(the two axes are asset weights), and its height is the objective. The <b>grey region is "
        "infeasible</b> — those weights breach the downside limit — while the <b>coloured surface is "
        "the feasible region</b>. The optimum is the highest feasible point: it sits at the top edge "
        "of the coloured surface, right against the feasibility boundary. "
        "<br><br>"
        "Without derivatives it coincides with the Markowitz mean-variance optimum (the MVT ≡ MAT "
        "equivalence). Add a derivative — a protective put, a straddle — and the feasible region "
        "reshapes, letting the optimum climb to a higher expected return at the very same downside "
        "limit. That uplift, bought with skewness rather than free return, is the core idea of the framework."
    ),
    "Behavioral portfolio theory": (
        "Behavioral portfolio theory (BPT), developed by Shefrin & Statman (2000) and extended "
        "by Das & Statman (2009), integrates psychological insights into portfolio construction. "
        "Rather than maximising a utility function over total wealth, investors set safety-first "
        "constraints (mental accounts) and maximise expected return subject to them. "
        "BPT explains observed investor behaviour such as holding both lottery tickets and "
        "insurance, and provides a framework for including derivatives in optimal portfolios."
    ),
    "MVT/MAT equivalence": (
        "The MVT/MAT equivalence, proven in Das, Markowitz, Scheid & Statman (2010), shows that "
        "mean-variance theory (MVT) and mental-accounting theory (MAT) are equivalent "
        "when no derivatives are present. "
        "For any threshold H and shortfall probability α, there exists a unique implied "
        "risk-aversion coefficient λ such that both methods produce the same optimal portfolio. "
        "This equivalence breaks down when derivatives are added — the behavioral approach "
        "can then exploit asymmetric payoffs that mean-variance cannot capture."
    ),
    "Implied risk aversion lambda": (
        "The implied risk-aversion coefficient λ is the value such that the Markowitz optimal "
        "portfolio (maximising w'μ - (λ/2)w'Σw) is identical to the behavioral optimal portfolio "
        "under the mental-account constraint (H, α). "
        "This app computes λ dynamically as you adjust H and α in the sidebar. "
        "At H=-10% and α=5%, λ=3.795 for the default base case. "
        "Higher α (more risk tolerance) implies lower λ; tighter H implies higher λ."
    ),
    "Gaussian copula": (
        "A Gaussian copula models the dependence structure between assets independently of their "
        "marginal distributions. It maps each asset's returns through their individual CDFs to "
        "uniform scores, then models their joint dependence using a multivariate normal distribution. "
        "This allows non-normal marginal distributions (as produced by options) while still "
        "capturing realistic correlations between assets. "
        "This app uses a Gaussian copula in Step 2 to assign probabilities to the state space."
    ),
    "Black-Scholes pricing": (
        "The Black-Scholes model prices European options under assumptions of log-normal asset "
        "prices, constant volatility, and no arbitrage. "
        "The formula gives call price = S·N(d1) - K·e^(-rT)·N(d2) where d1 and d2 depend on "
        "spot price, strike, volatility, risk-free rate, and time to maturity. "
        "This app uses Black-Scholes to compute derivative payoffs in each scenario of the "
        "state space, enabling the optimizer to price the derivative's contribution to portfolio risk and return."
    ),
    # ── Academic references ───────────────────────────────────────────────────
    "Das & Statman (2009) — Beyond Mean-Variance": (
        "Das, Sanjiv and Meir Statman (2009) — 'Beyond Mean-Variance: Portfolios with Derivatives "
        "and Non-Normal Returns in Mental Accounts'. "
        "This paper introduces the core algorithm used in this app. It shows how to construct "
        "optimal portfolios including derivatives under a mental-accounting downside constraint, "
        "using a discrete state space with Gaussian copula probabilities and a grid search optimizer. "
        "It demonstrates that including derivatives can improve "
        "portfolio expected returns while satisfying the same downside constraint."
    ),
    "Das, Markowitz, Scheid & Statman (2010) JFQA": (
        "Das, Sanjiv, Harry Markowitz, Jonathan Scheid and Meir Statman (2010) — "
        "'Portfolio Optimization with Mental Accounts', Journal of Financial and Quantitative Analysis, "
        "Vol. 45, No. 2, pp. 311-334. "
        "This paper proves the MVT/MAT equivalence: for any mental-account constraint (H, α), "
        "there exists an implied risk-aversion λ such that Markowitz mean-variance optimization "
        "and behavioral optimization produce identical portfolios when no derivatives are present. "
        "This is the theoretical foundation for the equivalence point shown on the frontier chart."
    ),
    "Jeddou (2012) MSc thesis USI Lugano": (
        "Jeddou, Sami (2012) — 'Beyond Mean-Variance: Options and Structured Products in "
        "Behavioral Portfolios', MSc Finance Thesis, Università della Svizzera italiana (USI Lugano), "
        "supervised by Prof. Enrico De Giorgi. "
        "This thesis implements the full Das & Statman (2009) algorithm in R and extends it to "
        "all major derivative and structured product types: puts, calls, safety and aggressive collars, "
        "straddles, strangles, capital-guaranteed notes (capped and uncapped), and barrier-M notes. "
        "This app is a Python reimplementation and extension of that work, adding further "
        "instruments built on the same Black-Scholes pricing principle: bull and bear spreads, "
        "a long butterfly and call condor, a reverse convertible, and discount and outperformance certificates."
        "<br><br><a href='https://thesis.bul.sbu.usi.ch/theses/1012-1112BenJeddou/pdf?1390987439' "
        "target='_blank' style='color:#E3C77E;font-weight:600;text-decoration:none'>"
        "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='#E3C77E' stroke-width='2' "
        "stroke-linecap='round' stroke-linejoin='round' style='vertical-align:-2px;margin-right:.35rem'>"
        "<path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/>"
        "<polyline points='14 2 14 8 20 8'/><line x1='16' y1='13' x2='8' y2='13'/>"
        "<line x1='16' y1='17' x2='8' y2='17'/></svg>Read the full thesis (PDF) →</a>"
    ),
    "Rockafellar & Uryasev (2000) — Optimization of CVaR": (
        "Rockafellar, R. Tyrrell and Stanislav Uryasev (2000) — 'Optimization of Conditional "
        "Value-at-Risk', Journal of Risk, Vol. 2, No. 3, pp. 21-41 (generalised to arbitrary "
        "loss distributions in Rockafellar & Uryasev (2002), Journal of Banking & Finance, "
        "Vol. 26, No. 7, pp. 1443-1471). "
        "This paper shows that Conditional Value-at-Risk (CVaR, also called Expected Shortfall) "
        "can be minimised — or held above a floor — through a convex linear program, using one "
        "auxiliary variable for the VaR level and one slack variable per scenario, without having "
        "to estimate VaR first. The scalable Monte-Carlo engine in this app uses exactly this "
        "formulation: it samples joint return and payoff scenarios through a copula and solves the "
        "goal as an LP, so the cost grows linearly in the number of assets rather than as the exact "
        "grid's m^N state space."
    ),
    "Capital-guaranteed note (CGN)": (
        "A capital-guaranteed note combines a zero-coupon bond, which secures a protected floor at "
        "maturity, with a long option that provides upside participation. The investor gives up some "
        "upside — through a participation rate below 100% or a cap — in exchange for that downside "
        "protection. This app prices both an uncapped version (full participation above the floor) "
        "and a capped version (participation up to a ceiling), each built from Black-Scholes legs."
    ),
    "Digital option": (
        "A digital (binary) option pays a fixed amount if the underlying finishes beyond a strike and "
        "nothing otherwise — an all-or-nothing payoff rather than the linear payoff of a vanilla "
        "option. Because the payoff jumps at the strike, its value is highly sensitive to the "
        "underlying near expiry. In this app it serves as a building block for custom structured "
        "products priced on the same Black-Scholes engine."
    ),
    "Zero-coupon bond": (
        "A zero-coupon bond pays no interim coupons; it is bought at a discount and repays its face "
        "value at maturity, so its whole return comes from that price-to-face appreciation. It is the "
        "riskless building block of a capital-guaranteed note: the bond secures the protected floor "
        "and the remaining budget buys the option that supplies the upside. In this app it is valued "
        "by discounting the face value at the risk-free rate."
    ),
    "Alpha (Jensen's alpha)": (
        "Alpha is the return a portfolio earns beyond what its market exposure alone would justify. "
        "In this app's back-test it is Jensen's alpha: each holding's excess returns (over the "
        "risk-free rate) are regressed on the benchmark's excess returns, and the annualised "
        "intercept is the alpha. A positive alpha means the holding out-performed the return its beta "
        "predicted; a negative alpha means it under-performed. Because it is measured over the "
        "evaluation window, it is a realised (ex-post) alpha, not a forecast."
    ),
    "Beta": (
        "Beta measures how strongly a holding moves with the benchmark. A beta of 1.0 moves "
        "one-for-one with the market; below 1.0 is more defensive, above 1.0 more aggressive, and a "
        "negative beta moves opposite the market. It is the slope of the regression of the holding's "
        "excess returns on the benchmark's excess returns over the evaluation window. The portfolio "
        "beta is measured directly from the realised buy-and-hold path, so a derivative's changing "
        "(non-linear) sensitivity is captured rather than approximated by a weighted average."
    ),
    "R-squared (R²)": (
        "R-squared is the fraction of a holding's return variation that the benchmark explains, from "
        "0 (no relationship) to 1 (fully explained). It signals how trustworthy the alpha and beta "
        "are: a high R² means the benchmark is a good reference and beta is reliable, while a low R² "
        "means much of the return is unrelated to that benchmark, so the alpha and beta should be "
        "read with caution. Short back-test windows tend to give noisier, lower-R² estimates."
    ),
    "Benchmark": (
        "A benchmark is the reference market that alpha and beta are measured against. The choice "
        "matters: a single equity index such as the S&P 500 suits an equity-heavy portfolio, while a "
        "global index (ACWI) or a 60/40 stock-bond blend is fairer for multi-asset, bond, gold or "
        "crypto mixes. Against a poorly matched benchmark, betas can look low or odd for the "
        "non-equity sleeves. In this app you can pick the S&P 500, global ACWI, a 60/40 SPY-AGG "
        "blend, or any Yahoo Finance ticker."
    ),
    "CAPM (Capital Asset Pricing Model)": (
        "The Capital Asset Pricing Model links a holding's expected return to its market sensitivity: "
        "required return = risk-free rate + beta × (expected market return − risk-free rate), where "
        "the bracketed term is the equity risk premium. In this app CAPM is optional and used only if "
        "you supply an expected market return E[Rₘ]: it then shows each holding's CAPM required "
        "return and an ex-ante (expected) alpha = the model's expected return minus that required "
        "return. The realised alpha and beta never depend on this forecast."
    ),
    "Excess return": (
        "An excess return is a return measured above the risk-free rate (return − risk-free rate). "
        "Alpha and beta are computed on excess returns rather than raw returns, because CAPM "
        "describes how investors are compensated for risk relative to a safe asset. In this app the "
        "risk-free rate is an input you set, and it defines the excess returns used in the alpha/beta "
        "regression."
    ),
    "Risk-free rate": (
        "The risk-free rate is the return on a (theoretically) riskless asset, such as a short-term "
        "government bill, and it is the baseline from which excess returns — and therefore alpha and "
        "beta — are measured. In this app it is an annual input; the per-period rate is that annual "
        "rate divided by the number of periods per year. Raising or lowering it shifts the excess "
        "returns and slightly changes the alpha; setting it to 0% simply uses raw returns."
    ),
    "α-CVaR (Conditional VaR)": (
        "Conditional Value-at-Risk (CVaR), also called Expected Shortfall, is the average loss in the "
        "worst α% of outcomes — it captures how bad the tail is, not just how often a threshold is "
        "breached. The scalable Monte-Carlo engine uses the α-CVaR form (the average loss beyond the "
        "α-quantile) because it can be minimised with a linear program. This differs subtly from the "
        "exact grid's fixed-threshold ES (the average loss below a chosen H): the two coincide only "
        "when H equals the α-quantile."
    ),
    "Monte-Carlo scenario generation": (
        "Monte-Carlo scenario generation samples a large number of possible joint outcomes for the "
        "assets and their derivative payoffs, instead of enumerating a fixed grid of states. The "
        "scalable engine draws thousands of correlated return scenarios through a copula, prices "
        "every derivative under each scenario, and optimises over that scenario set. Because cost "
        "grows with the number of scenarios rather than exponentially with the number of assets, it "
        "scales to large, multi-derivative portfolios the exact grid cannot handle."
    ),
    "Student-t copula": (
        "A copula joins individual asset distributions into one joint distribution while preserving "
        "each asset's own (possibly non-normal) shape. A Student-t copula has heavier joint tails "
        "than a Gaussian copula, so extreme moves are more likely to occur together — it captures the "
        "tendency of assets to crash at the same time. The scalable engine offers both Gaussian and "
        "Student-t copulas for generating its Monte-Carlo scenarios."
    ),
    "Rockafellar–Uryasev CVaR linear program": (
        "Rockafellar and Uryasev (2000) showed that minimising CVaR can be cast as a linear program "
        "by adding one variable for the value-at-risk level and one slack variable per scenario. The "
        "scalable engine uses this formulation, solved with a fast LP solver, to maximise expected "
        "return subject to a CVaR (Expected-Shortfall) floor. It is what lets the engine handle many "
        "assets and several derivatives at once while keeping the problem convex and quick to solve."
    ),
    "Common random numbers (CRN)": (
        "Common random numbers means reusing the same set of Monte-Carlo scenarios across every point "
        "on the efficient frontier. Because all points are evaluated on identical draws, the "
        "differences between them reflect the portfolios themselves rather than sampling noise, so "
        "the frontier comes out smooth and the points are directly comparable."
    ),
    "Out-of-sample back-test": (
        "An out-of-sample back-test checks whether an optimised portfolio behaves as expected on data "
        "it was not built on. This app builds the optimal weights on a construction window, then "
        "buy-and-holds those fixed weights through a separate, later evaluation window — with any "
        "derivative marked to market — and compares expected versus realised return, volatility and "
        "the loss-threshold outcome. It also reports the realised alpha and beta of each holding and "
        "of the portfolio against a chosen benchmark, testing the efficiency of the optimisation "
        "methods rather than just their in-sample fit."
    ),
    "Construction vs evaluation window": (
        "The construction window is the historical period used to estimate means, volatilities and "
        "correlations and to build the optimal weights. The evaluation window is a separate, later "
        "period over which those fixed weights are held and measured. Keeping the two apart is what "
        "makes the test out-of-sample: the model never sees the evaluation data when it chooses the "
        "portfolio."
    ),
    "Mark-to-market": (
        "Marking to market means revaluing a position at current prices at each point in time, rather "
        "than only at maturity. In the back-test the derivative is repriced with Black-Scholes at the "
        "current spot, the shrinking remaining maturity and the volatility assumption, so it has a "
        "value at every date. This gives the option its own return series, which is needed to compute "
        "realised portfolio volatility and the distributional loss-threshold check — a "
        "held-to-maturity payoff would give only a single end-point number."
    ),
    "Buy-and-hold": (
        "Buy-and-hold means fixing the portfolio weights at the start and holding them without "
        "rebalancing. In the back-test the weights chosen on the construction window are held through "
        "the evaluation window and allowed to drift naturally with prices. This isolates the quality "
        "of the original optimisation decision, with no transaction costs or rebalancing rules mixed "
        "in."
    ),
    "Rigorous Expected Shortfall (beyond thesis)": (
        "The thesis's Expected-Shortfall method enforces the ES floor only when seeding the grid, "
        "while its refinement step still targets the VaR penalty — so the final portfolio can drift "
        "below the ES limit. The rigorous-ES mode instead optimises with a genuinely ES-aware "
        "objective, so the result stays ES-feasible. In tests it recovers expected return the thesis "
        "method leaves unused (for example about +2.4 percentage points at L = −15%). It runs at high "
        "precision (m = 51) with a fast coarse-to-fine search."
    ),
    "Turbo solver (coarse-to-fine)": (
        "Turbo reproduces High-precision results for the VaR constraint but replaces the exhaustive "
        "weight-grid search with a coarse-to-fine search plus pruning of negligible states. It runs "
        "in seconds instead of 15–30 minutes, matching High precision to within about 0.1 percentage "
        "point of expected return. It is VaR-only and limited to four or fewer securities with no "
        "derivative — Expected-Shortfall and larger problems fall back to the standard solver — which "
        "is why it is not offered in the back-test."
    ),
    "Differential evolution": (
        "Differential evolution is a global, population-based stochastic optimiser. The app uses it "
        "for larger problems (five or more securities), where the exhaustive weight grid would be too "
        "expensive, because it searches the weight space efficiently without enumerating every "
        "combination. For four or fewer securities the app uses the exact grid search instead."
    ),
}

def get_explanation(term):
    """Look up explanation from static dictionary. No API call."""
    return EXPLANATIONS.get(term,
        f"No pre-written explanation available for '{term}'. "
        "Please use the custom question box below for AI-generated answers.")

def get_ai_chat_response(question, portfolio_context=""):
    """Get AI response for custom questions via Anthropic API."""
    try:
        _key = _anthropic_key()
        if not _key:
            return None
        import anthropic
        client = anthropic.Anthropic(api_key=_key)
        system = (
            "You are a financial expert assistant embedded in a behavioral portfolio optimization app. "
            "Give clear, concise answers in 3-5 sentences. Focus on portfolio optimization context.")
        paper_ctx = (
            "The app is based on: Das & Statman (2009) Beyond Mean-Variance; "
            "Das, Markowitz, Scheid & Statman (2010) JFQA MVT/MAT equivalence; "
            "Jeddou (2012) MSc thesis USI Lugano.")
        prompt = f"{paper_ctx}\n{f'Portfolio context: {portfolio_context}' if portfolio_context else ''}\n\nQuestion: {question}"
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception:
        return None


def _anthropic_key():
    """Return the Anthropic key from env or a secrets.toml, WITHOUT triggering
    Streamlit's 'No secrets found' UI error when no secrets file exists locally.
    (Streamlit Cloud writes a secrets.toml when secrets are configured, so the
    file check stays cloud-safe.)"""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    _paths = [os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
              os.path.expanduser("~/.streamlit/secrets.toml")]
    if any(os.path.exists(p) for p in _paths):
        try:
            return st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            return None
    return None


def get_ratio_explanation_ai(ticker, label, value):
    """Neutral, educational AI explanation of ONE financial ratio for a ticker.
    Strictly no advice / no valuation verdict. Returns None if unavailable."""
    try:
        _key = _anthropic_key()
        if not _key:
            return None
        import anthropic
        client = anthropic.Anthropic(api_key=_key)
        system = (
            "You explain a single financial metric (a ratio or a reported figure) in plain language for a "
            "non-expert, neutrally and educationally. Say what the metric measures, how to read it, and add "
            "one caveat or peer/industry-context note. STRICT RULES — you must never: give investment advice; "
            "express a buy, sell or hold view; give a price target; call the company or the number cheap, "
            "expensive, good, bad, attractive or a concern; or make a suitability judgement. Stay purely "
            "descriptive. 3–5 sentences, no preamble.")
        _vtxt = f", whose current value is {value}" if value and str(value) != "—" else ""
        prompt = (f"Explain the financial metric '{label}' for {str(ticker).upper()}{_vtxt}. "
                  f"Describe it neutrally — do not judge whether the value is good or bad.")
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=320,
            system=system,
            messages=[{"role": "user", "content": prompt}])
        return message.content[0].text
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=900)
def _cached_ticker_info(symbol):
    """Cached yfinance fundamentals fetch (15-min TTL) for the ticker-analytics view."""
    return fetch_ticker_info(symbol)


@st.cache_data(show_spinner=False, ttl=900)
def _cached_ticker_history(symbol, period):
    return fetch_ticker_history(symbol, period)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ticker_financials(symbol):
    return fetch_ticker_financials(symbol)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ticker_cashflow(symbol):
    return fetch_ticker_cashflow(symbol) if fetch_ticker_cashflow else ([], None)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ticker_balance(symbol):
    return fetch_ticker_balance(symbol) if fetch_ticker_balance else ({}, None)


def _tk_margins_fig(rows):
    """Gross / operating / net margin (%) over time. Descriptive."""
    import plotly.graph_objects as _go
    yrs = [r["year"] for r in rows]

    def _pct(key):
        out = []
        for r in rows:
            rev, v = r.get("revenue"), r.get(key)
            out.append((v / rev * 100.0) if (rev and v is not None and rev != 0) else None)
        return out

    fig = _go.Figure()
    for nm, key, col in (("Gross", "gross_profit", "#4a9eff"),
                         ("Operating", "operating_income", "#f5b942"),
                         ("Net", "net_income", "#26a641")):
        fig.add_trace(_go.Scatter(x=yrs, y=_pct(key), mode="lines+markers", name=nm,
                                  line=dict(color=col, width=2), marker=dict(size=6),
                                  hovertemplate="%{x}<br>" + nm + " margin %{y:.1f}%<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, margin=dict(l=10, r=10, t=34, b=10),
        xaxis=dict(type="category", gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title="Margin", ticksuffix="%"),
        legend=dict(orientation="h", y=1.09, x=0.5, xanchor="center", font=dict(color="#c9d1d9")))
    return fig


def _tk_cashflow_fig(rows, cur):
    """Operating & free cash-flow bars (billions), labelled."""
    import plotly.graph_objects as _go
    yrs = [r["year"] for r in rows]
    o = [(r["ocf"] / 1e9 if r["ocf"] is not None else None) for r in rows]
    f = [(r["fcf"] / 1e9 if r["fcf"] is not None else None) for r in rows]

    def _lbl(vals):
        return ["" if v is None else (f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}") for v in vals]

    fig = _go.Figure()
    fig.add_trace(_go.Bar(x=yrs, y=o, name="Operating CF", marker_color="#4a9eff",
                          text=_lbl(o), textposition="outside", cliponaxis=False,
                          textfont=dict(color="#9ec5ff", size=10)))
    fig.add_trace(_go.Bar(x=yrs, y=f, name="Free CF", marker_color="#26a641",
                          text=_lbl(f), textposition="outside", cliponaxis=False,
                          textfont=dict(color="#86e0b0", size=10)))
    fig.update_layout(
        barmode="group", bargap=0.42, bargroupgap=0.12, template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, margin=dict(l=10, r=10, t=34, b=10),
        xaxis=dict(type="category", gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title=f"Billions{(' ' + cur) if cur else ''}"),
        legend=dict(orientation="h", y=1.09, x=0.5, xanchor="center", font=dict(color="#c9d1d9")),
        uniformtext=dict(mode="hide", minsize=8))
    return fig


def _tk_balance_fig(bal, cur):
    """Balance-sheet identity as two equal-length stacked bars (Assets = Liabilities + Equity):
    Assets = Cash + Other assets; Liab. & equity = Debt + Other liabilities + Equity. Both sum
    to total assets. Falls back to simple per-item bars when assets/equity are unavailable."""
    import plotly.graph_objects as _go
    _B = 1e9
    ta = bal.get("total_assets")
    eq = bal.get("equity")
    debt = bal.get("total_debt")
    cash = bal.get("cash")
    _axis = f"Billions{(' ' + cur) if cur else ''}"

    if ta is not None and eq is not None and ta > 0:
        _debt = debt or 0.0
        _cash = cash or 0.0
        other_assets = max(ta - _cash, 0.0)
        other_liab = max(ta - eq - _debt, 0.0)

        def _seg(row, val, name, color):
            x = val / _B
            return _go.Bar(y=[row], x=[x], name=name, orientation="h", marker_color=color,
                           text=[(f"{x:.0f}" if x >= 10 else f"{x:.1f}") if x >= 0.05 else ""],
                           textposition="inside", insidetextanchor="middle",
                           textfont=dict(color="#0d1117", size=10),
                           hovertemplate="%{fullData.name}: %{x:.1f}B<extra></extra>")

        fig = _go.Figure()
        fig.add_trace(_seg("Assets", _cash, "Cash", "#26a641"))
        fig.add_trace(_seg("Assets", other_assets, "Other assets", "#4a9eff"))
        fig.add_trace(_seg("Liab. &amp; equity", _debt, "Debt", "#d15866"))
        fig.add_trace(_seg("Liab. &amp; equity", other_liab, "Other liabilities", "#7d8aa0"))
        fig.add_trace(_seg("Liab. &amp; equity", eq, "Equity", "#f5b942"))
        fig.update_layout(
            barmode="stack", bargap=0.78, template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=300, margin=dict(l=10, r=20, t=20, b=10),
            xaxis=dict(gridcolor="#1e2130", title=_axis),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=1.16, x=0.5, xanchor="center",
                        font=dict(color="#c9d1d9", size=10)),
            uniformtext=dict(mode="hide", minsize=8))
        return fig

    # Fallback — simple per-item bars
    items = [("Total assets", ta, "#4a9eff"), ("Total debt", debt, "#d15866"),
             ("Cash", cash, "#26a641"), ("Equity", eq, "#f5b942")]
    items = [(n, v, c) for n, v, c in items if v is not None]
    fig = _go.Figure(_go.Bar(
        y=[n for n, _, _ in items], x=[v / _B for _, v, _ in items], orientation="h",
        marker_color=[c for _, _, c in items],
        text=[f"{v / _B:.0f}" if abs(v / _B) >= 10 else f"{v / _B:.1f}" for _, v, _ in items],
        textposition="outside", cliponaxis=False, textfont=dict(color="#c9d1d9", size=10),
        hovertemplate="%{y}: %{x:.1f}B<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, margin=dict(l=10, r=20, t=20, b=10), showlegend=False,
        xaxis=dict(gridcolor="#1e2130", title=_axis), yaxis=dict(autorange="reversed"))
    return fig


def _tk_price_fig(df, name):
    """Descriptive price-history line chart (dark theme). No signals/annotations."""
    import plotly.graph_objects as _go
    fig = _go.Figure()
    fig.add_trace(_go.Scatter(
        x=df.index, y=df["Close"], mode="lines", line=dict(color="#4a9eff", width=1.8),
        fill="tozeroy", fillcolor="rgba(74,158,255,0.08)", name="Close",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>"))
    _lo = float(df["Close"].min()); _hi = float(df["Close"].max())
    _pad = (_hi - _lo) * 0.08 or 1.0
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
        title=dict(text=f"{name} — price history", x=0.5, font=dict(color="#E3C77E", size=14)),
        xaxis=dict(gridcolor="#1e2130", showspikes=True, spikethickness=1, spikecolor="#3a3a5a"),
        yaxis=dict(gridcolor="#1e2130", title="Price", range=[max(0, _lo - _pad), _hi + _pad]))
    return fig


def _tk_candle_fig(df, name):
    """Descriptive OHLC candlestick chart (dark theme). No signals/annotations."""
    import plotly.graph_objects as _go
    fig = _go.Figure(_go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#26a641", decreasing_line_color="#d15866",
        increasing_fillcolor="#26a641", decreasing_fillcolor="#d15866", name="OHLC",
        whiskerwidth=0.4))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
        title=dict(text=f"{name} — candlestick", x=0.5, font=dict(color="#E3C77E", size=14)),
        xaxis=dict(gridcolor="#1e2130", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#1e2130", title="Price"))
    return fig


def _tk_revenue_fig(rows, cur):
    """Annual revenue & net-income bars (descriptive). Values in billions, labelled on the bars."""
    import plotly.graph_objects as _go
    yrs = [r["year"] for r in rows]
    rev = [(r["revenue"] / 1e9 if r["revenue"] is not None else None) for r in rows]
    ni = [(r["net_income"] / 1e9 if r["net_income"] is not None else None) for r in rows]

    def _lbl(vals):
        return ["" if v is None else (f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}") for v in vals]

    fig = _go.Figure()
    fig.add_trace(_go.Bar(x=yrs, y=rev, name="Revenue", marker_color="#4a9eff",
                          text=_lbl(rev), textposition="outside", cliponaxis=False,
                          textfont=dict(color="#9ec5ff", size=10),
                          hovertemplate="%{x}<br>Revenue %{y:.2f}B<extra></extra>"))
    fig.add_trace(_go.Bar(x=yrs, y=ni, name="Net income", marker_color="#26a641",
                          text=_lbl(ni), textposition="outside", cliponaxis=False,
                          textfont=dict(color="#86e0b0", size=10),
                          hovertemplate="%{x}<br>Net income %{y:.2f}B<extra></extra>"))
    fig.update_layout(
        barmode="group", bargap=0.42, bargroupgap=0.12, template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, margin=dict(l=10, r=10, t=34, b=10),
        xaxis=dict(type="category", gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title=f"Billions{(' ' + cur) if cur else ''}"),
        legend=dict(orientation="h", y=1.09, x=0.5, xanchor="center", font=dict(color="#c9d1d9")),
        uniformtext=dict(mode="hide", minsize=8))
    return fig


DEFAULT_MEANS = [0.05, 0.10, 0.25]
DEFAULT_SIGS  = [0.05, 0.20, 0.50]
DEFAULT_CORR  = [[1.0,0.0,0.0],[0.0,1.0,0.4],[0.0,0.4,1.0]]
DEFAULT_NAMES = ["Sec 1 — Low risk","Sec 2 — Mid risk","Sec 3 — High risk"]

GRID_OPTIONS = {
    "Turbo (High-precision accuracy, ~seconds)": (51,'turbo'),
    "Fast (m=21, m'=15)":           (21,15),
    "Standard (m=35, m'=50)":      (35,50),
    "High precision (m=51, m'=99)": (51,99),
}

GRID_EXPLANATIONS = {
    "Fast (m=21, m'=15)": (
        "Uses a coarse grid of 21 return scenarios per security and 15 weight steps per dimension. "
        "Runs quickly — seconds for a small universe. Results are directionally correct and useful for exploring "
        "parameters, but weights and expected returns may differ from the precise solution by "
        "a few percentage points. Recommended for initial exploration and parameter sensitivity testing."
    ),
    "Standard (m=35, m'=50)": (
        "Uses a medium grid with 35 return scenarios and 50 weight steps per dimension. "
        "Minutes-scale on a full local run (longer with more securities or derivatives). Provides a good balance between speed and accuracy — "
        "results are close to the precise solution in most cases. "
        "Recommended for most use cases once you have identified the right parameters."
    ),
    "High precision (m=51, m'=99)": (
        "Matches the original thesis parameters exactly — 51 return scenarios and 99 weight steps, "
        "the same values used in Das & Statman (2009) and Jeddou (2012). "
        "Results are publication-quality and directly comparable to academic benchmarks. "
        "Minutes to tens of minutes on a full local run (longer with more securities or derivatives). "
        "Recommended for final results and for verifying the equivalence point."
    ),
    "Turbo (High-precision accuracy, ~seconds)": (
        "Reproduces High-precision results (the m=51 state space and m'=99 weight resolution) "
        "for the VaR constraint, but replaces the exhaustive weight-grid search with a "
        "coarse-to-fine search plus pruning of negligible states. Runs in a few seconds instead "
        "of 15–30 minutes, matching High precision to within ~0.1 percentage point of expected "
        "return. Expected-Shortfall and 5+-security problems automatically use the standard solver. "
        "Recommended for fast, publication-quality VaR results."
    ),
    "Rigorous ES — high-precision accuracy (~seconds)": (
        "Rigorous ES does not use the resolution grid you would choose for VaR. It runs on the "
        "high-precision m=51 state space with a fast coarse-to-fine weight search, delivering "
        "high-precision-grade accuracy in a few seconds. The setting is fixed for this mode."
    ),
}

BENCHMARK_EXPLANATION = (
    "Alpha and beta measure your portfolio against a benchmark — the \u201cmarket\u201d you choose "
    "here. Beta is how strongly the portfolio moves with that benchmark (1.0 = moves one-for-one; "
    "below 1.0 = more defensive); alpha is the return earned beyond what that beta exposure alone "
    "would explain. They are computed by regressing each holding's excess returns (over the "
    "risk-free rate) on the benchmark's excess returns across the evaluation window: beta is the "
    "regression slope, alpha its annualised intercept (Jensen's alpha), and R\u00b2 shows how well "
    "the benchmark explains the holding. "
    "Pick a benchmark that fits the portfolio: a single equity index such as the S&P 500 suits an "
    "equity-heavy book, while a global index (ACWI) or a 60/40 blend is fairer for multi-asset, "
    "bond, gold or crypto mixes — alpha and beta are only as meaningful as the benchmark behind "
    "them, and a single equity index will show low or odd betas for non-equity sleeves. The "
    "risk-free rate sets the excess returns used in the regression. "
    "Leave the expected market return E[R\u2098] blank to report realised alpha and beta only. If "
    "you enter a value, a CAPM view is added: a required return = rf + beta \u00d7 (E[R\u2098] \u2212 rf) "
    "and an ex-ante (expected) alpha = the model's expected return minus that required return. "
    "The forecast affects only that ex-ante column — never the realised figures."
)

CONSTRAINT_EXPLANATIONS = {
    "var": (
        "The **Value at Risk (VaR) constraint** requires that the probability of the portfolio "
        "return falling below the threshold H does not exceed α. "
        "Formally: P(return < H) ≤ α. "
        "For example, with H = -10% and α = 5%, the optimizer finds the highest expected return "
        "portfolio where there is at most a 5% chance of losing more than 10%. "
        "A key theoretical result: this constraint is equivalent to a Markowitz portfolio with "
        "an implied risk-aversion coefficient λ — shown dynamically below the sliders."
    ),
    "es": (
        "The **Expected Shortfall (ES) constraint** — also called Conditional VaR (CVaR) — "
        "requires that the average portfolio return in the worst scenarios (those where return "
        "falls below H) is at least L. "
        "Formally: E[return | return < H] ≥ L. "
        "ES captures the severity of losses beyond the threshold, not just their probability, "
        "making it a more complete risk measure than VaR. "
        "It is a coherent risk measure and is preferred by regulators under Basel III/IV. "
        "For example, with H = -10% and L = -15%, the optimizer ensures that when losses exceed "
        "10%, their average is no worse than 15%."
    ),
    "es_rigorous": (
        "**Rigorous Expected Shortfall (beyond thesis).** Maximises expected return subject to "
        "E[return | return < H] ≥ L, enforcing the ES limit *in the final optimisation itself*. "
        "The standard ES option reproduces the thesis exactly — its refinement targets the VaR "
        "boundary, so the resulting ES can drift past L. This corrected mode keeps the portfolio "
        "ES-feasible and will usually report a higher, feasible expected return for the same H and L. "
        "Use it for decision-making; use standard ES to reproduce the thesis."
    ),
}

PREDEFINED_DERIVATIVES = {
    "None — primary securities only":               None,
    "Put option":                                    "put",
    "Call option":                                   "call",
    "Safety collar (long put + short call)":         "safety_collar",
    "Aggressive collar (long call + short put)":     "aggressive_collar",
    "Straddle (long call + long put)":               "straddle",
    "Strangle (long call + long put, diff strikes)": "strangle",
    "Capital-guaranteed note — uncapped":            "cgn_uncapped",
    "Capital-guaranteed note — capped":              "cgn_capped",
    "Barrier-M note":                                "barrier_m",
    "Bull call spread (long call + short higher call)": "bull_call_spread",
    "Bear put spread (long put + short lower put)":   "bear_put_spread",
    "Long butterfly (calls)":                         "butterfly_call",
    "Call condor":                                    "condor_call",
    "Reverse convertible (bond − short put)":         "reverse_convertible",
    "Discount certificate (capped underlying)":       "discount_certificate",
    "Outperformance certificate (geared upside)":     "outperformance_certificate",
    "🔧 Custom structured product":                  "custom",
}

COMPONENT_TYPES = [
    "long_call","short_call","long_put","short_put",
    "long_digital_call","short_digital_call",
    "long_digital_put","short_digital_put","zcb"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
# corr_to_cov -> moved to core/markets.py

# clean_returns -> moved to core/markets.py

# parse_csv -> moved to core/markets.py

# fetch_tickers -> moved to core/markets.py

# Synthetic long underlying via put-call parity (prices to exactly 1.0, payoff = spot_T)
# _SYN_UNDERLYING -> moved to core/pricing.py
# _COMPONENT_PRESETS -> moved to core/pricing.py

# preset_components -> moved to core/pricing.py

# build_der_config -> moved to core/pricing.py

@st.cache_data
def compute_mv_frontier(means_t, cov_t):
    means = np.array(means_t); cov = np.array(cov_t); n = len(means)
    def mv_opt(lam):
        def obj(w): return -(w@means-(lam/2)*(w@cov@w))
        cons=[{"type":"eq","fun":lambda w:w.sum()-1}]; bounds=[(0,1)]*n
        best=None
        for x0 in [np.ones(n)/n]+[np.eye(n)[i]*0.6+np.ones(n)*0.4/n for i in range(n)]:
            r=minimize(obj,x0,method="SLSQP",bounds=bounds,constraints=cons)
            if r.success and (best is None or r.fun<best.fun): best=r
        if best is None: return None
        w=best.x
        return float(np.sqrt(w@cov@w))*100, float(w@means)*100
    pts=[mv_opt(l) for l in np.concatenate([np.linspace(0.5,3,40), np.linspace(3,25,60), np.linspace(25,200,40)])]
    pts=[p for p in pts if p]
    # Sort by std dev ascending so line draws left to right
    pts=sorted(set(pts), key=lambda p: p[0])
    eq=mv_opt(3.7950)
    return [p[0] for p in pts],[p[1] for p in pts],eq

# run_opt -> moved to core/grid.py

# build_frontier -> moved to core/grid.py

def plot_frontier_plotly(mv_x, mv_y, mv_eq,
                         nd_x, nd_y, nd_lbls,
                         der_x, der_y, der_lbls,
                         der_label, H_sel, alpha,
                         p3_x=None, p3_y=None,
                         nd_res_actual=None, lam_actual=None, L=None, mv_eq_lam_str='',
                         benchmarks=None):
    """Interactive Plotly version of the frontier chart with hover tooltips."""
    fig = go.Figure()

    # ── Mean-variance frontier ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=mv_x, y=mv_y, mode='lines',
        name='Mean-variance efficient frontier (Markowitz) — no derivative',
        legendrank=1,
        line=dict(color='#a855f7', width=2, dash='dash'),
        hovertemplate='<b>Mean-Variance Efficient Frontier (Markowitz)</b><br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — no derivative ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=nd_x, y=nd_y, mode='lines+markers',
        name='Behavioural efficient frontier — no derivative',
        legendrank=2,
        line=dict(color='#1a6bbf', width=2.5),
        marker=dict(size=9, color='#1a6bbf', symbol='circle'),
        text=nd_lbls,
        hovertemplate='<b>Behavioral (no derivative)</b><br>Threshold: %{text}<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — with derivative ──────────────────────────────────────────
    if der_x:
        fig.add_trace(go.Scatter(
            x=der_x, y=der_y, mode='markers',
            name=f'Behavioural optimum portfolios — derivative frontier ({der_label})',
            legendrank=3,
            marker=dict(size=8, color='#f59e0b', symbol='square'),
            opacity=0.7,
            text=der_lbls,
            hovertemplate=f'<b>Behavioural optimal portfolio (with {der_label})</b><br>Threshold: %{{text}}<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))

        # ── Portfolio (2) highlighted point at selected H ─────────────────────
        if der_x and der_y and der_lbls:
            try:
                _target_h = f'H={H_sel:.0%}'
                if _target_h in der_lbls:
                    _i2 = der_lbls.index(_target_h)
                else:
                    _i2 = 0
                    _best = float('inf')
                    for _ii, _lbl in enumerate(der_lbls):
                        try:
                            _hv = float(_lbl.replace('H=','').replace('%','')) / 100
                            if abs(_hv - H_sel) < _best:
                                _best = abs(_hv - H_sel)
                                _i2 = _ii
                        except Exception:
                            pass
                fig.add_trace(go.Scatter(
                    x=[der_x[_i2]], y=[der_y[_i2]], mode='markers',
                    name=f'Portfolio (2) — behavioural optimum with {der_label} at H={H_sel:.0%}',
                    legendrank=6,
                    marker=dict(size=14, color='#ff6b00', symbol='square',
                               line=dict(color='white', width=1.5)),
                    hovertemplate=(f'<b>Portfolio (2)</b><br>Behavioural optimum with {der_label}<br>'
                                  f'Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>')
                ))
                fig.add_annotation(
                    x=der_x[_i2], y=der_y[_i2],
                    ax=-90, ay=-70,
                    xref='x', yref='y', axref='pixel', ayref='pixel',
                    showarrow=True, arrowhead=2, arrowsize=1.0,
                    arrowwidth=1.5, arrowcolor='#ff6b00',
                    text=(f'<b>Portfolio (2)</b><br>'
                          f'Behavioural optimum with {der_label}<br>'
                          f'H={H_sel:.0%}, same constraint as (1)<br>'
                          f'Return = {der_y[_i2]:.1f}%  |  Std dev = {der_x[_i2]:.1f}%'),
                    font=dict(color='#ff6b00', size=9),
                    bgcolor='rgba(13,17,23,0.9)',
                    bordercolor='#ff6b00', borderwidth=1,
                    align='right', xanchor='right'
                )
            except Exception:
                pass

        # Gain arrow at selected H
        try:
            i0 = nd_lbls.index(f'H={H_sel:.0%}')
            i1 = der_lbls.index(f'H={H_sel:.0%}')
            x0, y0 = nd_x[i0], nd_y[i0]
            x1, y1 = der_x[i1], der_y[i1]
            gain = y1 - y0
            # Dashed white line with triangle marker as arrowhead at end
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines+markers',
                line=dict(color='#ffffff', width=2, dash='dash'),
                marker=dict(
                    symbol=['circle', 'arrow'],
                    size=[0, 12],
                    color='#ffffff',
                    angleref='previous'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Text connected by arrow to the gold end point (behavioural with derivative)
            fig.add_annotation(
                x=x1, y=y1,
                ax=x1 + max((x1-x0)*0.3, 8),
                ay=y1 + (y1-y0)*0.2,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.0,
                arrowwidth=1.5, arrowcolor='#ffffff',
                text=f'<b>+{gain:.1f} pp return (with derivative)</b><br>same H & α constraint<br>(same risk aversion λ)',
                font=dict(color='#ffffff', size=10),
                bgcolor='rgba(13,17,23,0.85)',
                bordercolor='#ffffff', borderwidth=1,
                align='left', xanchor='left'
            )
        except (ValueError, IndexError):
            pass

    # ── Portfolio (1) / Equivalence point ───────────────────────────────────────
    # Use actual optimised Portfolio (1) point if available, else fall back to default
    if nd_res_actual:
        _p1_x = nd_res_actual['std_dev'] * 100
        _p1_y = nd_res_actual['expected_return'] * 100
        _lam_str = f"λ={lam_actual:.4f}" if lam_actual else "λ undefined (constraint non-binding)"
        _h_str = (f"H={H_sel:.0%}, α={alpha:.0%}" if alpha is not None
                  else (f"H={H_sel:.0%}, L={L:.0%}" if L is not None else f"H={H_sel:.0%}"))
        fig.add_trace(go.Scatter(
            x=[_p1_x], y=[_p1_y], mode='markers',
            name=f'Portfolio (1) — Behavioural optimum without derivatives ({_h_str})',
            legendrank=5,
            marker=dict(size=13, color='#10b981', symbol='diamond',
                        line=dict(width=0)),
            showlegend=True,
            hovertemplate=f'<b>Portfolio (1)</b><br>Behavioural optimum without derivatives<br>{_h_str} ↔ {_lam_str}<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))
        fig.add_annotation(
            x=_p1_x, y=_p1_y,
            ax=80, ay=70,
            xref='x', yref='y', axref='pixel', ayref='pixel',
            showarrow=True, arrowhead=2, arrowcolor='#10b981',
            arrowwidth=1.5,
            text=f'<b>Portfolio (1)</b><br>Behavioural optimum — no derivative<br>{_h_str} ↔ {_lam_str}<br>Return = {_p1_y:.1f}%  |  Std dev = {_p1_x:.1f}%',
            font=dict(color='#10b981', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#10b981', borderwidth=1,
            align='left', xanchor='left'
        )
    # ── Markowitz MV optimum (always shown when available) ───────────────────
    # Distinct from the behavioural Portfolio (1): this is the unconstrained
    # mean-variance optimum at the reference risk-aversion (lambda = 3.795),
    # shown even when no behavioural optimum is feasible so the Markowitz
    # reference is always visible. When a behavioural optimum exists and
    # coincides with it (the equivalence case), the green diamond overlays it.
    if mv_eq:
        fig.add_trace(go.Scatter(
            x=[mv_eq[0]], y=[mv_eq[1]], mode='markers',
            name='Portfolio (0) — Markowitz MV optimum',
            legendrank=4,
            marker=dict(size=14, color='#a855f7', symbol='circle',
                        line=dict(width=2, color='#ffffff')),
            showlegend=True,
            hovertemplate=f'<b>Portfolio (0) — Markowitz MV optimum</b><br>Minimum-variance portfolio at the expected return of Portfolio (1)<br>{mv_eq_lam_str}<br>Coincides with Portfolio (1) when it is MV-efficient — the MVT/MAT equivalence<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))
        fig.add_annotation(
            x=mv_eq[0], y=mv_eq[1],
            ax=80, ay=-70,
            xref='x', yref='y', axref='pixel', ayref='pixel',
            showarrow=True, arrowhead=2, arrowcolor='#a855f7',
            arrowwidth=1.5,
            text=f'<b>Portfolio (0)</b><br>Markowitz MV optimum<br>{mv_eq_lam_str}<br>Return = {mv_eq[1]:.1f}%  |  Std dev = {mv_eq[0]:.1f}%',
            font=dict(color='#a855f7', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#a855f7', borderwidth=1,
            align='left', xanchor='left'
        )

    # ── Naive benchmarks (securities only): equal-weight / min-variance / max-Sharpe ──
    if benchmarks:
        fig.add_trace(go.Scatter(
            x=[b[1] for b in benchmarks], y=[b[2] for b in benchmarks],
            mode='markers+text', name='Benchmarks (securities only)', legendrank=7,
            marker=dict(size=10, color='#9aa7bd', symbol='diamond',
                        line=dict(color='#0d1117', width=1)),
            text=[b[0] for b in benchmarks], textposition='bottom center',
            textfont=dict(color='#9aa7bd', size=9),
            hovertemplate='<b>%{text}</b><br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
        ))

    if not nd_res_actual:
        fig.add_annotation(
            xref='paper', yref='paper', x=0.5, y=0.5,
            text='No feasible Portfolio (1) at the selected H, alpha.<br>Widen H or relax alpha (see the panel below).',
            showarrow=False,
            font=dict(color='#f0a500', size=11),
            bgcolor='rgba(13,17,23,0.9)', bordercolor='#f0a500', borderwidth=1,
            xanchor='center', yanchor='middle'
        )

    # ── MVT/MAT note ──────────────────────────────────────────────────────────
    fig.add_annotation(
        xref='paper', yref='paper', x=0.5, y=1.0,
        text='MV and behavioral frontiers converge without derivatives (MVT = Mean-Variance Theory / MAT = Mental Accounts Theory)',
        showarrow=False,
        font=dict(color='#ffffff', size=10, style='italic'),
        bgcolor='rgba(13,17,23,0.85)',
        bordercolor='#3a3a5a', borderwidth=1,
        xanchor='center', yanchor='bottom'
    )

    # ── Portfolio (3) point ──────────────────────────────────────────────────
    if p3_x is not None and p3_y is not None:
        fig.add_trace(go.Scatter(
            x=[p3_x], y=[p3_y], mode='markers',
            name=f'Portfolio (3) — same variance as Portfolio (1), with {der_label}',
            legendrank=7,
            marker=dict(size=14, color='#e76f51', symbol='star',
                       line=dict(color='white', width=1)),
            hovertemplate=(f'<b>Portfolio (3)</b><br>Same std dev as Portfolio (1)<br>'
                          f'Std Dev: {p3_x:.2f}%<br>Expected Return: {p3_y:.2f}%<extra></extra>')
        ))
        fig.add_annotation(
            x=p3_x, y=p3_y,
            ax=90, ay=-70,
            xref='x', yref='y', axref='pixel', ayref='pixel',
            showarrow=True, arrowhead=2, arrowsize=1.0,
            arrowwidth=1.5, arrowcolor='#e76f51',
            text=(f'<b>Portfolio (3)</b><br>'
                  f'Same variance as (1)<br>'
                  f'Std dev = {p3_x:.1f}%<br>'
                  f'Return = {p3_y:.1f}% (interpolated)'),
            font=dict(color='#e76f51', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#e76f51', borderwidth=1,
            align='left', xanchor='left'
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1b2330',
        plot_bgcolor='#0e1521',
        title=dict(
            text='Mean-Variance vs Behavioural Portfolio Efficient Frontier',
            font=dict(color='#E3C77E', size=15),
            x=0.5,
            xanchor='center',
            xref='paper'
        ),
        xaxis=dict(
            title=dict(text='Portfolio Risk — Standard Deviation (%)',
                       font=dict(color='#c0c8d8', size=13)),
            gridcolor='#27344e', gridwidth=1, showgrid=True, griddash='dot',
            color='#c0c8d8', zerolinecolor='#2a2a3a',
            showline=True, linecolor='#46566f', linewidth=1, mirror=True,
            range=[max(0, min(mv_x) - 1),
                   max(max(mv_x), max(der_x) if der_x else 0) * 1.06]
        ),
        yaxis=dict(
            title=dict(text='Expected Return (%)',
                       font=dict(color='#c0c8d8', size=13)),
            gridcolor='#27344e', gridwidth=1, showgrid=True, griddash='dot',
            color='#c0c8d8', zerolinecolor='#2a2a3a',
            showline=True, linecolor='#46566f', linewidth=1, mirror=True,
            range=[min(min(mv_y), min(nd_y)) - 2,
                   max(max(mv_y), max(der_y) if der_y else 0) * 1.08]
        ),
        legend=dict(
            bgcolor='rgba(26,26,46,0.9)',
            bordercolor='#3a3a5a', borderwidth=1,
            font=dict(color='white', size=10),
            x=0.01, y=0.99
        ),
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='#1a6bbf',
            font=dict(color='white', size=11)
        ),
        margin=dict(t=80, b=60, l=60, r=20),
        height=560
    )

    # Update margin only
    fig.update_layout(margin=dict(t=80, b=140, l=60, r=20))
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=-0.22,
        text='Behavioural frontiers shown at discrete H levels (-5% to -20%) — each point optimal for that constraint | MV frontier continuous via λ sweep | Both converge via MVT/MAT equivalence when no derivatives present',
        showarrow=False,
        font=dict(color='#8896a8', size=9, style='italic'),
        xanchor='center'
    )
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=-0.27,
        text='Jeddou (2026) — Beyond Mean-Variance Portfolio Optimiser  |  Built on Das & Statman (2009), Das, Markowitz, Scheid & Statman (2010) JFQA & Jeddou (2012)',
        showarrow=False,
        font=dict(color='#ffffff', size=9, style='italic'),
        xanchor='center'
    )

    return fig


def _render_frontier_synced(fig, height=620):
    """Render the 2D frontier so each 'Portfolio (N)' annotation box hides/shows together
    with its marker when that series is toggled in the legend. Plotly annotations are not
    natively bound to traces, so a tiny JS listener mirrors each annotation's visibility to
    the matching trace's visibility on legend clicks. Falls back to a normal Streamlit chart
    if anything goes wrong, so the chart always renders."""
    try:
        import streamlit.components.v1 as _components
        import uuid as _uuid
        _div = "frontsync_" + _uuid.uuid4().hex[:8]
        _inner = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id=_div,
                             config={"responsive": True, "displayModeBar": True,
                                     "edits": {"legendPosition": True,
                                               "annotationPosition": True,
                                               "annotationTail": True}})
        _js = """
<script>
(function(){
  var DIV = "__DIV__";
  function attach(){
    var gd = document.getElementById(DIV);
    if(!gd || !gd.on || !window.Plotly){ return setTimeout(attach, 120); }
    function sync(){
      var anns = (gd.layout && gd.layout.annotations) || [];
      var upd = {};
      anns.forEach(function(a, i){
        var m = (a.text || "").match(/Portfolio \\((\\d)\\)/);
        if(!m) return;
        var pf = "Portfolio (" + m[1] + ")";
        var found = false, vis = true;
        (gd.data || []).forEach(function(tr){
          if((tr.name || "").indexOf(pf) === 0){
            found = true;
            vis = !(tr.visible === "legendonly" || tr.visible === false);
          }
        });
        if(found) upd["annotations[" + i + "].visible"] = vis;
      });
      if(Object.keys(upd).length) window.Plotly.relayout(gd, upd);
    }
    gd.on("plotly_restyle", function(){ setTimeout(sync, 0); });
    gd.on("plotly_legendclick", function(){ setTimeout(sync, 0); return true; });
    gd.on("plotly_legenddoubleclick", function(){ setTimeout(sync, 0); return true; });
  }
  attach();
})();
</script>
""".replace("__DIV__", _div)
        _components.html(_inner + _js, height=height, scrolling=False)
        return True
    except Exception:
        st.plotly_chart(fig, use_container_width=True,
                        config={'responsive': True, 'edits': {'annotationPosition': True, 'annotationTail': True, 'legendPosition': True}, 'displayModeBar': True})
        return False


def plot_payoff(components, vol, S0, r, T, asset_name):
    returns = np.linspace(-0.8, 1.5, 300)
    payoffs, price0 = compute_structured_payoff(returns, components, vol, S0, r, T)
    fig,ax=plt.subplots(figsize=(8,3.5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.axhline(0,color="#3a3a5a",linewidth=0.8,linestyle="--")
    ax.axvline(0,color="#3a3a5a",linewidth=0.8,linestyle="--")
    pos=payoffs>=0; neg=payoffs<0
    ax.fill_between(returns*100,payoffs*100,where=pos,color="#10b981",alpha=0.25)
    ax.fill_between(returns*100,payoffs*100,where=neg,color="#ef4444",alpha=0.25)
    ax.plot(returns*100,payoffs*100,color="#f59e0b",linewidth=2)
    ax.set_xlabel(f"Return of {asset_name} (%)",color="#c0c8d8",fontsize=10)
    ax.set_ylabel("Structured product return (%)",color="#c0c8d8",fontsize=10)
    ax.set_title(f"Payoff diagram  |  Fair value = {price0:.4f}",
                 color="white",fontsize=11,fontweight="bold")
    ax.tick_params(colors="#8896a8",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(True,color="#1e2130",linewidth=0.5,linestyle="--",alpha=0.7)
    plt.tight_layout()
    return fig

def plot_named_payoff(der_config, asset_name, x_lo=-0.8, x_hi=1.5, N=241):
    """Payoff/return diagram for any instrument, computed straight from the
    engine so it matches exactly what the optimiser prices."""
    span = x_hi - x_lo
    mean = (x_lo + x_hi) / 2.0
    sig  = span / 10.0  # ±5σ spans [x_lo, x_hi]
    cfg  = {**der_config, "underlying_index": 0}
    U, _ = build_state_space([mean], [sig], m=N, derivative_config=cfg)
    x = U[:, 0]; y = U[:, 1]
    order = np.argsort(x); x, y = x[order], y[order]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.axhline(0, color="#3a3a5a", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#3a3a5a", linewidth=0.8, linestyle="--")
    ax.fill_between(x*100, y*100, where=(y >= 0), color="#10b981", alpha=0.25)
    ax.fill_between(x*100, y*100, where=(y < 0),  color="#ef4444", alpha=0.25)
    ax.plot(x*100, y*100, color="#f59e0b", linewidth=2)
    ax.set_xlabel(f"Return of {asset_name} (%)", color="#c0c8d8", fontsize=10)
    ax.set_ylabel("Derivative return (%)", color="#c0c8d8", fontsize=10)
    ax.set_title("Payoff diagram", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8896a8", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(True, color="#1e2130", linewidth=0.5, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig

def plot_frontier(mv_x,mv_y,mv_eq,nd_x,nd_y,nd_lbls,
                  der_x,der_y,der_lbls,der_label,H_sel,alpha):
    fig,ax=plt.subplots(figsize=(11,6.5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.grid(True,color="#1e2130",linewidth=0.6,linestyle="--",alpha=0.8)
    ax.set_axisbelow(True)
    ax.plot(mv_x,mv_y,color="#a855f7",linewidth=2,linestyle="--",
            label="Mean-variance frontier (Markowitz)",zorder=2,alpha=0.9)
    ax.plot(nd_x,nd_y,color="#1a6bbf",linewidth=2.5,marker="o",markersize=7,
            markerfacecolor="#1a6bbf",label="Behavioral — no derivative",zorder=3)
    for x,y,l in zip(nd_x,nd_y,nd_lbls):
        ax.annotate(l,xy=(x,y),xytext=(x,y-1.8),
                    color="#7fb3e8",fontsize=7.5,ha="center",zorder=4)
    if der_x:
        ax.scatter(der_x,der_y,color="#f59e0b",s=65,marker="s",zorder=3,
                   label=f"Behavioral — {der_label}")
        for x,y,l in zip(der_x,der_y,der_lbls):
            if l==f"H={H_sel:.0%}":
                ax.annotate(f"{l}, α={alpha:.0%}",xy=(x,y),
                            xytext=(x-8,y+2),color="#f59e0b",fontsize=8,
                            arrowprops=dict(arrowstyle="->",color="#f59e0b",lw=1.2),
                            bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                                      edgecolor="#f59e0b",alpha=0.85))
        try:
            i0=nd_lbls.index(f"H={H_sel:.0%}")
            i1=der_lbls.index(f"H={H_sel:.0%}")
            x0,y0=nd_x[i0],nd_y[i0]; x1,y1=der_x[i1],der_y[i1]
            ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                        arrowprops=dict(arrowstyle="->",color="#ffffff",
                                        lw=1.6,linestyle="dashed"))
            ax.text(0.55, 0.45,
                    f"+{y1-y0:.1f} pp return\nsame H & α constraint\n(same risk aversion λ)",
                    color="#ffffff", fontsize=8, ha='center', va='center',
                    transform=ax.transAxes)
        except (ValueError,IndexError): pass
    if mv_eq:
        ax.scatter(*mv_eq,color="#10b981",s=130,zorder=5,marker="D")
        ax.annotate(f"Equivalence point\nλ=3.795 ↔ H=-10%, α=5%\n={mv_eq[1]:.1f}%",
                    xy=mv_eq,xytext=(mv_eq[0]+3,mv_eq[1]-5),color="#10b981",fontsize=8,
                    arrowprops=dict(arrowstyle="->",color="#10b981",lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                              edgecolor="#10b981",alpha=0.9),zorder=6)
    ax.text(0.5,0.97,
            "MV and behavioral frontiers converge without derivatives\n"
            "(MVT/MAT equivalence — Das, Markowitz, Scheid & Statman 2010)",
            transform=ax.transAxes,color="#ffffff",fontsize=7.5,
            ha="center",va="top",style="italic",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                      edgecolor="#3a3a5a",alpha=0.95),zorder=10)
    ax.set_xlabel("Portfolio Risk — Standard Deviation (%)",color="#c0c8d8",fontsize=10,labelpad=6)
    ax.set_ylabel("Expected Return (%)",color="#c0c8d8",fontsize=10,labelpad=6)
    ax.set_title("Mean-Variance vs Behavioral Portfolio Frontier",
                 color="white",fontsize=13,fontweight="bold",pad=12)
    ax.tick_params(colors="#8896a8",labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.legend(loc="upper left",fontsize=9,facecolor="#1a1a2e",
              edgecolor="#3a3a5a",labelcolor="white",framealpha=0.9)
    fig.text(0.5,0.001,
             "Jeddou (2026) — Beyond Mean-Variance Portfolio Optimiser  |  "
             "Built on Das & Statman (2009), Das, Markowitz, Scheid & Statman (2010) JFQA & Jeddou (2012)",
             ha="center",color="#ffffff",fontsize=7,style="italic")
    all_x=mv_x+nd_x+(der_x if der_x else [])
    all_y=mv_y+nd_y+(der_y if der_y else [])
    ax.set_xlim(0,max(all_x)*1.15); ax.set_ylim(min(all_y)-3,max(all_y)+6)
    plt.tight_layout(rect=[0,0.02,1,1])
    return fig

# ═══════════════════════════════════════════════════════════════════════════════════
# TILE-LAUNCHER HOME  (navigation only — no section content is changed)
# ═══════════════════════════════════════════════════════════════════════════════════
import os as _os, base64 as _b64
_HOME_DIR = _os.path.dirname(_os.path.abspath(__file__))
_VIEWS = ("home", "optimiser", "scalable", "backtest", "riskprofile", "ticker", "about", "glossary", "portfolio")

@st.cache_data(show_spinner=False)
def _home_assets(mtimes=None):
    out = {}
    for k, fn in (("OPT", "home_optimiser_grid3d.png"), ("MC", "home_optimiser_mcvar6.png"), ("BT", "home_backtest2.png"), ("RP", "home_riskprofile.png"), ("TKR", "home_ticker3.png"), ("LP", "home_liveportfolio.png")):
        fp = _os.path.join(_HOME_DIR, fn)
        out[k] = ("data:image/png;base64," + _b64.b64encode(open(fp, "rb").read()).decode()) if _os.path.exists(fp) else ""
    return out

def _home_assets_key():
    ms = []
    for fn in ("home_optimiser_grid3d.png", "home_optimiser_mcvar6.png", "home_backtest2.png", "home_riskprofile.png", "home_ticker3.png", "home_liveportfolio.png"):
        fp = _os.path.join(_HOME_DIR, fn)
        ms.append(_os.path.getmtime(fp) if _os.path.exists(fp) else 0)
    return tuple(ms)

def _mc_joint_scatter(R_sec, names, weights, alpha, ia=None, ib=None):
    """Plotly scatter of two assets' Monte-Carlo joint-return scenarios, with the joint
    lower-tail (both <= their alpha-quantile) highlighted — the actual scenarios fed to
    the scalable CVaR program."""
    import numpy as _np, plotly.graph_objects as _go
    R = _np.asarray(R_sec, dtype=float)
    N = len(names)
    wv = _np.abs(_np.asarray(weights, dtype=float)[:N]) if weights is not None else _np.ones(N)
    order = list(_np.argsort(-wv))
    if ia is None: ia = order[0]
    if ib is None: ib = order[1] if N > 1 else order[0]
    if ia == ib: ib = (ia + 1) % N
    xa = R[:, ia] * 100.0; yb = R[:, ib] * 100.0
    qa = _np.quantile(xa, alpha); qb = _np.quantile(yb, alpha)
    crash = (xa <= qa) & (yb <= qb)
    S = len(xa)
    if S > 4000:
        keep = _np.zeros(S, bool)
        keep[_np.random.RandomState(0).choice(S, 4000, replace=False)] = True
    else:
        keep = _np.ones(S, bool)
    fig = _go.Figure()
    fig.add_trace(_go.Scattergl(
        x=xa[keep & ~crash], y=yb[keep & ~crash], mode="markers",
        marker=dict(size=3, color="#4a9eff", opacity=0.28),
        name="joint scenarios", hovertemplate=f"{names[ia]}: %{{x:.1f}}%<br>{names[ib]}: %{{y:.1f}}%<extra></extra>"))
    fig.add_trace(_go.Scattergl(
        x=xa[keep & crash], y=yb[keep & crash], mode="markers",
        marker=dict(size=4, color="#d15866", opacity=0.75),
        name="joint tail (both \u2264 %d%% quantile)" % int(round(alpha * 100)),
        hovertemplate=f"{names[ia]}: %{{x:.1f}}%<br>{names[ib]}: %{{y:.1f}}%<extra>tail</extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=460, margin=dict(l=10, r=10, t=46, b=10),
        title=dict(text="Joint return scenarios \u00b7 %s vs %s" % (names[ia], names[ib]),
                   x=0.5, font=dict(color="#E3C77E", size=15)),
        xaxis=dict(title="%s return (%%)" % names[ia], gridcolor="#1e2130",
                   zeroline=True, zerolinecolor="#3a3a5a"),
        yaxis=dict(title="%s return (%%)" % names[ib], gridcolor="#1e2130",
                   zeroline=True, zerolinecolor="#3a3a5a"),
        legend=dict(bgcolor="rgba(13,17,23,0.6)", bordercolor="#3a3a5a", borderwidth=1,
                    font=dict(color="#ffffff", size=12), x=0.01, y=0.99, xanchor="left", yanchor="top"))
    return fig


@st.fragment
def _mc_joint_scatter_view(R_sec, names, weights, alpha):
    """Asset-pair selector + joint-return scatter, wrapped in a Streamlit fragment so
    changing the selected assets re-renders ONLY this chart — the rest of the
    optimisation results stay on screen (no full rerun, no need to click Run again)."""
    import numpy as _np
    names = list(names)
    st.markdown('#### <span style="color:#E3C77E">Joint return scenarios</span>', unsafe_allow_html=True)
    st.caption("The actual Monte-Carlo scenarios fed to the CVaR program. The red "
               "lower-left cluster is the joint-crash tail dependence the copula "
               "captures (more pronounced with a Student-t copula).")
    if len(names) < 2:
        st.info("Add at least two securities to see their joint scenarios.")
        return
    _wabs = _np.abs(_np.asarray(weights, dtype=float)[:len(names)])
    _ord = list(_np.argsort(-_wabs))
    _ca, _cb = st.columns(2)
    _ia = _ca.selectbox("Asset A", names, index=int(_ord[0]), key="mc_scat_a")
    _ib = _cb.selectbox("Asset B", names, index=int(_ord[1]), key="mc_scat_b")
    st.plotly_chart(
        _mc_joint_scatter(R_sec, names, weights, alpha,
                          ia=names.index(_ia), ib=names.index(_ib)),
        use_container_width=True, config={'edits': {'legendPosition': True}, 'displayModeBar': True})


_G3D_METHOD_NOTE_HTML = (
    "<div style='line-height:1.55;color:#aebccd;font-size:.85rem'>"
    "<b>What it shows.</b> The chosen objective over the two selected instruments' weights; the "
    "other instruments are held at the optimum's proportions, and the remaining weight fills the "
    "long-only simplex.<br>"
    "<b>It is a Monte-Carlo illustration, not the exact grid solve.</b> Securities are drawn from a "
    "Gaussian copula on the estimated means/covariances, and the derivative is priced by "
    "Black-Scholes on its underlying (the <i>scalable</i> engine's model). The exact grid engine "
    "evaluates a discrete state space, so the surface can differ slightly from it.<br>"
    "<b>The marked optimum uses the exact grid weights</b> (Portfolio 1 / 2) from the results table, "
    "so the marker's coordinates match the summary above.<br>"
    "<b>Grey = infeasible portfolios</b> that breach the downside limit (P(r&lt;H) &le; &alpha;, or the ES "
    "floor); the coloured surface is the feasible region. <b>The optimum sits at the top edge of the "
    "coloured region</b> — it's the best portfolio the engine is allowed to pick. The greyed peak "
    "beyond it scores higher but breaks the limit, which is exactly what optimising under a downside "
    "constraint means.<br>"
    "The one case that lands <i>exactly</i> on a peak is <i>Mean-variance</i> with <b>Portfolio (1)</b> "
    "(no derivative) — a genuine unconstrained hill-top. <i>Expected return</i> is linear (optimum on "
    "the boundary edge). <i>Downside-utility</i> is offered <b>only when the derivative is on an axis</b> "
    "(the lens for the protective instrument; &kappa; &asymp; __KAP__), where the optimum is closest to the "
    "peak — but still just inside the boundary.<br>"
    "<b>Coordinates are always exact</b> (they match the summary). &kappa; is a heuristic scale match, not the "
    "engine's exact shadow price."
    "</div>"
)


@st.fragment
def _grid_obj_surface_view():
    """Unified interactive 3D landscape for the grid optimiser. Two selectboxes pick any
    two instruments (securities OR the derivative) for the axes; a third picks the objective
    (expected return / mean-variance / downside-utility). The surface is computed from a joint
    Monte-Carlo scenario set; the marked optimum uses the EXACT grid weights (Portfolio 1/2)."""
    import numpy as _np
    d = st.session_state.get('_grid3d_data')
    if not d or d.get('R') is None:
        return
    # Gild the enclosing expander's title (the "Objective landscape in 3D…" summary).
    st.markdown(
        "<style>details:has(.bmv-g3d-mk) summary,details:has(.bmv-g3d-mk) summary *"
        "{color:#E3C77E !important}</style><span class='bmv-g3d-mk'></span>",
        unsafe_allow_html=True)
    labels = list(d['labels']); R = _np.asarray(d['R'], dtype=float)
    M = R.shape[1]; lam = d.get('lam') or 3.795
    w1 = d.get('w1'); w2 = d.get('w2')
    if w2 is not None and len(w2) == M:
        w_opt = _np.asarray(w2, float); opt_name = 'Portfolio (2)'
        opt_legend = 'Portfolio (2) — Behavioural optimum (with derivative)'
    elif w1 is not None:
        w_opt = _np.zeros(M); w_opt[:len(w1)] = _np.asarray(w1, float); opt_name = 'Portfolio (1)'
        opt_legend = 'Portfolio (1) — Behavioural optimum'
    else:
        return
    if M < 3:
        st.caption("The 3D landscape needs at least three instruments (two free axes + a remainder).")
        return

    st.caption("Pick any two instruments for the axes and an objective. The surface is a "
               "**Monte-Carlo illustration** (see the method note below). **Grey = portfolios that breach "
               "the downside limit** (infeasible); the coloured part is the feasible region, and the marked "
               "optimum (exact grid weights) sits at its top edge — the best of what's allowed.")
    _ord = list(_np.argsort(-_np.abs(w_opt)))
    n_sec = int(d.get('n_sec', len(labels)))
    _c1, _c2, _c3 = st.columns(3)
    _ia = _c1.selectbox("Instrument — X axis", labels, index=int(_ord[0]), key="g3d_x")
    _ib = _c2.selectbox("Instrument — Y axis", labels, index=int(_ord[1]), key="g3d_y")
    i, j = labels.index(_ia), labels.index(_ib)
    # Downside-utility is offered ONLY when the derivative is on an axis — it's the lens for the
    # instrument the optimum holds for downside protection, and only there does the optimum sit
    # near the peak. On a securities-only slice it would mislead, so it's hidden.
    _has_der_axis = (i >= n_sec) or (j >= n_sec)
    _obj_opts = (["Mean-variance  (E[r] − ½λ·var)", "Downside-utility  (E[r] − κ·CVaR)", "Expected return"]
                 if _has_der_axis else ["Mean-variance  (E[r] − ½λ·var)", "Expected return"])
    if st.session_state.get("g3d_obj") not in _obj_opts:
        st.session_state["g3d_obj"] = _obj_opts[0]
    _obj = _c3.selectbox("Objective", _obj_opts, key="g3d_obj")
    if i == j:
        st.info("Pick two different instruments for the two axes.")
        return
    others = [k for k in range(M) if k not in (i, j)]
    w_oth = w_opt[others]; s_oth = float(w_oth.sum())
    base = (w_oth / s_oth) if abs(s_oth) > 1e-9 else _np.ones(len(others)) / max(len(others), 1)

    _is_ret = _obj.startswith("Expected"); _is_dn = _obj.startswith("Downside")
    _ac = 0.05
    _Hc = d.get('H'); _alpha_c = d.get('alpha'); _use_es = bool(d.get('use_es')); _Lc = d.get('L')
    def _eval(w):
        port = R @ w
        mean = float(port.mean())
        var = 0.0 if _is_ret else float(port.var())
        cvar = 0.0
        if _is_dn:
            kk = max(1, int(_ac * len(port)))
            cvar = float(_np.partition(port, kk)[:kk].mean())
        # feasibility under the run's downside constraint
        feas = True
        if _Hc is not None:
            below = port < _Hc
            if _use_es and _Lc is not None:
                feas = True if int(below.sum()) == 0 else (float(port[below].mean()) >= _Lc)
            else:
                feas = (float(below.mean()) <= (_alpha_c if _alpha_c is not None else 0.05))
        return mean, var, cvar, feas
    _m0, _v0, _cv0, _ = _eval(w_opt)
    kappa = (0.5 * lam * _v0 / abs(_cv0)) if (_is_dn and abs(_cv0) > 1e-9) else 1.0
    st.session_state['_g3d_kappa'] = float(kappa)   # method note renders full-width below the columns
    def _objf(mean, var, cvar):
        if _is_ret:
            return mean * 100
        if _is_dn:
            return (mean + kappa * cvar) * 100      # cvar < 0; + rewards a less-negative tail
        return (mean - 0.5 * lam * var) * 100

    Ms = 42
    g = _np.linspace(0, 1, Ms); X, Y = _np.meshgrid(g, g); Rem = 1 - X - Y
    Zf = _np.full_like(X, _np.nan)   # feasible (coloured by objective)
    Zi = _np.full_like(X, _np.nan)   # infeasible — breaches the downside limit (greyed)
    for a in range(Ms):
        for b in range(Ms):
            rr = Rem[a, b]
            if rr >= -1e-9:
                w = _np.zeros(M); w[i] = X[a, b]; w[j] = Y[a, b]
                for t, k in enumerate(others):
                    w[k] = max(rr, 0.0) * base[t]
                _mn, _vr, _cv, _fe = _eval(w)
                _zv = _objf(_mn, _vr, _cv)
                if _fe:
                    Zf[a, b] = _zv
                else:
                    Zi[a, b] = _zv
    ox, oy = float(w_opt[i] * 100), float(w_opt[j] * 100)
    oz = _objf(_m0, _v0, _cv0)
    _mc = '#f59e0b' if opt_name == 'Portfolio (2)' else '#10b981'
    _tc = '#fcd34d' if opt_name == 'Portfolio (2)' else '#6ee7b7'
    try:
        import plotly.graph_objects as _go
        fig = _go.Figure()
        # infeasible portfolios (breach the downside limit) — greyed out
        fig.add_trace(_go.Surface(
            x=X * 100, y=Y * 100, z=Zi, showscale=False, opacity=0.5,
            colorscale=[[0, '#39404e'], [1, '#39404e']],
            hovertemplate=f'{_ia}: %{{x:.0f}}%<br>{_ib}: %{{y:.0f}}%'
                          '<br><b>infeasible</b> — breaches the downside limit<extra></extra>'))
        # feasible region — coloured by objective; the optimum sits at its top edge
        fig.add_trace(_go.Surface(
            x=X * 100, y=Y * 100, z=Zf, colorscale='RdYlGn', showscale=True,
            colorbar=dict(title=dict(text='Objective'), len=0.7, thickness=14),
            hovertemplate=f'{_ia}: %{{x:.0f}}%<br>{_ib}: %{{y:.0f}}%<br>Objective: %{{z:.2f}}<extra></extra>'))
        fig.add_trace(_go.Scatter3d(
            x=[ox], y=[oy], z=[oz], mode='markers+text', name=opt_legend,
            marker=dict(size=10, color=_mc, symbol='diamond', line=dict(color='#0d1117', width=2)),
            text=[opt_name], textposition='top center', textfont=dict(color=_tc, size=15),
            hovertemplate=f'<b>{opt_name}</b> — grid optimum (exact weights)<extra></extra>'))
        # legend-only swatches (Surface traces don't appear in the legend), so the
        # grey zone is labelled as the non-feasible region right on the chart.
        fig.add_trace(_go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=11, color='#39404e', symbol='square'),
            name='Infeasible region — breaches the downside limit', showlegend=True,
            hoverinfo='skip'))
        fig.add_trace(_go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=11, color='#5fb56a', symbol='square'),
            name='Feasible region — coloured by objective', showlegend=True,
            hoverinfo='skip'))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', height=560,
            margin=dict(l=0, r=0, t=42, b=0), showlegend=True,
            legend=dict(x=0.0, y=0.0, xanchor='left', yanchor='bottom',
                        bgcolor='rgba(13,17,23,0.62)', bordercolor='#27344e', borderwidth=1,
                        font=dict(color='#c9d1d9', size=11)),
            title=dict(text=f'{opt_name} landscape — {_obj.split("  ")[0]}  ·  Monte-Carlo illustration',
                       x=0.5, xanchor='center', y=0.97, font=dict(color='#E3C77E', size=13)),
            scene=dict(bgcolor='#0d1117',
                       xaxis=dict(title=f'{_ia} weight (%)', backgroundcolor='#0d1117', gridcolor='#27344e'),
                       yaxis=dict(title=f'{_ib} weight (%)', backgroundcolor='#0d1117', gridcolor='#27344e'),
                       zaxis=dict(title='Objective', backgroundcolor='#0d1117', gridcolor='#27344e'),
                       camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        # The "Method & approximations" note is rendered full-width *below* the columns
        # (see the Results section) so expanding it never disturbs the side-by-side alignment.
    except Exception as _e3:
        st.caption(f"(3D landscape unavailable: {_e3})")


def _mc_pnl_distribution(port, alpha, es, floor, er, overlays=None):
    """Histogram of the optimal portfolio's scenario returns with the worst-alpha tail
    shaded; VaR (alpha-quantile), realised alpha-CVaR (= es) and the floor L marked.
    Uses the SAME scenario returns the CVaR program optimised, so the shaded tail's
    average equals the reported realised alpha-CVaR by construction.

    `overlays` (optional): list of (label, returns_array, color) drawn as smoothed
    density-outline curves on the *same* bins \u2014 e.g. same-return benchmark portfolios,
    to compare the shape of the left tail."""
    import numpy as _np, plotly.graph_objects as _go
    r = _np.asarray(port, dtype=float) * 100.0
    if r.size == 0:
        return _go.Figure()
    var = float(_np.quantile(r, alpha)); cvar = float(es) * 100.0
    L = float(floor) * 100.0; mean = float(er) * 100.0
    # span the bins over the optimum AND any overlays so the curves are comparable
    _lo, _hi = float(r.min()), float(r.max())
    _ovs = []
    if overlays:
        for _lbl, _ov, _c in overlays:
            _o = _np.asarray(_ov, dtype=float) * 100.0
            if _o.size:
                _ovs.append((_lbl, _o, _c))
                _lo = min(_lo, float(_o.min())); _hi = max(_hi, float(_o.max()))
    nb = 60
    edges = _np.linspace(_lo, _hi, nb + 1)
    counts, _ = _np.histogram(r, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:]); bw = float(edges[1] - edges[0])
    colors = ["#d15866" if c <= var else "#4a9eff" for c in centers]
    fig = _go.Figure()
    fig.add_trace(_go.Bar(x=centers, y=counts, width=bw * 0.96, showlegend=False,
                          marker=dict(color=colors, line=dict(width=0)), hovertemplate="Return: %{x:.1f}%<br>Scenarios: %{y:.0f}<extra></extra>"))
    # legend proxies (the bar uses per-bin colours, so it can't carry one clear swatch)
    fig.add_trace(_go.Scatter(x=[None], y=[None], mode="markers", name="Portfolio (1) \u2014 CVaR optimum",
                              marker=dict(size=11, symbol="square", color="#4a9eff"), hoverinfo="skip"))
    fig.add_trace(_go.Scatter(x=[None], y=[None], mode="markers", name="Portfolio (1) \u2014 worst-\u03b1 tail (shaded)",
                              marker=dict(size=11, symbol="square", color="#d15866"), hoverinfo="skip"))
    _ymax = float(counts.max()) if counts.size else 1.0
    for _lbl, _o, _c in _ovs:
        _oc, _ = _np.histogram(_o, bins=edges)
        _ymax = max(_ymax, float(_oc.max()))
        fig.add_trace(_go.Scatter(x=centers, y=_oc, mode="lines", name=_lbl,
                                  line=dict(color=_c, width=2.2, shape="spline", dash="dash"),
                                  opacity=0.85,
                                  hovertemplate=_lbl + "<br>Return: %{x:.1f}%<br>Scenarios: %{y:.0f}<extra></extra>"))
    ymax = _ymax * 1.12
    def _vline(xv, color, dash, text, ylab):
        fig.add_shape(type="line", x0=xv, x1=xv, y0=0, y1=ymax * 0.99,
                      line=dict(color=color, width=1.7, dash=dash))
        fig.add_annotation(x=xv, y=ylab, text=text, showarrow=False,
                           font=dict(color=color, size=10.5),
                           bgcolor="rgba(13,17,23,0.72)", borderpad=2)
    _vline(mean, "#cbd5e1", "dot",  "E[r] %.1f%%" % mean, ymax * 1.04)
    _vline(var,  "#f5b342", "dash", "VaR %.0f%%" % var,   ymax * 0.92)
    _vline(cvar, "#fb6a78", "dash", "CVaR %.1f%%" % cvar, ymax * 0.80)
    _vline(L,    "#9aa7bd", "dot",  "floor L %.0f%%" % L, ymax * 0.66)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      height=420, bargap=0.02, showlegend=True, margin=dict(l=10, r=10, t=46, b=10),
                      title=dict(text="Portfolio return distribution \u00b7 worst-\u03b1 tail shaded",
                                 x=0.5, font=dict(color="#E3C77E", size=15)),
                      legend=dict(bgcolor="rgba(13,17,23,0.72)", bordercolor="#3a3a5a", borderwidth=1,
                                  font=dict(color="#e7ecf4", size=9), x=0.99, y=0.99,
                                  xanchor="right", yanchor="top"),
                      xaxis=dict(title="Portfolio return (%)", gridcolor="#1e2130",
                                 zeroline=True, zerolinecolor="#3a3a5a"),
                      yaxis=dict(title="Scenarios", gridcolor="#1e2130"))
    return fig


_HOME_CSS = "<style>[data-testid='stAppViewContainer']{background:radial-gradient(1000px 560px at 78% -12%,rgba(74,158,255,.12),transparent 60%),radial-gradient(760px 420px at -5% 112%,rgba(245,185,66,.07),transparent 55%),#0d1117 !important}[data-testid='stHeader']{background:transparent !important}[data-testid='stMain'],section.main{background:transparent !important}section[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none!important}.bmv-home{--blue:#4a9eff;--gold:#f5b942;--gold2:#caa14a;--green:#16a34a;--border:#30363d;--surface:#161b22;--surface2:#1b2330;--text:#fafafa;--muted:#8b949e;--text2:#c9d1d9;font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--text)}.bmv-home *{box-sizing:border-box}.bmv-hero{display:flex;flex-direction:row;gap:14px;align-items:center;justify-content:center;text-align:left;margin:.2rem 0 1.5rem}.bmv-mark{width:46px;height:46px;border-radius:11px;flex:none;display:grid;place-items:center;background:linear-gradient(135deg,var(--gold),var(--gold2));color:#1a1205;font-weight:700;font-family:'IBM Plex Serif',Georgia,'Times New Roman',serif;font-size:1.5rem}.bmv-eyebrow{font-size:.84rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9;margin-bottom:9px}.bmv-eyebrow .w{color:var(--gold);font-style:italic}.bmv-h1{font-family:'IBM Plex Serif',Georgia,'Times New Roman',serif;font-weight:600;font-size:1.95rem;line-height:1.08;margin-bottom:8px}.bmv-h1 .em{color:#E3C77E}.bmv-lede{color:var(--text2);font-size:.92rem;max-width:62ch}.bmv-sub{font-family:'IBM Plex Serif',Georgia,'Times New Roman',serif;font-size:1.2rem;font-weight:500;color:#aeb9c9;margin-top:5px}.bmv-label{font-size:.64rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 12px}.bmv-rule{height:1px;border:0;width:min(560px,78%);margin:6px auto 20px;background:linear-gradient(90deg,transparent,rgba(227,199,126,.55),transparent)}.bmv-tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:44px;margin-bottom:18px}.bmv-tile{position:relative;display:flex;flex-direction:column;text-decoration:none;color:var(--text);overflow:hidden;background:linear-gradient(165deg,var(--surface2),var(--surface));border:1px solid var(--border);border-radius:16px;transition:.22s cubic-bezier(.2,.7,.3,1)}.bmv-tile:hover{transform:translateY(-4px);border-color:var(--accent,var(--blue));box-shadow:0 22px 46px -24px var(--glow,rgba(74,158,255,.5))}.bmv-thumb{height:104px;background:#0a0e15;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:center;overflow:hidden}.bmv-thumb img{width:100%;height:100%;object-fit:contain;display:block;transition:.3s}.bmv-tile:hover .bmv-thumb img{transform:scale(1.03)}.bmv-body{padding:11px 14px;display:flex;flex-direction:column;flex:1}.bmv-thead{display:flex;align-items:center;justify-content:center;gap:9px;margin-bottom:7px}.bmv-ico{width:30px;height:30px;border-radius:8px;display:grid;place-items:center;font-size:1.05rem;flex:none;background:var(--icobg,rgba(74,158,255,.12));border:1px solid var(--icobd,rgba(74,158,255,.3))}.bmv-tt{font-weight:600;font-size:0.97rem;letter-spacing:.05em}.bmv-tile.blue .bmv-tt,.bmv-tile.gold .bmv-tt,.bmv-tile.green .bmv-tt,.bmv-tile.slate .bmv-tt,.bmv-tile.red .bmv-tt,.bmv-tile.purple .bmv-tt,.bmv-tile.teal .bmv-tt{color:#E3C77E}.bmv-home a.bmv-tile,.bmv-home a.bmv-tile:hover,.bmv-home a.bmv-tile:focus{text-decoration:none!important}.bmv-tile.blue .bmv-td,.bmv-tile.gold .bmv-td,.bmv-tile.green .bmv-td,.bmv-tile.red .bmv-td,.bmv-tile.purple .bmv-td,.bmv-tile.teal .bmv-td,.bmv-tiles:not(.ref) .bmv-tile.slate .bmv-td{text-align:center}.bmv-tile.purple .bmv-badge,.bmv-tile.teal .bmv-badge{align-self:center;text-align:center}.bmv-tile.blue .bmv-foot,.bmv-tile.gold .bmv-foot,.bmv-tile.green .bmv-foot,.bmv-tile.red .bmv-foot,.bmv-tile.purple .bmv-foot,.bmv-tile.teal .bmv-foot,.bmv-tiles:not(.ref) .bmv-tile.slate .bmv-foot{justify-content:center}.bmv-td{font-size:.76rem;color:var(--muted);line-height:1.42}.bmv-td b{color:var(--text2);font-weight:600}.bmv-foot{margin-top:auto;padding-top:9px;display:flex;align-items:center;justify-content:space-between}.bmv-tag{font-family:'IBM Plex Mono','SF Mono',Menlo,Consolas,monospace;font-size:.66rem;color:var(--text2);background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:3px 8px}.bmv-arw{font-size:1.05rem;color:var(--accent,var(--blue));opacity:0;transform:translateX(-4px);transition:.22s}.bmv-tile:hover .bmv-arw{opacity:1;transform:translateX(0)}.bmv-badge{display:inline-flex;align-items:center;gap:5px;font-family:'IBM Plex Mono','SF Mono',Menlo,Consolas,monospace;font-size:.63rem;color:var(--blue);background:rgba(74,158,255,.1);border:1px solid rgba(74,158,255,.32);border-radius:6px;padding:4px 8px;margin-top:11px;line-height:1.3;width:fit-content}.bmv-tile.blue{--accent:#4a9eff;--glow:rgba(74,158,255,.5);--icobg:rgba(74,158,255,.12);--icobd:rgba(74,158,255,.32)}.bmv-tile.gold{--accent:#f5b942;--glow:rgba(245,185,66,.45);--icobg:rgba(245,185,66,.12);--icobd:rgba(245,185,66,.32)}.bmv-tile.green{--accent:#16a34a;--glow:rgba(22,163,74,.45);--icobg:rgba(22,163,74,.14);--icobd:rgba(22,163,74,.34)}.bmv-tile.slate{--accent:#7d8aa0;--glow:rgba(125,138,160,.4);--icobg:rgba(125,138,160,.12);--icobd:rgba(125,138,160,.3)}.bmv-tile.red{--accent:#f87171;--glow:rgba(248,113,113,.42);--icobg:rgba(248,113,113,.12);--icobd:rgba(248,113,113,.32)}.bmv-tile.purple{--accent:#a855f7;--glow:rgba(168,85,247,.45);--icobg:rgba(168,85,247,.12);--icobd:rgba(168,85,247,.34)}.bmv-tile.teal{--accent:#2dd4bf;--glow:rgba(45,212,191,.4);--icobg:rgba(45,212,191,.12);--icobd:rgba(45,212,191,.32)}.bmv-soon{color:#5b6675;font-size:.92rem;font-weight:600;letter-spacing:.04em;border:1px dashed #30363d;border-radius:10px;padding:.45rem 1.1rem}.bmv-tiles.ref{grid-template-columns:repeat(3,1fr)}.bmv-tiles.ref .bmv-tile{flex-direction:row;flex-wrap:wrap;align-items:center;justify-content:center;gap:7px 9px;padding:15px 16px}.bmv-tiles.ref .bmv-tile>div{display:contents}.bmv-tiles.ref .bmv-td{flex-basis:100%}.bmv-tiles.ref .bmv-tt,.bmv-tiles.ref .bmv-td{text-align:center}.bmv-tiles.ref .bmv-ico{width:38px;height:38px;font-size:1.15rem}.bmv-tiles.ref .bmv-tt{font-size:.95rem}.bmv-tiles.ref .bmv-td{font-size:.74rem;margin-top:2px}.bmv-aipill{display:inline-block;font-size:.55rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;vertical-align:middle;margin-left:7px;padding:2px 7px;border-radius:999px;color:#1a1205;background:linear-gradient(135deg,var(--gold),var(--gold2))}@media(max-width:640px){.bmv-tiles{grid-template-columns:1fr 1fr}.bmv-tiles.ref{grid-template-columns:1fr}}</style>"
_HOME_HTML = '<div class="bmv-home">\n  <div class="bmv-hero">\n    <div class="bmv-mark">&beta;</div>\n    <div>\n      <div class="bmv-eyebrow">Portfolio Optimisation <span class="w">with</span> Derivatives &amp; Structured Products</div>\n      <div class="bmv-h1">Beyond <span class="em">Mean-Variance</span></div>\n      <div class="bmv-sub">Mental Accounting Framework</div>\n    </div>\n  </div>\n  <div class="bmv-rule"></div><div class="bmv-label">Tools</div>\n  <div class="bmv-tiles">\n    <a class="bmv-tile red" href="?view=riskprofile" target="_self">\n      <div class="bmv-thumb"><img src="__RP__"></div>\n      <div class="bmv-body">\n        <div class="bmv-thead"><span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg></span><span class="bmv-tt">Risk Profile</span></div>\n        <div class="bmv-td">13-question Grable&ndash;Lytton scale that sets your simulation&rsquo;s risk parameters (H, &alpha;, L).</div>\n        <div class="bmv-foot"><span class="bmv-tag">find your risk level</span><span class="bmv-arw">&rarr;</span></div>\n      </div>\n    </a>\n    <a class="bmv-tile green" href="?view=ticker" target="_self">\n      <div class="bmv-thumb"><img src="__TKR__"></div>\n      <div class="bmv-body">\n        <div class="bmv-thead"><span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#26a641" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span><span class="bmv-tt">Ticker Analytics</span></div>\n        <div class="bmv-td">Enter a ticker for its price, return, risk level and key ratios, each with a plain-language explanation.</div>\n        <div class="bmv-foot"><span class="bmv-tag">stocks · ETFs · indices</span><span class="bmv-arw">&rarr;</span></div>\n      </div>\n    </a>\n    <a class="bmv-tile gold" href="?view=optimiser" target="_self">\n      <div class="bmv-thumb"><img src="__OPT__"></div>\n      <div class="bmv-body">\n        <div class="bmv-thead"><span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f5b942" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg></span><span class="bmv-tt">Grid Portfolio Optimiser</span></div>\n        <div class="bmv-td">Exact grid engine on the Das&ndash;Statman states — VaR, thesis-faithful ES and rigorous-ES, with derivatives.</div>\n        <div class="bmv-foot"><span class="bmv-tag">best for small portfolios</span><span class="bmv-arw">&rarr;</span></div>\n      </div>\n    </a>\n    <a class="bmv-tile blue" href="?view=scalable" target="_self">\n      <div class="bmv-thumb"><img src="__MC__"></div>\n      <div class="bmv-body">\n        <div class="bmv-thead"><span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></span><span class="bmv-tt">Scalable Portfolio Optimiser</span></div>\n        <div class="bmv-td">Monte-Carlo scenarios + &alpha;-CVaR linear program — scales to large, multi-derivative portfolios.</div>\n        <div class="bmv-foot"><span class="bmv-tag">best for many assets · beta</span><span class="bmv-arw">&rarr;</span></div>\n      </div>\n    </a>\n    <a class="bmv-tile purple" href="?view=backtest" target="_self">\n      <div class="bmv-thumb"><img src="__BT__"></div>\n      <div class="bmv-body">\n        <div class="bmv-thead"><span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></span><span class="bmv-tt">Backtest</span></div>\n        <div class="bmv-td">Out-of-sample walk-forward of the <b>Optimiser\'s</b> portfolios, derivative marked to market.</div>\n        <div class="bmv-badge">&#8627; realised alpha &amp; beta vs a benchmark</div>\n        <div class="bmv-foot"><span class="bmv-tag">performance check</span><span class="bmv-arw">&rarr;</span></div>\n      </div>\n    </a>\n    <a class="bmv-tile teal" href="?view=portfolio" target="_self">\n      <div class="bmv-thumb"><img src="__LP__"></div>\n      <div class="bmv-body">\n        <div class="bmv-thead"><span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2dd4bf" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="11" width="3" height="6"/><rect x="12" y="7" width="3" height="10"/><rect x="17" y="4" width="3" height="13"/></svg></span><span class="bmv-tt">Live Portfolio</span></div>\n        <div class="bmv-td">Build a portfolio, save it, and track its return, risk level, alpha and beta over time.</div>\n        <div class="bmv-badge">&#8627; stress-test against historical crises</div>\n        <div class="bmv-foot"><span class="bmv-tag">build · save · track</span><span class="bmv-arw">&rarr;</span></div>\n      </div>\n    </a>\n  </div>\n  <div class="bmv-label">Reference</div>\n  <div class="bmv-tiles ref">\n    <a class="bmv-tile slate" href="?view=about" target="_self">\n      <span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9aa7bd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></span>\n      <div><div class="bmv-tt">About</div><div class="bmv-td">Methods, framework and research.</div></div>\n    </a>\n    <a class="bmv-tile slate" href="?view=glossary" target="_self">\n      <span class="bmv-ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9aa7bd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg></span>\n      <div><div class="bmv-tt">Glossary <span class="bmv-aipill">AI-powered</span></div><div class="bmv-td">VaR, ES, &alpha;-CVaR, copulas — plus natural-language Q&amp;A.</div></div>\n    </a>\n    <a class=\"bmv-tile slate\" href=\"app/static/Beyond_Mean_Variance_Portfolio_Optimiser_User_Guide.pdf\" target=\"_blank\">\n      <span class=\"bmv-ico\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#9aa7bd\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z\"/><polyline points=\"14 2 14 8 20 8\"/><line x1=\"16\" y1=\"13\" x2=\"8\" y2=\"13\"/><line x1=\"16\" y1=\"17\" x2=\"8\" y2=\"17\"/><polyline points=\"10 9 9 9 8 9\"/></svg></span>\n      <div><div class=\"bmv-tt\">User Guide</div><div class=\"bmv-td\">Step-by-step tour of the app (PDF download).</div></div>\n    </a>\n  </div>\n</div>'

_NAV_FOOTER = (
    '<div style="text-align:center;color:#556a8a;font-size:.78rem;margin-top:2.2rem;padding:.6rem 0 1rem">'
    'Built by <b style="color:#7d8aa0">Sami Jeddou</b> &nbsp;·&nbsp; '
    '<a href="?view=about" target="_self" style="color:#7fb3e8;text-decoration:none">About &amp; contact</a> &nbsp;·&nbsp; '
    '<a href="https://www.linkedin.com/in/sami-jeddou-25787a404" target="_blank" style="color:#7fb3e8;text-decoration:none">Connect on LinkedIn</a>'
    '<div style="margin-top:.5rem;color:#46566f;font-size:.72rem">\u00a9 2026 Sami Jeddou \u00b7 All rights reserved</div>'
    '</div>'
)

def _render_home():
    a = _home_assets(_home_assets_key())
    html = _HOME_HTML.replace("__OPT__", a["OPT"]).replace("__MC__", a["MC"]).replace("__BT__", a["BT"]).replace("__RP__", a["RP"]).replace("__TKR__", a["TKR"]).replace("__LP__", a["LP"])
    try:   # cache-bust the static User Guide link by file mtime so updates aren't served from browser cache
        _ugv = int(_os.path.getmtime("static/Beyond_Mean_Variance_Portfolio_Optimiser_User_Guide.pdf"))
    except Exception:
        _ugv = 0
    html = html.replace('User_Guide.pdf"', f'User_Guide.pdf?v={_ugv}"')
    st.markdown(_HOME_CSS, unsafe_allow_html=True)   # styles only (own call)
    st.markdown(html, unsafe_allow_html=True)        # cards only (own call)
    st.markdown(_NAV_FOOTER, unsafe_allow_html=True)

def _go_home():
    st.query_params["view"] = "home"

def _apply_risk_profile(engine, h_pct, a_pct, l_pct):
    """Push the questionnaire's mapped parameters into the chosen engine's sliders and
    jump to that engine. Runs as a button on_click callback (before the next rerun), so
    writing the widget-keyed session_state is safe.
      engine='grid'     -> grid optimiser uses (H, alpha)
      engine='scalable' -> scalable CVaR engine uses (alpha, L)"""
    if engine == "scalable":
        st.session_state["mc_alpha"] = int(a_pct)
        st.session_state["mc_L"] = int(l_pct)
        st.query_params["view"] = "scalable"
    else:
        st.session_state["grid_H_pct"] = int(h_pct)
        st.session_state["grid_alpha_pct"] = int(a_pct)
        st.query_params["view"] = "optimiser"

def _rp_gauge_svg(score, band):
    """Semicircular risk-tolerance speedometer: five equal colour bands (green→red)
    for Low / Below-average / Average / Above-average / High, with a needle pointing
    to the user's band (nudged by score within the band) and a centred read-out."""
    import math as _math
    bands = ["Low", "Below-average", "Average / moderate", "Above-average", "High"]
    short = ["Low", "Below-avg", "Average", "Above-avg", "High"]
    colors = ["#16a34a", "#84cc16", "#f5b942", "#f97316", "#dc2626"]
    ranges = [(13, 18), (19, 22), (23, 28), (29, 32), (33, 47)]
    cx, cy, R, w = 170.0, 150.0, 120.0, 26.0

    def pt(ang, r):
        a = _math.radians(ang)
        return (cx + r * _math.cos(a), cy - r * _math.sin(a))

    # Smooth green->red gradient along the arc: many thin slices with interpolated
    # colours. Anchors sit at the band centres so each band still reads as its colour
    # while the boundaries blend seamlessly.
    def _hx(c):
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))

    def _ramp(t):
        stops = [(0.1, colors[0]), (0.3, colors[1]), (0.5, colors[2]),
                 (0.7, colors[3]), (0.9, colors[4])]
        if t <= stops[0][0]:
            return colors[0]
        if t >= stops[-1][0]:
            return colors[4]
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                r0, g0, b0 = _hx(c0)
                r1, g1, b1 = _hx(c1)
                return f'#{round(r0+(r1-r0)*f):02x}{round(g0+(g1-g0)*f):02x}{round(b0+(b1-b0)*f):02x}'
        return colors[4]

    NSEG = 80
    segs = []
    for i in range(NSEG):
        a0 = 180.0 - 180.0 * i / NSEG
        a1 = max(0.0, 180.0 - 180.0 * (i + 1.6) / NSEG)   # slight overlap avoids seams
        x0, y0 = pt(a0, R)
        x1, y1 = pt(a1, R)
        cseg = _ramp((i + 0.5) / NSEG)
        segs.append(f'<path d="M {x0:.2f} {y0:.2f} A {R:.0f} {R:.0f} 0 0 1 {x1:.2f} {y1:.2f}" '
                    f'fill="none" stroke="{cseg}" stroke-width="{w:.0f}"/>')
    ranges_txt = ["13–18", "19–22", "23–28", "29–32", "33–47"]
    labels = []
    for i, lab in enumerate(short):
        lx, ly = pt(162 - 36 * i, R + 34)   # well outside the arc for legibility
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#aeb9c9" font-size="11" font-weight="600" '
                      f'text-anchor="middle">{lab}'
                      f'<tspan x="{lx:.1f}" dy="12.5" fill="#8b97a8" font-size="9" font-weight="400">{ranges_txt[i]}</tspan>'
                      f'</text>')
    try:
        b = bands.index(band)
    except ValueError:
        b = 0
    lo, hi = ranges[b]
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (score - lo) / (hi - lo)))
    needle_ang = (180 - 36 * b) - (4 + frac * 28)
    nx, ny = pt(needle_ang, R)                            # value-marker centred ON the coloured band
    _a = _math.radians(needle_ang)
    _ux, _uy = _math.cos(_a), -_math.sin(_a)              # unit vector hub -> badge
    _px, _py = -_uy, _ux                                  # perpendicular
    _rtip = R - 12.5                                      # arrow tip just touches the badge edge
    _L, _W = 11.0, 6.0                                    # arrowhead length / half-width
    _tx, _ty = cx + _rtip * _ux, cy + _rtip * _uy
    _bx, _by = cx + (_rtip - _L) * _ux, cy + (_rtip - _L) * _uy
    _ncol = "#d8dee9"   # neutral silver needle/arrow (distinct from the gradient colours)
    needle = (f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{_bx:.1f}" y2="{_by:.1f}" stroke="{_ncol}" '
              f'stroke-width="3.2" stroke-linecap="round"/>'
              f'<polygon points="{_tx:.1f},{_ty:.1f} {_bx + _W * _px:.1f},{_by + _W * _py:.1f} '
              f'{_bx - _W * _px:.1f},{_by - _W * _py:.1f}" fill="{_ncol}"/>'
              f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="7" fill="{_ncol}"/>')
    # value marker at the needle tip showing the assessment score
    _bcol = colors[b]                                    # band colour for the badge + label
    tip = (f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="12.5" fill="{_bcol}" stroke="#0d1117" stroke-width="2.4"/>'
           f'<text x="{nx:.1f}" y="{ny:.1f}" fill="#111111" font-size="11.5" font-weight="700" '
           f'text-anchor="middle" dominant-baseline="central">{int(score)}</text>')
    readout = (f'<text x="{cx:.0f}" y="{cy + 34:.0f}" fill="{_bcol}" font-size="16" font-weight="700" '
               f'text-anchor="middle">{short[b]}</text>'
               f'<text x="{cx:.0f}" y="{cy + 51:.0f}" fill="#9aa7bd" font-size="10.5" '
               f'text-anchor="middle">assessment score {int(score)} of 47</text>')
    return ('<svg viewBox="-12 -22 364 236" width="100%" style="max-width:400px;display:block;margin:.2rem auto .4rem">'
            + "".join(segs) + "".join(labels) + needle + tip + readout + '</svg>')

import time as _time
def _gauth_cfg():
    """Google OAuth config from secrets, or {} if not configured."""
    try:
        c = dict(st.secrets.get("google_oauth", {}))
    except Exception:
        c = {}
    return c if (c.get("client_id") and c.get("client_secret") and c.get("redirect_uri")) else {}

# ── Google OAuth callback (runs before view routing, on any page) ──
_gcfg = _gauth_cfg()
if _gcfg and ("code" in st.query_params) and ("state" in st.query_params):
    _pend = _gd.pending_pop(st.query_params.get("state"))
    _ret = "portfolio"
    if _pend:
        try:
            _tok = _gd.exchange_code(_gcfg["client_id"], _gcfg["client_secret"],
                                     st.query_params.get("code"), _gcfg["redirect_uri"],
                                     _pend.get("verifier"))
            st.session_state["gd_tokens"] = _tok
            st.session_state["gd_user"] = _gd.userinfo(_tok.get("access_token", ""))
            st.session_state.pop("gd_error", None)
            st.session_state.pop("gd_authurl", None)
            _ret = _pend.get("return_view", "portfolio")
        except Exception as _e:
            st.session_state["gd_error"] = "Sign-in failed: %s" % _e
    else:
        st.session_state["gd_error"] = "Sign-in session expired — please try again."
    st.query_params.clear()
    st.query_params["view"] = _ret
    st.rerun()

def _gd_token():
    """Current valid Drive access token (refreshing if near expiry), or None."""
    tok = st.session_state.get("gd_tokens")
    if not tok or not _gcfg:
        return None
    if _time.time() - tok.get("obtained_at", 0) > (tok.get("expires_in", 3600) - 120):
        if tok.get("refresh_token"):
            try:
                new = _gd.refresh_token(_gcfg["client_id"], _gcfg["client_secret"], tok["refresh_token"])
                new.setdefault("refresh_token", tok["refresh_token"])
                st.session_state["gd_tokens"] = new
                tok = new
            except Exception:
                pass
    return tok.get("access_token")

def _gd_friendly(_e):
    """Turn a raw Drive/OAuth exception into a short, human-readable hint."""
    _s = str(_e)
    if "401" in _s or "Unauthorized" in _s or "invalid_grant" in _s or "invalid_token" in _s:
        return "your Google session has expired — click Sign out, then sign in again."
    if "403" in _s or "insufficient" in _s.lower():
        return "Google denied access — re-check the app's Drive permission on sign-in."
    if "429" in _s:
        return "Google is rate-limiting requests — wait a moment and try again."
    return "Google Drive is unavailable right now — please try again in a moment."

_view = st.query_params.get("view", "home")
if _view not in _VIEWS:
    _view = "home"

# Uppercase all button labels + centre all AI-powered boxes on the three tool
# pages (Grid, Scalable, Backtest)
if _view in ("optimiser", "scalable", "backtest", "about", "riskprofile", "ticker", "portfolio"):
    st.markdown(
        "<style>section[data-testid='stMain'] .stButton button,"
        "section[data-testid='stMain'] .stDownloadButton button,"
        "section[data-testid='stMain'] [data-testid='stFormSubmitButton'] button,"
        "section[data-testid='stMain'] [data-testid='stFileUploaderDropzone'] button,"
        "section[data-testid='stSidebar'] .stButton button,"
        "section[data-testid='stSidebar'] .stDownloadButton button,"
        "section[data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'] button"
        "{text-transform:uppercase}"
        "section[data-testid='stMain'] [data-testid='stBaseLinkButton-primary'],"
        "section[data-testid='stMain'] [data-testid='stBaseLinkButton-primary'] *"
        "{text-transform:uppercase !important;color:#0d1117 !important}"
        "section[data-testid='stMain'] details[style*='rgba(74, 158, 255, 0.1)'],"
        "section[data-testid='stSidebar'] details[style*='rgba(74, 158, 255, 0.1)']"
        "{margin-left:auto !important;margin-right:auto !important;max-width:290px;box-sizing:border-box}"
        "section[data-testid='stMain'] details[style*='rgba(74, 158, 255, 0.1)'] summary,"
        "section[data-testid='stSidebar'] details[style*='rgba(74, 158, 255, 0.1)'] summary"
        "{text-align:center}"
        "section[data-testid='stMain'] [data-testid='stFormSubmitButton']{display:flex;justify-content:center}"
        "section[data-testid='stMain'] [id^='sec-about-']{scroll-margin-top:90px}"
        "section[data-testid='stMain'] [id^='sec-about-'],section[data-testid='stMain'] [id^='sec-about-'] *{color:#E3C77E !important}</style>",
        unsafe_allow_html=True)

if _view == "home":
    _render_home()
    st.stop()

# A section is selected: hide the optimiser sidebar on views that don't use it,
# and show a discreet back-to-launcher control.
# Push content clear of Streamlit's fixed top header so the back button isn't clipped.
st.markdown(
    "<style>[data-testid='stMainBlockContainer'],section.main>.block-container,"
    ".block-container{padding-top:3.5rem !important}"
    "@media (max-width:740px){"
    "section[data-testid='stMain'] div[data-testid='stVerticalBlockBorderWrapper']:has(.bmv-banner):has(h2)"
    "{position:static !important;top:auto !important;box-shadow:none !important;border-bottom:none !important;"
    "padding:.2rem 0 .4rem !important;margin-bottom:.3rem !important}"
    "section[data-testid='stMain'] [data-testid='stMainBlockContainer']{padding-top:1rem !important}"
    "section[data-testid='stMain'] [style*='100% - 570px']{max-width:100% !important}"
    "}"
    "section[data-testid='stSidebar'] [data-testid='stSidebarHeader'],"
    "section[data-testid='stSidebar'] [data-testid='stSidebarCollapseButton'],"
    "section[data-testid='stSidebar'] [data-testid='stBaseButton-headerNoPadding']"
    "{z-index:1000 !important;position:relative !important}"
    "</style>",
    unsafe_allow_html=True)
if _view != "optimiser":
    st.markdown(
        "<style>section[data-testid='stSidebar'],"
        "[data-testid='stSidebarCollapsedControl']{display:none!important;}</style>",
        unsafe_allow_html=True)
    if _view not in ("scalable", "backtest", "about", "glossary", "riskprofile", "ticker", "portfolio"):
        _bk, _bk_rest = st.columns([1, 6])
        with _bk:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    with st.container():
        st.markdown('<style>section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.sb-stick){position:sticky;top:0;z-index:100;background:#0d1117;padding:47px 0 .45rem;margin-top:-47px;box-shadow:0 8px 10px -8px rgba(0,0,0,.85)}</style><span class="sb-stick"></span>', unsafe_allow_html=True)
        if _view == "optimiser":
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        st.markdown('<h3 style="text-align:center;margin-bottom:0">⚙️ <span style="color:#E3C77E">Optimisation Parameters</span></h3><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin-top:-.55rem">▼</div>', unsafe_allow_html=True)
    st.divider()

    # ── 1. Data source ────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-header" id="sec1hdr"><span style="display:inline-block;background:#E3C77E;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">1</span><span style="display:block"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>PORTFOLIO DATA</span></div>', unsafe_allow_html=True)
        st.markdown(
            '<details style="background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;padding:.4rem .8rem;margin:.3rem 0 .6rem 0;font-size:.82rem">'
            '<summary style="cursor:pointer;color:#79b6ff;font-weight:600;list-style:none">✨ AI-powered: How these inputs are built</summary>'
            '<div style="color:#aebccd;margin-top:.4rem">From the prices of your chosen securities, the app computes period returns, cleans them (drops stale rows, winsorises ±5σ outliers), and annualises them into the three inputs the optimiser needs — expected returns, volatilities, and the correlation matrix (×252 for daily data, ×12 for monthly). These primary securities are assumed <b>multivariate normal</b>, the foundation of the framework; any derivative is then priced by Black-Scholes on the <b>highest-volatility security</b> and layered on top, which is how non-normal payoffs enter an otherwise Gaussian portfolio.</div></details>',
            unsafe_allow_html=True)
        data_mode = st.radio("Data source",
            ["Default (3-asset sample case)",
             "Live market data (Yahoo Finance)",
             "Enter manually",
             "Upload CSV"],
            index=0, label_visibility="collapsed", key="data_mode")

        means_in=DEFAULT_MEANS[:]; sigs_in=DEFAULT_SIGS[:]
        corr_in=[r[:] for r in DEFAULT_CORR]; names_in=DEFAULT_NAMES[:]
        last_prices={}; data_valid=True

        if data_mode=="Default (3-asset sample case)":
            st.markdown('<div class="ok-box">✓ Default base case loaded — Means: 5%, 10%, 25% | Std devs: 5%, 20%, 50%</div>',
                        unsafe_allow_html=True)

        elif data_mode=="Live market data (Yahoo Finance)":
            ticker_str=st.text_input("Ticker symbols (comma-separated)",
                                      value="AAPL, MSFT, JPM",
                                      placeholder="e.g. AAPL, MSFT, BNP.PA, GS")
            col1,col2=st.columns(2)
            d_start=col1.date_input("From", value=date(2020,1,1))
            d_end  =col2.date_input("To",   value=date.today()-timedelta(days=1))
            freq   =st.radio("Return frequency",["Daily","Monthly"],horizontal=True)
            st.session_state['_live_period'] = f"{d_start.isoformat()} → {d_end.isoformat()}"
            st.session_state['_live_freq'] = freq
            fetch_btn=st.button(":material/cloud_download: Fetch data", use_container_width=True, key="fetch_btn")
            if fetch_btn:
                tickers=[t.strip().upper() for t in ticker_str.split(",") if t.strip()]
                with st.spinner(f"Fetching {len(tickers)} tickers from Yahoo Finance..."):
                    m,s,c,n,lp,err,cleaning=fetch_tickers(tickers,d_start,d_end,freq)
                if err:
                    st.error(f"Fetch failed: {err}"); data_valid=False
                else:
                    st.session_state["live_data"]=(m,s,c,n,lp)
                    st.markdown(f'<div class="ok-box">✓ Loaded: {", ".join(n)} '                            f'({cleaning.get("observations","?")} observations after cleaning)</div>',
                                unsafe_allow_html=True)
                    if cleaning.get("stale_rows_removed"):
                        st.markdown(f'<div class="warn-box">⚠️ {cleaning["stale_rows_removed"]} stale price rows removed.</div>',
                                    unsafe_allow_html=True)
                    if cleaning.get("outliers_winsorised"):
                        st.markdown(f'<div class="warn-box">⚠️ {cleaning["outliers_winsorised"]} outlier returns winsorised (±5σ).</div>',
                                    unsafe_allow_html=True)
                    if cleaning.get("warning"):
                        st.markdown(f'<div class="warn-box">⚠️ {cleaning["warning"]}</div>',
                                    unsafe_allow_html=True)
            if "live_data" in st.session_state:
                means_in,sigs_in,corr_in,names_in,last_prices=st.session_state["live_data"]
                factor_label="annualised" if freq=="Daily" else "annualised (monthly)"
                with st.expander("Preview statistics"):
                    df_prev=pd.DataFrame({
                        "Asset":names_in,
                        f"Mean ({factor_label})":[f"{m*100:.2f}%" for m in means_in],
                        "Std dev":[f"{s*100:.2f}%" for s in sigs_in]})
                    st.dataframe(df_prev,hide_index=True)
                    st.markdown("**Correlation matrix**")
                    corr_df=pd.DataFrame(corr_in,index=names_in,columns=names_in)
                    st.dataframe(corr_df.round(3))
            else:
                data_valid=False

        elif data_mode=="Enter manually":
            n_assets=st.number_input("Number of securities",2,10,3,1)
            names_in,means_in,sigs_in=[],[],[]
            st.markdown("**Returns (annualised)**")
            for i in range(n_assets):
                c1,c2,c3=st.columns([2,1,1])
                nm=c1.text_input(f"Name",value=f"Asset {i+1}",key=f"nm_{i}")
                mn=c2.number_input("Mean%",value=DEFAULT_MEANS[i]*100 if i<3 else 10.0,
                                    key=f"mn_{i}",format="%.1f")/100
                sg=c3.number_input("Std%", value=DEFAULT_SIGS[i]*100  if i<3 else 20.0,
                                    key=f"sg_{i}",format="%.1f")/100
                names_in.append(nm); means_in.append(mn); sigs_in.append(sg)
            st.markdown("**Correlations**")
            corr_in=[[1.0]*n_assets for _ in range(n_assets)]
            for i in range(n_assets):
                for j in range(i+1,n_assets):
                    dv=DEFAULT_CORR[i][j] if i<3 and j<3 else 0.0
                    v=st.slider(f"ρ({names_in[i]}, {names_in[j]})",-1.0,1.0,dv,0.05,
                                 key=f"cr_{i}_{j}")
                    corr_in[i][j]=corr_in[j][i]=v
            cv=corr_to_cov(sigs_in,corr_in)
            if np.any(np.linalg.eigvalsh(cv)<-1e-8):
                st.error("⚠️ Correlation matrix not positive semi-definite."); data_valid=False

        elif data_mode=="Upload CSV":
            st.markdown('<div style="background:rgba(255,255,255,.05);border:1px solid #34527a;border-radius:8px;padding:.6rem 1rem;margin-bottom:.5rem;color:#c0c8d8;font-size:.82rem">'
                        '📋 <b>Format:</b> First col = dates, remaining cols = asset prices.</div>',
                        unsafe_allow_html=True)
            sample="""Date,Low_Risk,Mid_Risk,High_Risk
    2020-01-02,100,100,100
    2020-01-03,100.05,100.15,100.40
    2020-01-06,100.08,100.30,100.85
    2020-01-07,100.12,100.10,101.20
    2020-01-08,100.09,100.45,100.60"""
            st.download_button("⬇ Sample CSV",sample,"sample.csv","text/csv",key="sample_csv_btn")
            uploaded=st.file_uploader("Upload CSV",type=["csv"])
            if uploaded:
                try:
                    means_in,sigs_in,corr_in,names_in=parse_csv(uploaded)
                    st.markdown(f'<div class="ok-box">✓ {len(means_in)} assets: '
                                f'{", ".join(names_in)}</div>',unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Parse error: {e}"); data_valid=False
            else:
                data_valid=False

        # Method notice
        n_sec_total=len(means_in)
        if n_sec_total>=5:
            st.markdown('<div class="warn-box">⚡ 5+ securities detected — '
                        'differential evolution optimizer will be used automatically.</div>',
                        unsafe_allow_html=True)

    st.divider()

    # ── 2. Derivative ─────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-header" id="sec2hdr"><span style="display:inline-block;background:#E3C77E;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">2</span><span style="display:block"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>DERIVATIVE / STRUCTURED PRODUCT</span></div>', unsafe_allow_html=True)
        der_label_sel=st.selectbox("Type",list(PREDEFINED_DERIVATIVES.keys()),
                                    index=0,label_visibility="collapsed",key="der_label_sel")
        der_type=PREDEFINED_DERIVATIVES[der_label_sel]
        der_params={}

        # Payoff diagram + AI explanation render here — directly under the dropdown,
        # above the parameters. Filled in after the sliders below set the params.
        _diag_box = st.container()

        # Underlying selector (shown for all non-None derivative types)
        if der_type is not None:
            underlying_idx=st.selectbox(
                "Underlying security",
                options=list(range(len(names_in))),
                format_func=lambda i: names_in[i],
                index=min(len(names_in)-1, 2))
            der_params["underlying_idx"]=underlying_idx

            # Volatility — default = std dev of underlying security (model-consistent)
            auto_vol=sigs_in[underlying_idx]
            vol_override=st.number_input(
                "Volatility (annualised %)",
                value=round(auto_vol*100,1), min_value=1.0, max_value=200.0,
                format="%.1f", step=0.5) / 100
            der_params["vol"]=vol_override
            st.caption(f"Default: {auto_vol*100:.1f}% (std dev of {names_in[underlying_idx]}). Override for implied vol.")

            rf=st.number_input("Risk-free rate (%)",value=3.0,min_value=0.0,
                                max_value=20.0,format="%.1f",step=0.1)/100
            mat=st.slider("Maturity (years)",0.25,3.0,1.0,0.05)
            der_params["r"]=rf; der_params["T"]=mat

        if der_type in ("put","call","straddle"):
            der_params["strike"]=st.slider(
                "Strike (fraction of spot)",0.5,2.0,
                1.4 if der_type=="put" else (1.2 if der_type=="call" else 0.7),0.05)
        elif der_type in ("safety_collar","aggressive_collar"):
            der_params["strike_p"]=st.slider("Put strike",0.5,1.5,1.2,0.05)
            der_params["strike_c"]=st.slider("Call strike",1.0,2.0,1.6,0.05)
        elif der_type=="strangle":
            der_params["strike_kp"]=st.slider("Put strike (Kp)",0.5,1.2,0.8,0.05)
            der_params["strike_kc"]=st.slider("Call strike (Kc)",0.8,1.5,0.9,0.05)
        elif der_type in ("cgn_uncapped","cgn_capped"):
            der_params["floor"]        =st.slider("Floor (%)",0.0,10.0,1.0,0.5)/100
            der_params["participation"]=st.slider("Participation (%)",50,150,100,10)/100
            der_params["premium"]      =st.slider("Premium (%)",0.0,20.0,0.0,1.0)/100
            if der_type=="cgn_capped":
                der_params["cap"]=st.slider("Cap (%)",5.0,50.0,20.0,5.0)/100
        elif der_type=="barrier_m":
            der_params["M"]         =st.slider("Barrier M (%)",10,60,40,5)/100
            der_params["premium_bm"]=st.slider("Premium (%)",0.0,20.0,10.0,1.0)/100

        elif der_type=="bull_call_spread":
            der_params["k1"]=st.slider("Lower call strike (K₁)",0.7,1.3,1.0,0.05)
            der_params["k2"]=st.slider("Upper call strike (K₂)",0.9,2.0,1.3,0.05)
            st.caption("Bullish, capped. Keep K₂ > K₁.")
        elif der_type=="bear_put_spread":
            der_params["k1"]=st.slider("Upper put strike (K₁)",0.7,1.3,1.0,0.05)
            der_params["k2"]=st.slider("Lower put strike (K₂)",0.4,1.1,0.8,0.05)
            st.caption("Bearish / cheaper hedge, capped. Keep K₂ < K₁.")
        elif der_type=="butterfly_call":
            der_params["center"]=st.slider("Centre strike",0.7,1.4,1.0,0.05)
            der_params["width"] =st.slider("Wing width",0.05,0.5,0.15,0.05)
            st.caption("Pays most when the underlying finishes near the centre strike.")
        elif der_type=="condor_call":
            der_params["center"]=st.slider("Centre strike",0.7,1.4,1.0,0.05)
            der_params["w_in"] =st.slider("Inner half-width",0.05,0.30,0.10,0.05)
            der_params["w_out"]=st.slider("Outer half-width",0.15,0.60,0.25,0.05)
            st.caption("Flat maximum payoff between the inner strikes. Keep outer > inner.")
        elif der_type=="reverse_convertible":
            der_params["kp"]     =st.slider("Put strike (Kp)",0.5,1.1,0.9,0.05)
            der_params["premium"]=st.slider("Issuer premium (%)",0.0,20.0,0.0,1.0)/100
            st.caption("High coupon, capped upside; principal at risk below Kp.")
        elif der_type=="discount_certificate":
            der_params["kc"]     =st.slider("Cap — call strike (Kc)",1.0,1.8,1.2,0.05)
            der_params["premium"]=st.slider("Issuer premium (%)",0.0,20.0,0.0,1.0)/100
            st.caption("Underlying bought at a discount, upside capped at Kc.")
        elif der_type=="outperformance_certificate":
            der_params["k"]      =st.slider("Participation strike (K)",0.8,1.5,1.0,0.05)
            der_params["premium"]=st.slider("Issuer premium (%)",0.0,20.0,0.0,1.0)/100
            st.caption("Full downside, geared (>100%) upside above K.")

        elif der_type=="custom":
            st.markdown("**Build your structured product**")
            st.markdown("*Add components one by one:*")
            if "components" not in st.session_state:
                st.session_state["components"]=[]

            with st.expander("➕ Add component"):
                ct=st.selectbox("Component type",COMPONENT_TYPES,key="ct_sel")
                if "zcb" not in ct:
                    k_val=st.number_input("Strike (fraction of spot)",
                                           0.5,2.0,1.0,0.05,key="k_inp")
                else:
                    k_val=1.0
                notional=st.number_input("Notional",0.01,10.0,1.0,0.1,key="n_inp")
                mat_c=st.number_input("Maturity (years)",0.25,3.0,
                                        der_params.get("T",1.0),0.05,key="mc_inp")
                if st.button("Add component"):
                    st.session_state["components"].append({
                        "type":ct,"strike":k_val,
                        "notional":notional,"maturity":mat_c})

            if st.session_state["components"]:
                st.markdown("**Current components:**")
                for i,c in enumerate(st.session_state["components"]):
                    cols=st.columns([4,1])
                    label=f"{c['type']} | K={c['strike']} | N={c['notional']} | T={c['maturity']}y"
                    cols[0].markdown(f"`{label}`")
                    if cols[1].button("✕",key=f"rm_{i}"):
                        st.session_state["components"].pop(i)
                        st.rerun()
                der_params["components"]=st.session_state["components"]

                # Live payoff diagram
                if len(der_params["components"])>0:
                    vol_c=der_params.get("vol",sigs_in[der_params.get("underlying_idx",0)])
                    fig_pay=plot_payoff(
                        der_params["components"],vol_c,1.0,
                        der_params.get("r",0.03),der_params.get("T",1.0),
                        names_in[der_params.get("underlying_idx",0)])
                    st.pyplot(fig_pay,use_container_width=True)
                    plt.close(fig_pay)
            else:
                data_valid=False
                st.info("Add at least one component to continue.")

        # Render the payoff diagram + AI explanation into the box reserved directly
        # under the dropdown (above the parameters). Computed here so der_params is set.
        if der_type is not None and der_type != "custom":
            with _diag_box:
                if "underlying_idx" in der_params:
                    try:
                        _cfg = build_der_config(der_type, der_params,
                                                np.array(sigs_in), der_params["underlying_idx"])
                        if _cfg:
                            _cfg["r"] = der_params.get("r", 0.03)
                            _cfg["T"] = der_params.get("T", 1.0)
                            _figp = plot_named_payoff(_cfg, names_in[der_params["underlying_idx"]])
                            st.pyplot(_figp, use_container_width=True)
                            plt.close(_figp)
                    except Exception:
                        pass
                st.markdown(
                    f'<details style="background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;padding:.4rem .8rem;margin:.3rem 0;font-size:.82rem">'
                    f'<summary style="cursor:pointer;color:#79b6ff;font-weight:600;list-style:none">✨ AI-powered: What is this instrument?</summary>'
                    f'<div style="color:#aebccd;margin-top:.4rem">{get_explanation(der_label_sel)}</div></details>',
                    unsafe_allow_html=True)

    st.divider()

    # ── 3. Constraint ─────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-header" id="sec3hdr"><span style="display:inline-block;background:#E3C77E;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">3</span><span style="display:block"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>MENTAL-ACCOUNT CONSTRAINT</span></div>', unsafe_allow_html=True)

        # AI explanation renders here — right after the title, above the choices.
        _con_box = st.container()

        # VaR / ES toggle
        constraint_type = st.radio(
            "Constraint type",
            ["VaR — Value at Risk", "ES — Expected Shortfall", "ES — Rigorous (beyond thesis)"],
            index=0, horizontal=True)
        use_es = constraint_type.startswith("ES")
        use_es_rigorous = "Rigorous" in constraint_type

        # Fill the box reserved above (under the title) with the matching explanation.
        with _con_box:
            if not use_es:
                _ckey, _clabel = "var", "What is the VaR constraint?"
            else:
                _ckey   = "es_rigorous" if use_es_rigorous else "es"
                _clabel = "What is the ES constraint?"
            st.markdown(
                f'<details style="background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;padding:.4rem .8rem;margin:.3rem 0 .6rem 0;font-size:.82rem">'
                f'<summary style="cursor:pointer;color:#79b6ff;font-weight:600;list-style:none">✨ AI-powered: {_clabel}</summary>'
                f'<div style="color:#aebccd;margin-top:.4rem">{CONSTRAINT_EXPLANATIONS[_ckey]}</div></details>',
                unsafe_allow_html=True)

        # Keys + session-state defaults so the Risk-profile questionnaire can pre-fill
        # these (via _apply_risk_profile); the user can still move the sliders freely.
        st.session_state.setdefault("grid_H_pct", -10)
        H_val = st.slider("Threshold H (%)", -40, -1, step=1, key="grid_H_pct") / 100
        st.markdown(
            '<div style="background:rgba(255,255,255,.05);border:1px solid #3a3a5a;border-radius:6px;'
            'padding:.3rem .8rem;color:#9aa7bd;font-size:.76rem;margin-top:.2rem">'
            'Range extended to -40% to accommodate highly volatile assets '
            '(e.g. cryptocurrencies, emerging market equities).</div>',
            unsafe_allow_html=True)

        if not use_es:
            st.session_state.setdefault("grid_alpha_pct", 5)
            alpha_val = st.slider("Max shortfall probability α (%)", 1, 15, step=1, key="grid_alpha_pct") / 100
            L_val     = None
            # Formula box — white background
            st.markdown(
                '<div style="background:rgba(255,255,255,.05);border:1px solid #3a3a5a;border-radius:6px;'
                'padding:.4rem 1rem;color:#c0c8d8;font-size:.78rem;margin-top:.3rem">'
                'VaR constraint: P(return &lt; H) ≤ α</div>',
                unsafe_allow_html=True)
            # Implied lambda — between formula and AI explanation
            cov_for_lam = corr_to_cov(sigs_in, corr_in)
            lam = implied_lambda(H_val, alpha_val, means_in, cov_for_lam)
            if lam is not None:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.05);border:1px solid #1a6bbf;border-radius:6px;'
                    f'padding:.5rem 1rem;margin-top:.3rem;color:#9ec5ff;font-size:.85rem">'
                    f'<b>Implied risk-aversion λ = {lam:.4f}</b><br>'
                    f'<span style="color:#9aa7bd;font-size:.78rem">'
                    f'MV optimal at λ={lam:.2f} ≡ behavioural optimal at H={H_val:.0%}, α={alpha_val:.0%}'
                    f'</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:rgba(245,158,11,.10);border:1px solid #f59e0b;border-radius:6px;'
                            'padding:.4rem 1rem;color:#f0b860;font-size:.78rem;margin-top:.3rem">'
                            '⚠️ Implied λ not available — the VaR constraint may be too tight or too loose for the current portfolio.</div>',
                            unsafe_allow_html=True)
        else:
            alpha_val = None
            L_val     = st.slider("ES lower bound L (%)", -50, -1, -15, 1) / 100
            # Formula box — white background
            st.markdown(
                '<div style="background:rgba(255,255,255,.05);border:1px solid #3a3a5a;border-radius:6px;'
                'padding:.4rem 1rem;color:#c0c8d8;font-size:.78rem;margin-top:.3rem">'
                'ES constraint: E[return | return &lt; H] ≥ L</div>',
                unsafe_allow_html=True)

        # Implied lambda block already handled above for VaR case
        if use_es:
            pass  # no lambda for ES

    st.divider()

    # ── 4. Grid ───────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-header" id="sec4hdr"><span style="display:inline-block;background:#E3C77E;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">4</span><span style="display:block"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>GRID RESOLUTION</span></div>', unsafe_allow_html=True)
        # Turbo accelerates the VaR path only; hide it when ES is selected. Rigorous
        # ES uses the high-precision (m=51) state space via the fast coarse-to-fine
        # engine, so its resolution is fixed and the selector does not apply.
        _n_total = n_sec_total + (1 if der_type is not None else 0)
        if use_es_rigorous:
            grid_lbl="Rigorous ES — high-precision accuracy (~seconds)"
            m_val,mp_val=51,99
        else:
            # Turbo (coarse-to-fine) only runs on the VaR path with <=4 TOTAL
            # securities — the derivative counts toward the total. For ES, or for
            # 5+ total securities, optimize_portfolio_turbo silently delegates to
            # the original solver (differential evolution), so the "~seconds"
            # promise no longer holds and the label would misrepresent which
            # engine ran. Hide the Turbo option in those cases. This mirrors the
            # engine gate: use_de = (method=='auto' and n_securities >= 5).
            _n_total = n_sec_total + (1 if der_type is not None else 0)
            _hide_turbo = use_es or (_n_total >= 5)
            _res_keys=[k for k in GRID_OPTIONS
                       if not (_hide_turbo and GRID_OPTIONS[k][1]=='turbo')]
            if st.session_state.get("grid_lbl") not in _res_keys:
                st.session_state["grid_lbl"]=_res_keys[0]
            grid_lbl=st.selectbox("Resolution",_res_keys,
                                   label_visibility="collapsed",key="grid_lbl")
            m_val,mp_val=GRID_OPTIONS[grid_lbl]

        # AI-powered grid explanation
        _ai_expl = GRID_EXPLANATIONS.get(grid_lbl, "No explanation available.")
        if _n_total >= 5 and not use_es_rigorous:
            _ai_expl += (" ⚠️ Note — with 5+ instruments the exhaustive grid is too large, so the "
                         "optimiser automatically uses the differential-evolution solver instead; the "
                         "result is a near-optimal approximation, not an exact every-combination search.")
        st.markdown(
            f'<details style="background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;padding:.4rem .8rem;margin:.3rem 0;font-size:.82rem">'        '<summary style="cursor:pointer;color:#79b6ff;font-weight:600;list-style:none">✨ AI-powered: What does this resolution mean?</summary>'        f'<div style="color:#aebccd;margin-top:.4rem">{_ai_expl}</div></details>',
            unsafe_allow_html=True)

        if "Rigorous" in grid_lbl:
            st.markdown('<div class="warn-box">⚡ Fast for small universes. For many assets or CVaR at scale, use the Scalable Monte-Carlo optimiser.</div>',
                        unsafe_allow_html=True)
        elif "Turbo" in grid_lbl:
            st.markdown('<div class="warn-box">⚡ Returns in seconds (VaR, up to 4 assets).</div>',
                        unsafe_allow_html=True)
        elif _n_total >= 5:
            st.markdown('<div class="warn-box">⚠️ With ' + str(_n_total) + ' instruments the exhaustive grid is too large to enumerate, so the optimiser automatically switches to the <b>differential-evolution</b> solver — an automatic guided search. Your <b>' + grid_lbl.split("(")[0].strip() + '</b> choice no longer runs an exact, every-combination grid; the result is a fast, near-optimal approximation. For an exact grid keep to 4 instruments or fewer, or use the Scalable Monte-Carlo optimiser for larger / CVaR portfolios.</div>',
                        unsafe_allow_html=True)
        elif "High precision" in grid_lbl:
            st.markdown('<div class="warn-box">⚠️ The heaviest mode — likely to stall on the free hosted demo. Use Turbo (the default) for fast VaR, the Scalable Monte-Carlo optimiser for larger / CVaR portfolios, or <a href="https://github.com/SamiJeddou/behavioral-portfolio-optimizer/blob/main/Run_Locally_Guide.pdf" target="_blank" rel="noopener" style="color:#fde68a;text-decoration:underline;font-weight:600">run locally</a> for exact results.</div>',
                        unsafe_allow_html=True)
        elif "Standard" in grid_lbl:
            st.markdown('<div class="warn-box">⏱️ Heavy on the free hosted demo — may not finish beyond a few assets. For larger or CVaR portfolios use the Scalable Monte-Carlo optimiser; <a href="https://github.com/SamiJeddou/behavioral-portfolio-optimizer/blob/main/Run_Locally_Guide.pdf" target="_blank" rel="noopener" style="color:#fde68a;text-decoration:underline;font-weight:600">run locally</a> for the exact result.</div>',
                        unsafe_allow_html=True)

    st.divider()
    run_btn=st.button(
        "▶  RUN OPTIMISER",
        type="primary",
        use_container_width=True,
        disabled=not data_valid,
        key="run_opt_btn")

    reset_btn=st.button(
        "↩  RESET / NEW SIMULATION",
        type="secondary",
        use_container_width=True,
        key="reset_sim_btn")

    # No session_state mutations inside sidebar — handled outside


# Handle all session state mutations OUTSIDE sidebar to prevent double-render
if reset_btn:
    for _k in ['_run_active','_needs_compute','_cached_results',
               '_pdf_bytes','_fig_png','_fig_plotly']:
        st.session_state.pop(_k, None)
    st.rerun()

if run_btn:
    st.session_state['_run_active'] = True
    st.session_state['_needs_compute'] = True
    st.session_state.pop('_cached_results', None)
    st.session_state.pop('_pdf_bytes', None)
    st.session_state.pop('_fig_png', None)
    st.rerun()

# ── Session state logic OUTSIDE sidebar to prevent double-render ─────────────
_run_active = st.session_state.get('_run_active', False)
_has_results = st.session_state.get('_cached_results') is not None
_needs_compute = st.session_state.get('_needs_compute', False)

# Detect fresh page load
if _run_active and not _has_results and not _needs_compute:
    st.session_state.pop('_run_active', None)
    _run_active = False

# Reset if key parameters changed since last run
if not run_btn and _run_active and _has_results:
    _cached = st.session_state.get('_cached_results', {})
    _prev_der   = _cached.get('der_label_sel', '__unset__')
    _prev_H     = _cached.get('H_val', None)
    _prev_alpha = _cached.get('_alpha', None)
    _prev_data  = _cached.get('data_mode', '__unset__')
    _prev_grid  = _cached.get('grid_lbl', '__unset__')
    _prev_names    = _cached.get('names_in', [])
    _prev_undl     = _cached.get('underlying_idx', 0)
    _cur_undl      = der_params.get('underlying_idx', 0) if der_type is not None else 0
    _prev_der_params = _cached.get('der_params_fp', '')
    _cur_der_params  = str(sorted(der_params.items())) if der_type is not None else ''
    if (_prev_der != der_label_sel or
        _prev_H != H_val or
        _prev_alpha != alpha_val or
        _prev_data != data_mode or
        _prev_grid != grid_lbl or
        _prev_names != list(names_in) or
        _prev_undl != _cur_undl or
        _prev_der_params != _cur_der_params):
        for _k in ['_run_active','_needs_compute','_cached_results',
                   '_pdf_bytes','_fig_png','_fig_plotly']:
            st.session_state.pop(_k, None)
        _run_active = False
        _has_results = False
        _needs_compute = False

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def show_portfolio_data(names_in, means_in, sigs_in, corr_in):
    st.markdown('<span style="font-size:16px;font-weight:600;color:#E3C77E"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>Portfolio data</span>', unsafe_allow_html=True)
    with st.container():
        hs = "background:rgba(16,185,129,.12);color:#86e0b0;font-weight:bold;padding:6px 10px;text-align:center"
        cs = "background:#1b2330;color:#e7ecf4;padding:5px 10px;border-bottom:1px solid #30363d;text-align:center"
        rows = "".join(
            f"<tr><td style='{cs}'>{names_in[i]}</td>"
            f"<td style='{cs}'>{means_in[i]*100:.2f}%</td>"
            f"<td style='{cs}'>{sigs_in[i]*100:.2f}%</td></tr>"
            for i in range(len(names_in)))
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><th style='{hs}'>Asset</th><th style='{hs}'>Mean return</th>"
            f"<th style='{hs}'>Std deviation</th></tr>{rows}</table>",
            unsafe_allow_html=True)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<span style="font-size:16px;font-weight:600;color:#E3C77E">Correlation matrix</span>', unsafe_allow_html=True)
        n = len(names_in)
        corr_rows = "".join(
            f"<tr><td style='{hs}'>{names_in[i]}</td>"
            + "".join(f"<td style='{cs};text-align:center'>{corr_in[i][j]:.3f}</td>" for j in range(n))
            + "</tr>" for i in range(n))
        col_headers = "".join(f"<th style='{hs};text-align:center'>{names_in[j]}</th>" for j in range(n))
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><th style='{hs}'></th>{col_headers}</tr>{corr_rows}</table>",
            unsafe_allow_html=True)

# ── Finance banner (hidden — removed at user request) ───────────────────────
if False: st.markdown(f'''
<div style="width:100%;background:#020c1b;padding:0;margin:0">

<div style="max-width:900px;margin:0 auto;background:linear-gradient(135deg,#020c1b 0%,#071428 40%,#0a1a35 70%,#020c1b 100%);border-radius:0;overflow:hidden;border:none;display:flex;align-items:stretch;font-family:monospace;margin-bottom:0">
  <div style="flex:1.5;padding:16px 18px;border-right:1px solid #1a3a5c;display:flex;flex-direction:column;justify-content:center;gap:10px">
    <div style="color:#ffffff;font-size:13px;font-weight:700;letter-spacing:0.03em;font-family:Georgia,serif;line-height:1.45">Portfolio Optimiser<br>with Derivatives &amp;<br>Structured Products</div>
    <div style="color:rgba(74,158,255,0.65);font-size:7.5px;letter-spacing:0.22em">BEYOND MEAN-VARIANCE · MENTAL ACCOUNTS FRAMEWORK</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px">
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">LIVE MARKET DATA</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">ALL FINANCIAL INSTRUMENTS</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">AI-POWERED</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">CRYPTO ASSETS</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">VaR / ES</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">MVT/MAT</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">GRID SEARCH · COBYLA</span>
    </div>
  </div>
  <div style="flex:3.2;display:grid;grid-template-columns:repeat(4,1fr)">
    <!-- Col 1: Live market ticker -->
    <div style="border-right:1px solid #1a3a5c;display:flex;flex-direction:column">
      <div style="height:82px;position:relative;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <svg width="100%" height="82" viewBox="0 0 80 82" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="80" height="82" fill="#020c1b"/>
          <rect x="2" y="4" width="76" height="12" fill="rgba(16,185,129,0.08)" rx="1"/>
          <text x="6" y="13" fill="#10b981" font-size="6" font-family="monospace">AAPL</text>
          <text x="34" y="13" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">189.42</text>
          <text x="63" y="13" fill="#10b981" font-size="6" font-family="monospace">+1.2%</text>
          <rect x="2" y="18" width="76" height="12" fill="rgba(239,68,68,0.07)" rx="1"/>
          <text x="6" y="27" fill="#ef4444" font-size="6" font-family="monospace">BTC</text>
          <text x="34" y="27" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">67,340</text>
          <text x="63" y="27" fill="#ef4444" font-size="6" font-family="monospace">-0.8%</text>
          <rect x="2" y="32" width="76" height="12" fill="rgba(16,185,129,0.08)" rx="1"/>
          <text x="6" y="41" fill="#10b981" font-size="6" font-family="monospace">SPY</text>
          <text x="34" y="41" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">521.80</text>
          <text x="63" y="41" fill="#10b981" font-size="6" font-family="monospace">+0.4%</text>
          <rect x="2" y="46" width="76" height="12" fill="rgba(239,68,68,0.07)" rx="1"/>
          <text x="6" y="55" fill="#ef4444" font-size="6" font-family="monospace">ETH</text>
          <text x="34" y="55" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">3,421</text>
          <text x="63" y="55" fill="#ef4444" font-size="6" font-family="monospace">-1.1%</text>
          <path d="M4,74 L10,70 L16,72 L22,66 L28,68 L34,62 L40,58 L46,60 L52,54 L58,50 L64,46 L70,42 L76,38" fill="none" stroke="#4a9eff" stroke-width="1.2" opacity="0.7"/>
          <line x1="0" y1="64" x2="80" y2="64" stroke="#0d2a4a" stroke-width="0.5"/>
        </svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">LIVE MARKET DATA</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#10b981">10,000+</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">tickers · equities · crypto · ETFs</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:75%;height:100%;border-radius:1px;background:#4a9eff"></div></div>
      </div>
    </div>
    <!-- Col 2: Candlesticks -->
    <div style="border-right:1px solid #1a3a5c;display:flex;flex-direction:column">
      <div style="height:82px;position:relative;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <svg width="100%" height="82" viewBox="0 0 80 82" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="80" height="82" fill="#020c1b"/>
          <line x1="0" y1="41" x2="80" y2="41" stroke="#0d2a4a" stroke-width="0.6"/>
          <line x1="7" y1="62" x2="7" y2="74" stroke="#10b981" stroke-width="0.8"/><rect x="3" y="65" width="8" height="7" fill="#10b981" rx="0.5"/>
          <line x1="18" y1="48" x2="18" y2="64" stroke="#10b981" stroke-width="0.8"/><rect x="14" y="52" width="8" height="12" fill="#10b981" rx="0.5"/>
          <line x1="29" y1="46" x2="29" y2="60" stroke="#ef4444" stroke-width="0.8"/><rect x="25" y="50" width="8" height="8" fill="#ef4444" rx="0.5"/>
          <line x1="40" y1="44" x2="40" y2="62" stroke="#ef4444" stroke-width="0.8"/><rect x="36" y="48" width="8" height="14" fill="#ef4444" rx="0.5"/>
          <line x1="51" y1="30" x2="51" y2="48" stroke="#10b981" stroke-width="0.8"/><rect x="47" y="34" width="8" height="12" fill="#10b981" rx="0.5"/>
          <line x1="62" y1="14" x2="62" y2="34" stroke="#10b981" stroke-width="0.8"/><rect x="58" y="18" width="8" height="16" fill="#10b981" rx="0.5"/>
          <line x1="73" y1="16" x2="73" y2="32" stroke="#ef4444" stroke-width="0.8"/><rect x="69" y="20" width="8" height="8" fill="#ef4444" rx="0.5"/>
          <path d="M3,72 C16,64 30,57 44,52 C56,42 66,28 78,20" fill="none" stroke="#f59e0b" stroke-width="1.3" opacity="0.8"/>
          <rect x="3" y="70" width="8" height="8" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="14" y="66" width="8" height="12" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="25" y="72" width="8" height="6" fill="#ef4444" opacity="0.35" rx="0.5"/>
          <rect x="36" y="68" width="8" height="10" fill="#ef4444" opacity="0.35" rx="0.5"/>
          <rect x="47" y="63" width="8" height="15" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="58" y="60" width="8" height="18" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="69" y="70" width="8" height="8" fill="#ef4444" opacity="0.35" rx="0.5"/>
        </svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">RETURN — WITH DERIVATIVE</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#f59e0b">11.4%</div>
        <div style="font-size:8px;margin-top:3px;color:#10b981">+1.2 pp vs no derivative</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:100%;height:100%;border-radius:1px;background:#f59e0b"></div></div>
      </div>
    </div>
    <!-- Col 3: Portfolio weights -->
    <div style="border-right:1px solid #1a3a5c;display:flex;flex-direction:column">
      <div style="height:82px;position:relative;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <svg width="100%" height="82" viewBox="0 0 80 82" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="80" height="82" fill="#020c1b"/>
          <text x="40" y="10" fill="rgba(100,160,220,0.6)" font-size="6" font-family="monospace" text-anchor="middle">PORTFOLIO WEIGHTS</text>
          <text x="6" y="24" fill="rgba(200,220,255,0.6)" font-size="5.5" font-family="monospace">EQ</text>
          <rect x="20" y="18" width="54" height="7" fill="#0d2a4a" rx="1"/><rect x="20" y="18" width="19" height="7" fill="#4a9eff" rx="1" opacity="0.8"/>
          <text x="41" y="24" fill="rgba(74,158,255,0.8)" font-size="5.5" font-family="monospace">35%</text>
          <text x="6" y="36" fill="rgba(200,220,255,0.6)" font-size="5.5" font-family="monospace">BD</text>
          <rect x="20" y="30" width="54" height="7" fill="#0d2a4a" rx="1"/><rect x="20" y="30" width="5" height="7" fill="#a855f7" rx="1" opacity="0.8"/>
          <text x="27" y="36" fill="rgba(168,85,247,0.8)" font-size="5.5" font-family="monospace">10%</text>
          <text x="6" y="48" fill="rgba(245,158,11,0.8)" font-size="5.5" font-family="monospace">CGN</text>
          <rect x="20" y="42" width="54" height="7" fill="#0d2a4a" rx="1"/><rect x="20" y="42" width="30" height="7" fill="#f59e0b" rx="1" opacity="0.85"/>
          <text x="52" y="48" fill="rgba(245,158,11,0.9)" font-size="5.5" font-family="monospace">55%</text>
          <rect x="6" y="56" width="68" height="10" fill="rgba(16,185,129,0.12)" rx="2"/>
          <text x="40" y="63" fill="#10b981" font-size="6" font-family="monospace" text-anchor="middle">P(r &lt; H) ≤ α ✓</text>
          <text x="6" y="76" fill="rgba(100,160,220,0.5)" font-size="5.5" font-family="monospace">No deriv.</text>
          <rect x="36" y="70" width="40" height="5" fill="#0d2a4a" rx="1"/><rect x="36" y="70" width="12" height="5" fill="#4a9eff" rx="1" opacity="0.7"/>
          <text x="50" y="76" fill="rgba(74,158,255,0.7)" font-size="5.5" font-family="monospace">10.2%</text>
        </svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">RETURN — NO DERIV.</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#4a9eff">10.2%</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">H=-10%, α=5%, λ=3.795</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:31%;height:100%;border-radius:1px;background:#4a9eff"></div></div>
      </div>
    </div>
    <!-- Col 4: Real app screenshot -->
    <div style="display:flex;flex-direction:column">
      <div style="height:82px;overflow:hidden;border-bottom:1px solid #0d2a4a;display:flex;align-items:center;justify-content:center">
        <svg width="80" height="82" viewBox="-10 -10 100 100" xmlns="http://www.w3.org/2000/svg"><rect x="-10" y="-10" width="100" height="100" fill="#020c1b"/><path d="M 40.0 8.0 A 32 32 0 1 1 30.1 70.4 L 34.1 58.1 A 19 19 0 1 0 40.0 21.0 Z" fill="#e63946" stroke="#020c1b" stroke-width="1.5"/><text x="81.5" y="46.6" fill="#e63946" font-size="7" font-weight="700" font-family="monospace" text-anchor="middle" dominant-baseline="central">55%</text><path d="M 30.1 70.4 A 32 32 0 0 1 14.1 21.2 L 24.6 28.8 A 19 19 0 0 0 34.1 58.1 Z" fill="#f4a261" stroke="#020c1b" stroke-width="1.5"/><text x="0.1" y="53.0" fill="#f4a261" font-size="7" font-weight="700" font-family="monospace" text-anchor="middle" dominant-baseline="central">30%</text><path d="M 14.1 21.2 A 32 32 0 0 1 40.0 8.0 L 40.0 21.0 A 19 19 0 0 0 24.6 28.8 Z" fill="#2a9d8f" stroke="#020c1b" stroke-width="1.5"/><text x="20.9" y="2.6" fill="#2a9d8f" font-size="7" font-weight="700" font-family="monospace" text-anchor="middle" dominant-baseline="central">15%</text><text x="40" y="38" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace" text-anchor="middle">Portfolio</text><text x="40" y="46" fill="rgba(255,255,255,0.5)" font-size="5.5" font-family="monospace" text-anchor="middle">weights</text><rect x="2" y="68" width="7" height="7" fill="#e63946" rx="1"/><text x="11" y="74" fill="rgba(200,220,255,0.8)" font-size="6" font-family="monospace">CGN</text><rect x="28" y="68" width="7" height="7" fill="#f4a261" rx="1"/><text x="37" y="74" fill="rgba(200,220,255,0.8)" font-size="6" font-family="monospace">EQ</text><rect x="52" y="68" width="7" height="7" fill="#2a9d8f" rx="1"/><text x="61" y="74" fill="rgba(200,220,255,0.8)" font-size="6" font-family="monospace">BD</text></svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.1em;margin-bottom:3px;min-height:18px;line-height:1.15">DERIVATIVES &amp; STRUCTURED PRODUCTS</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#a855f7">16+</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">options · spreads · collars · notes · certificates</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:95%;height:100%;border-radius:1px;background:#a855f7"></div></div>
      </div>
  </div>
</div>
</div>
</div>''', unsafe_allow_html=True)

DONUT_COLORS = ['#e63946','#f4a261','#e9c46a','#2a9d8f','#264653','#023e8a','#e76f51','#457b9d']

def make_donut_svg(weights, labels, colors, size=160):
    """Generate an SVG doughnut chart for portfolio weights."""
    import math
    cx, cy, r_out, r_in = size//2, size//2, size//2 - 8, size//2 - 28
    
    # Filter zero weights
    items = [(w, l, c) for w, l, c in zip(weights, labels, colors) if w > 0.001]
    if not items:
        return ""
    
    total = sum(i[0] for i in items)
    
    def polar(cx, cy, r, angle_deg):
        a = math.radians(angle_deg - 90)
        return cx + r * math.cos(a), cy + r * math.sin(a)
    
    paths = []
    legend_items = []
    angle = 0
    
    for w, label, color in items:
        sweep = (w / total) * 360
        large = 1 if sweep > 180 else 0
        
        x1o, y1o = polar(cx, cy, r_out, angle)
        x2o, y2o = polar(cx, cy, r_out, angle + sweep)
        x1i, y1i = polar(cx, cy, r_in, angle + sweep)
        x2i, y2i = polar(cx, cy, r_in, angle)
        
        path = (f'<path d="M {x1o:.1f} {y1o:.1f} '
                f'A {r_out} {r_out} 0 {large} 1 {x2o:.1f} {y2o:.1f} '
                f'L {x1i:.1f} {y1i:.1f} '
                f'A {r_in} {r_in} 0 {large} 0 {x2i:.1f} {y2i:.1f} Z" '
                f'fill="{color}" stroke="#0d1117" stroke-width="1.5"/>')
        paths.append(path)

        pct = w / total * 100
        # Percentage label OUTSIDE the ring with security color
        if pct >= 5:
            mid_angle = angle + sweep / 2
            r_label = r_out + 14
            lx, ly = polar(cx, cy, r_label, mid_angle)
            paths.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" '
                f'fill="{color}" font-size="10" font-weight="700" '
                f'font-family="sans-serif" text-anchor="middle" dominant-baseline="central">'
                f'{pct:.0f}%</text>'
            )
        # Legend
        short_lbl = label[:12] + "…" if len(label) > 12 else label
        legend_items.append((color, short_lbl, pct))
        
        angle += sweep
    
    svg = (f'<svg width="{size+30}" height="{size+30}" viewBox="-15 -15 {size+30} {size+30}" '
           f'xmlns="http://www.w3.org/2000/svg">'
           + "".join(paths)
           + '</svg>')
    return svg

st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# Backtest engine — out-of-sample validation (pure computation, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════
# Instruments supported in the v1 backtest are those carrying a strictly positive
# net entry premium, so the mark-to-market gross-return path V_t / paid is stable.
# Collars (near-zero / signed net premium), CGN and barrier-M use bespoke
# normalisations and are deferred to a later iteration.
_BT_SUPPORTED = {
    "put", "call", "straddle", "strangle",
    "safety_collar", "aggressive_collar", "cgn_uncapped", "cgn_capped",
    "bull_call_spread", "bear_put_spread", "butterfly_call", "condor_call",
    "reverse_convertible", "discount_certificate", "outperformance_certificate",
}

# fetch_close_prices -> moved to core/markets.py

# stats_from_prices -> moved to core/markets.py

# _bt_legs -> moved to core/pricing.py

# _leg_value -> moved to core/pricing.py

# mtm_gross_path -> moved to core/pricing.py

# _bt_portfolio_path -> moved to core/backtest.py

# _bt_metrics -> moved to core/backtest.py

# _capm_alpha_beta -> moved to core/backtest.py

def plot_backtest_paths(dates, pv1, pv2, label2):
    """Dark-themed cumulative value chart: no-derivative vs with-derivative."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    x = np.asarray(dates)
    ax.axhline(100, color='#30363d', lw=1, ls=(0, (4, 4)))
    ax.plot(x, 100.0 * np.asarray(pv1), color='#10b981', lw=2.2, label='No derivative (P1)')
    if pv2 is not None:
        ax.plot(x, 100.0 * np.asarray(pv2), color='#f59e0b', lw=2.2, label=label2)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.set_ylabel('Portfolio value (entry = 100)', color='#c9d1d9', fontsize=9)
    ax.set_title('Out-of-sample buy-and-hold performance', color='#c9d1d9', fontsize=10, pad=8)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9',
              fontsize=8, loc='best')
    try:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        fig.autofmt_xdate(rotation=0, ha='center')
    except Exception:
        pass
    ax.grid(True, color='#21262d', lw=0.6)
    fig.tight_layout()
    return fig


def plot_backtest_paths_plotly(dates, pv1, pv2, label2):
    """Interactive dark-themed cumulative value chart (Plotly): P1 vs P2."""
    x = list(dates)
    fig = go.Figure()
    fig.add_hline(y=100, line=dict(color='#30363d', width=1, dash='dash'))
    fig.add_trace(go.Scatter(
        x=x, y=(100.0 * np.asarray(pv1, dtype=float)), mode='lines',
        name='No derivative (P1)', line=dict(color='#10b981', width=2.4),
        hovertemplate='<b>No derivative (P1)</b><br>%{x|%d %b %Y}<br>Value: %{y:.2f}<extra></extra>'))
    if pv2 is not None:
        fig.add_trace(go.Scatter(
            x=x, y=(100.0 * np.asarray(pv2, dtype=float)), mode='lines',
            name=label2, line=dict(color='#f59e0b', width=2.4),
            hovertemplate='<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>Value: %{y:.2f}<extra></extra>'))
    fig.update_layout(
        title=dict(text='Out-of-sample portfolio value — buy-and-hold (entry = 100)',
                   font=dict(color='#E3C77E', size=14), x=0.5, xanchor='center'),
        paper_bgcolor='#1b2330', plot_bgcolor='#0e1521', font=dict(color='#c9d1d9'),
        hovermode='x unified', margin=dict(l=10, r=10, t=46, b=10), height=440,
        legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', borderwidth=1,
                    font=dict(color='#ffffff', size=12),
                    x=0.01, y=0.99, xanchor='left', yanchor='top'),
        yaxis=dict(title='Portfolio value (entry = 100)', color='#8b949e', zeroline=False),
        xaxis=dict(color='#8b949e'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#27344e', gridwidth=1, griddash='dot',
                     showline=True, linecolor='#46566f', linewidth=1, mirror=True,
                     showspikes=True, spikethickness=1)
    fig.update_yaxes(showgrid=True, gridcolor='#27344e', gridwidth=1, griddash='dot',
                     showline=True, linecolor='#46566f', linewidth=1, mirror=True,
                     showspikes=True, spikethickness=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Scalable Monte-Carlo + CVaR linear-program engine (Beta)
# Cost is O(S*(N+K)) — linear in securities and derivatives, independent of grid
# resolution. Reuses _bt_legs / _leg_value so every instrument prices exactly as
# in the Backtest tab and the exact grid engine. Approximate (sampling); complements
# the exact grid optimiser rather than replacing it.
# ═══════════════════════════════════════════════════════════════════════════════
from scipy.optimize import linprog as _linprog
from scipy.sparse import coo_matrix as _coo
from scipy.stats import norm as _norm, t as _student_t, chi2 as _chi2

# instruments the MC tab exposes (single/double-strike; collars/CGN/barrier deferred)
MC_DER_TYPES = {
    "Call":                       "call",
    "Put":                        "put",
    "Straddle":                   "straddle",
    "Strangle":                   "strangle",
    "Bull call spread":           "bull_call_spread",
    "Bear put spread":            "bear_put_spread",
    "Safety collar":              "safety_collar",
    "Aggressive collar":          "aggressive_collar",
    "Long butterfly (calls)":     "butterfly_call",
    "Call condor":                "condor_call",
    "Reverse convertible":        "reverse_convertible",
    "Discount certificate":       "discount_certificate",
    "Outperformance certificate": "outperformance_certificate",
    "Capital-guaranteed note":            "cgn_uncapped",
    "Capital-guaranteed note (capped)":   "cgn_capped",
}

# Barrier-M notes are intentionally absent: they are path-dependent (the payoff depends on
# whether the underlying touched a level *during* the option's life), but the scalable engine
# simulates only the horizon return, so they can't be priced here — use the Grid optimiser.


def _mc_der_struct(dt, k1, k2, k3):
    """Map a scalable derivative type + up to three numeric inputs (all ×-of-spot, S0=1) to the
    params dict the MC pricer expects, plus a short human description. Returns
    (params, desc, None) on success, or (None, None, warning) when a required input is missing.
    `dt` is a value from MC_DER_TYPES; the same helper feeds both the live parser and the
    resolved-settings preview so they never drift apart."""
    def _atm(x):
        return 1.0 if x is None else float(x)
    if dt in ("call", "put", "straddle"):
        k = _atm(k1)
        return {"strike": k}, f"strike {k:.2f}×", None
    if dt == "strangle":
        if k1 is None or k2 is None:
            return None, None, "needs both strikes (Strike, Strike-2)"
        return ({"strike_kp": min(k1, k2), "strike_kc": max(k1, k2)},
                f"put {min(k1, k2):.2f}× / call {max(k1, k2):.2f}×", None)
    if dt == "bull_call_spread":
        if k1 is None or k2 is None:
            return None, None, "needs both strikes (Strike, Strike-2)"
        return ({"k1": min(k1, k2), "k2": max(k1, k2)},
                f"long {min(k1, k2):.2f}× / short {max(k1, k2):.2f}× call", None)
    if dt == "bear_put_spread":
        if k1 is None or k2 is None:
            return None, None, "needs both strikes (Strike, Strike-2)"
        return ({"k1": max(k1, k2), "k2": min(k1, k2)},
                f"long {max(k1, k2):.2f}× / short {min(k1, k2):.2f}× put", None)
    if dt in ("safety_collar", "aggressive_collar"):
        if k1 is None or k2 is None:
            return None, None, "needs put & call strikes (Strike, Strike-2)"
        kp, kc = min(k1, k2), max(k1, k2)
        legs = ("long put / short call" if dt == "safety_collar" else "short put / long call")
        return ({"strike_p": kp, "strike_c": kc},
                f"{legs} — put {kp:.2f}× / call {kc:.2f}×", None)
    if dt == "butterfly_call":
        c = _atm(k1)
        if k2 is None or k2 <= 0:
            return None, None, "needs a width (Strike-2)"
        return ({"center": c, "width": float(k2)},
                f"center {c:.2f}×, width ±{float(k2):.2f}", None)
    if dt == "condor_call":
        c = _atm(k1)
        if k2 is None or k3 is None or k2 <= 0 or k3 <= 0:
            return None, None, "needs inner & outer widths (Strike-2, Strike-3)"
        wi, wo = min(k2, k3), max(k2, k3)
        return ({"center": c, "w_in": wi, "w_out": wo},
                f"center {c:.2f}×, inner ±{wi:.2f}, outer ±{wo:.2f}", None)
    if dt == "reverse_convertible":
        kp = _atm(k1)
        return {"kp": kp}, f"bond − short put @ {kp:.2f}×", None
    if dt == "discount_certificate":
        kc = _atm(k1)
        return {"kc": kc}, f"underlying capped at short call {kc:.2f}×", None
    if dt == "outperformance_certificate":
        k = _atm(k1)
        return {"k": k}, f"geared upside above {k:.2f}×", None
    if dt == "cgn_uncapped":
        g = _atm(k1)
        return ({"floor": g - 1.0, "participation": 1.0},
                f"guarantee {g*100:.0f}% of spot · 100% participation · uncapped", None)
    if dt == "cgn_capped":
        g = _atm(k1)
        if k2 is None:
            return None, None, "needs a cap level (Strike-2, e.g. 1.30 = +30%)"
        return ({"floor": g - 1.0, "participation": 1.0, "cap": float(k2) - 1.0},
                f"guarantee {g*100:.0f}% · cap at {float(k2)*100:.0f}% of spot · 100% participation", None)
    return None, None, "unsupported in the scalable engine"

# _mc_psd_cholesky -> moved to core/scenario.py

# mc_generate_scenarios -> moved to core/scenario.py

# _mc_leg_intrinsic -> moved to core/pricing.py

# _mc_leg_value_vec -> moved to core/pricing.py

# mc_der_returns -> moved to core/pricing.py

# mc_build_matrix -> moved to core/scenario.py

# mc_realised_es -> moved to core/scenario.py

# mc_analytical_es -> moved to core/scenario.py

# mc_gmv_weights -> moved to core/scenario.py

# _mc_cvar_rows -> moved to core/scenario.py

# mc_max_return_cvar -> moved to core/scenario.py

# mc_min_cvar -> moved to core/scenario.py

# mc_frontier -> moved to core/scenario.py

def plot_mc_frontier(rows):
    """E[r] vs ES floor, dark theme."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=130)
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    ok = [r for r in rows if r["ok"]]
    if ok:
        xs = [r["L"] * 100 for r in ok]; ys = [r["er"] * 100 for r in ok]
        ax.plot(xs, ys, '-o', color='#4a9eff', lw=2.2, ms=5)
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.set_xlabel('Expected-Shortfall floor L (%)', color='#c9d1d9', fontsize=9)
    ax.set_ylabel('Max expected return (%)', color='#c9d1d9', fontsize=9)
    ax.set_title('Return / tail-risk frontier (CVaR-LP)', color='#c9d1d9', fontsize=10, pad=8)
    ax.grid(True, color='#21262d', lw=0.6)
    fig.tight_layout()
    return fig


# Navigation is handled by the tile launcher above via the ?view= query param.
# The section views below render one at a time as the active view. The input
# sidebar is shown only on the Optimiser view; the hide for every other view is
# applied near the top of the file, immediately after _view is read.


if _view == "optimiser":
    import os
    with st.container():
        _bb_l, _bb_mid, _bb_x = st.columns([1, 3, 1], vertical_alignment="center")
        with _bb_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem 0 .4rem;text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em">Grid Portfolio Optimiser</h2></div>', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;align-items:center;gap:.55rem;background:transparent;border:1px solid rgba(231,236,244,0.2);border-radius:.5rem;padding:.68rem .9rem;margin:.2rem 0 1rem;color:#c0c8d8;font-size:.9rem"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg><span style="font-weight:600">Runs an exact grid optimisation of behavioural portfolios with derivatives &amp; structured products, under a VaR or Expected-Shortfall constraint you define.</span></div>', unsafe_allow_html=True)


    if not _run_active:
        st.markdown("""
<div class="info-box" style="color:#ffffff !important;background:linear-gradient(165deg,#1b2330,#161b22)">

<details>
<summary style="color:#E3C77E;font-size:1.4rem;font-weight:700;cursor:pointer"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.5rem"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>How to use the Grid Portfolio Optimiser</summary>

Follow these steps in the left sidebar:

<table style="width:100%;border-collapse:collapse;color:#ffffff;table-layout:fixed"><colgroup><col style="width:96px"><col style="width:215px"><col></colgroup>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">1</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem;white-space:nowrap;vertical-align:top"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg><strong>Portfolio data</strong></td>
  <td style="padding:.5rem .5rem .5rem .3rem;vertical-align:top">Choose a data source: default base case, live market tickers, manual entry, or CSV upload</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">2</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem;white-space:nowrap;vertical-align:top"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg><strong>Derivative &amp; parameters</strong></td>
  <td style="padding:.5rem .5rem .5rem .3rem;vertical-align:top">Select a derivative or structured product type and set its characteristics (strike, maturity, floor, participation, etc.)</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">3</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem;white-space:nowrap;vertical-align:top"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg><strong>Constraint</strong></td>
  <td style="padding:.5rem .5rem .5rem .3rem;vertical-align:top">Choose VaR or ES, set threshold H, then set α for VaR (P(r &lt; H) ≤ α) or L for ES (E[r | r &lt; H] ≥ L)</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">4</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem;white-space:nowrap;vertical-align:top"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg><strong>Grid resolution</strong></td>
  <td style="padding:.5rem .5rem .5rem .3rem;vertical-align:top">Turbo (default) gives thesis-level VaR accuracy in seconds; Fast for a quick preview; High precision for exact thesis-grade results (and for ES runs)</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">5</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem;white-space:nowrap;vertical-align:top"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16"/></svg><strong>Run</strong></td>
  <td style="padding:.5rem .5rem .5rem .3rem;vertical-align:top">Click <strong>▶ Run optimiser</strong></td>
</tr>
</table>
</details>

</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="info-box" style="color:#ffffff !important;background:linear-gradient(165deg,#1b2330,#161b22)">

<details>
<summary style="color:#E3C77E;font-size:1.4rem;font-weight:700;cursor:pointer"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.45rem"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>Output chart content</summary>

The chart shows the efficient frontiers and up to four portfolio markers:

<table style="width:100%;border-collapse:collapse;color:#ffffff;margin-top:.5rem;table-layout:fixed"><colgroup><col style="width:96px"><col style="width:215px"><col style="width:150px"><col></colgroup>
<tr><td colspan="4" style="padding:.3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Curves</td></tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:22px;border-top:2.5px dashed #a855f7;vertical-align:middle"></span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap"><strong>Purple dashed</strong></td>
  <td colspan="2" style="padding:.3rem .5rem">Mean-variance efficient frontier (Markowitz) — no derivative</td>
</tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:9px;height:9px;background:#4a9eff;border-radius:50%;vertical-align:middle"></span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap"><strong>Blue dots</strong></td>
  <td colspan="2" style="padding:.3rem .5rem">Behavioural efficient frontier — no derivative (each dot is the optimum portfolio for one H constraint level)</td>
</tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:10px;height:10px;background:#f5b942;vertical-align:middle"></span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap"><strong>Gold squares</strong></td>
  <td colspan="2" style="padding:.3rem .5rem">Behavioural optimum portfolios — derivative frontier (one point per H level, with the selected derivative)</td>
</tr>
<tr><td colspan="4" style="padding:1.6rem .5rem .3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Portfolio markers</td></tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:10px;height:10px;background:#a855f7;border:2px solid #fff;border-radius:50%;vertical-align:middle"></span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong>Purple dot (white frame)</strong></td>
  <td style="padding:.3rem .5rem;vertical-align:top"><strong style="color:#a855f7">Portfolio (0)</strong><br><span style="color:#9aa7bd;font-size:.82em">Markowitz MV optimum</span></td>
  <td style="padding:.3rem .5rem;vertical-align:top">The minimum-variance (mean-variance-efficient) portfolio at Portfolio (1)'s expected return. It lands on Portfolio (1) when Portfolio (1) is mean-variance efficient — the MVT/MAT equivalence. Shown whenever Portfolio (1) exists.</td>
</tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:9px;height:9px;background:#10b981;transform:rotate(45deg);vertical-align:middle"></span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong>Green diamond</strong></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong style="color:#10b981">Portfolio (1)</strong></td>
  <td style="padding:.3rem .5rem;vertical-align:top">Behavioural optimum without derivatives at the selected H and α constraint. Shown only when a feasible portfolio exists; when it coincides with the Markowitz MV optimum it confirms the MVT/MAT equivalence (Das, Markowitz, Scheid &amp; Statman, 2010).</td>
</tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border:2px solid #fff;vertical-align:middle"></span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong>Orange square (white frame)</strong></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong style="color:#f59e0b">Portfolio (2)</strong></td>
  <td style="padding:.3rem .5rem;vertical-align:top">Behavioural optimum portfolio with the selected derivative at the chosen H and α constraint. Highlighted separately from the frontier squares to identify the specific selected constraint point.</td>
</tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="color:#e76f51;font-size:1.15em;vertical-align:middle;text-shadow:0 0 1.5px #fff">★</span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong>Coral star (white frame)</strong></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong style="color:#e76f51">Portfolio (3)</strong></td>
  <td style="padding:.3rem .5rem;vertical-align:top">Interpolated point on the derivative frontier at the same standard deviation as Portfolio (1). Shows the return achievable with derivatives at equivalent risk — indicative only, not always available (requires derivative frontier to overlap with Portfolio (1) risk level).</td>
</tr>
<tr>
  <td style="padding:.3rem .6rem;text-align:center;white-space:nowrap;vertical-align:middle"><span style="display:inline-block;width:16px;border-top:2px dotted #fff;vertical-align:middle;margin-right:.1rem"></span><span style="color:#fff;vertical-align:middle">▸</span></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:middle"><strong>White dotted arrow</strong></td>
  <td style="padding:.3rem .5rem;white-space:nowrap;vertical-align:top"><strong style="color:#c0c8d8">Return gap</strong></td>
  <td style="padding:.3rem .5rem;vertical-align:top">Between Portfolio (1) and Portfolio (2) at the selected H and α constraint — illustrates the return uplift (or reduction) from adding derivatives.</td>
</tr>
</table>
</details>

</div>
""", unsafe_allow_html=True)
        with st.expander("**Up to four optimised portfolios — plus benchmark references**", expanded=False, icon=":material/insights:"):
            st.markdown('''
<div style="background:#ffffff;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;color:#111111;font-size:.82rem">
<b style="color:#a855f7">Portfolio (0)</b> — Markowitz mean-variance optimum (no derivative): the minimum-variance portfolio at Portfolio (1)'s expected return. It coincides with Portfolio (1) when Portfolio (1) is mean-variance efficient — directly demonstrating the MVT/MAT equivalence (shown whenever Portfolio (1) exists)<br>
<b style="color:#10b981">Portfolio (1)</b> — Behavioural optimum without derivatives at the chosen constraint (H, α): mean-variance efficient via the mental-accounting framework, and coincides with Portfolio (0) when the implied λ equals 3.795 (the MVT/MAT equivalence)<br>
<b style="color:#f59e0b">Portfolio (2)</b> — Behavioural optimum with derivative, same mental-accounting &amp; risk-aversion constraint (H, α ↔ λ): may reach higher expected returns by exploiting asymmetric derivative payoffs<br>
<b style="color:#e76f51">Portfolio (3)</b> — Portfolio with derivative and with the same variance as Portfolio (1): interpolated from the derivative frontier at equivalent risk level (indicative only)<br>
<b style="color:#556a8a">Equal-weight (1/N)</b> — naive diversification: every security weighted equally; an assumption-free reference that uses no estimates (sits inside the Markowitz frontier — mean-variance-dominated)<br>
<b style="color:#556a8a">Minimum-variance</b> — the long-only securities portfolio with the lowest variance: the left tip of the Markowitz frontier<br>
<b style="color:#556a8a">Max-Sharpe (tangency)</b> — the highest Sharpe-ratio securities portfolio: the classical mean-variance optimum, sitting on the Markowitz frontier
<div style="margin-top:.5rem;color:#555">Portfolios (0)–(3) are the optimisation's own outputs. The last three are <b>benchmark references</b> (securities only, long-only), plotted as slate diamonds on the chart for context — they are not optimisation results.</div>
</div>
''', unsafe_allow_html=True)
        st.markdown("""
<div style="border:1px solid #30363d;border-radius:12px;padding:.9rem 1.2rem;margin-top:.6rem;color:#111111;background:linear-gradient(165deg,#1b2330,#161b22)">

<details>
<summary style="color:#E3C77E;font-size:1.4rem;font-weight:700;cursor:pointer"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.45rem"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>Notes</summary>

<span style="color:#4a9eff;font-weight:700">MVT/MAT equivalence:</span><br>At the equivalence point (λ=3.795, H=-10%, α=5%), the purple and blue curves meet exactly — confirming the MVT/MAT equivalence proven in Das, Markowitz, Scheid & Statman (2010).<br>Adding derivatives shifts the frontier upward (gold squares above blue dots), revealing what the behavioural approach with derivatives can unlock beyond mean-variance.

<span style="color:#4a9eff;font-weight:700">Note on discrete vs continuous frontiers:</span><br>The behavioural frontiers are plotted at discrete constraint levels (H = -2%, -5%, -8%, -10%, -12%, -15%, -18%, -20%, -25%, -30%, -35%, -40%). Each point is the optimal portfolio for that specific mental-account threshold. The MV frontier is continuous as it is computed by sweeping the risk-aversion parameter λ — each MV portfolio corresponds to one behavioural portfolio via the MVT/MAT equivalence, demonstrating that both approaches converge to the same solution when no derivatives are present.

<span style="color:#4a9eff;font-weight:700">Why some behavioural points may appear below the MV frontier:</span><br>When derivatives are present, or when the downside constraint is particularly binding at certain H values, some behavioural frontier points may fall below the MV frontier. This is mathematically correct — the behavioural approach optimises under an additional constraint (the shortfall threshold) which can restrict the feasible set. Without derivatives, both frontiers should coincide closely.<br>With derivatives, the behavioural approach can outperform MV at higher risk levels while remaining protected at the threshold — this is the core insight of the framework. Use Standard or High precision resolution to reduce grid approximation errors.
</details>

</div>
""", unsafe_allow_html=True)
        # Sample chart
        import os
        _sample_img = "sample_output_annotated.png" if os.path.exists("sample_output_annotated.png") else "sample_output.png"
        if False:  # Sample Output section removed — the sample screenshot lives in the README
            st.markdown(
                '<div style="text-align:center;margin:1rem 0 .6rem 0">'
                '<span style="font-size:1.3rem;font-weight:700;color:#E3C77E">'
                '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.4rem"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>Sample Output</span><br>'
                '<span style="font-size:.85rem;color:#c0c8d8">'
                'Annotated example showing what each section looks like after a run — illustrative only, not live results</span>'
                '</div>',
                unsafe_allow_html=True)
            _col_l2, _col_img, _col_r2 = st.columns([1, 4, 1])
            with _col_img:
                st.image(_sample_img, use_container_width=True)

        pass  # welcome screen shown, About tab still renders

    _render_from_cache = False
    if _run_active and not _needs_compute and _has_results:
        # Restore all variables from cache so full results can re-render below
        _cache = st.session_state['_cached_results']
        nd_res = _cache['nd_res']
        dr_res = _cache['dr_res']
        p3_return = _cache['p3_return']
        p3_std = _cache['p3_std']
        constraint_label = _cache['constraint_label']
        der_config = _cache['der_config']
        der_label_sel = _cache['der_label_sel']
        H_val = _cache['H_val']
        _alpha = _cache['_alpha']
        use_es = _cache['use_es']
        _L = _cache['_L']
        lam_summary = _cache.get('lam_summary', '—')
        names_in = _cache.get('names_in', [])
        sigs_arr = _cache.get('sigs_arr', None)
        means_arr = _cache.get('means_arr', None)
        cov_mat = _cache.get('cov_mat', None)
        asset_labels = _cache.get('asset_labels', [])
        constraint_str = _cache.get('constraint_str', '')
        grid_lbl = _cache.get('grid_lbl', '')
        _ctype = _cache.get('_ctype', 'es' if use_es else 'var')
        _render_from_cache = True

    if _run_active and _needs_compute:
        # Fresh computation needed
        means_arr=np.array(means_in); sigs_arr=np.array(sigs_in)
        cov_mat=corr_to_cov(sigs_arr,corr_in)

        # Build derivative config
        der_config=None
        if der_type is not None:
            ui=der_params.get("underlying_idx",len(means_in)-1)
            sigs_for_config=sigs_arr.copy()
            sigs_for_config[ui]=der_params.get("vol",sigs_arr[ui])
            dc=build_der_config(der_type,der_params,sigs_for_config,ui)
            if dc:
                dc["r"]=der_params.get("r",0.03)
                dc["T"]=der_params.get("T",1.0)
                der_config=dc

        asset_labels=names_in+(["Derivative"] if der_config else [])

        _ctype = 'es_rigorous' if use_es_rigorous else ('es' if use_es else 'var')
        _alpha = alpha_val if not use_es else 0.05
        _L     = L_val if use_es else None

        _n_steps = 8 if der_config else 6
        import time as _time

        def _fmt_t(secs):
            if secs < 60: return f"{secs:.0f}s"
            return f"{int(secs//60)}m {int(secs%60):02d}s"

        def _step_html(steps, current, step_times=None, total_elapsed=0):
            rows = ""
            for i, entry in enumerate(steps):
                label, desc = entry[0], entry[1]
                is_sub = entry[2] if len(entry) > 2 else False
                is_timed = entry[3] if len(entry) > 3 else True
                indent = 'padding-left:1.2rem' if is_sub else ''
                t_str = (f' <span style="color:#556a8a;font-size:.7rem">'
                         f'(Execution time: {_fmt_t(step_times[i])})</span>'
                         if is_timed and step_times and i in step_times else "")
                if i < current:
                    rows += (f'<div style="display:flex;align-items:center;gap:.5rem;margin:.1rem 0;{indent}">'
                             f'<span style="color:#10b981;font-size:.75rem">✓</span>'
                             f'<span style="color:#10b981;font-size:.{("72" if is_sub else "75")}rem">{label}</span>{t_str}'
                             f'</div>')
                elif i == current:
                    rows += (f'<div style="display:flex;align-items:center;gap:.5rem;margin:.1rem 0;'
                             f'background:rgba(74,158,255,0.08);border-radius:4px;padding:.2rem .5rem;{indent}">'
                             f'<span style="color:#f59e0b;font-size:.8rem">▶</span>'
                             f'<span style="color:#4a9eff;font-size:.82rem;font-weight:700">{label}</span>'
                             f'<span style="color:#556a8a;font-size:.72rem"> — {desc}</span>'
                             f'</div>')
                else:
                    rows += (f'<div style="display:flex;align-items:center;gap:.5rem;margin:.1rem 0;{indent}">'
                             f'<span style="color:#1a3a5c;font-size:.75rem">○</span>'
                             f'<span style="color:#3a5a7a;font-size:.{("72" if is_sub else "75")}rem">{label}</span>'
                             f'</div>')
            t_str_total = (f' <span style="color:#556a8a;font-size:.7rem">'
                           f'(Execution time: {_fmt_t(total_elapsed)})</span>'
                           if total_elapsed > 0 else "")
            return ('<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;'
                    'padding:.6rem 1rem;margin-bottom:.5rem">'
                    '<div style="color:#4a9eff;font-weight:700;font-size:.78rem;margin-bottom:.4rem">'
                    f'⚙️ Computation in progress...{t_str_total}</div>' + rows + '</div>')

        def _step_html_done(steps, step_times, total_elapsed):
            rows = ""
            for i, entry in enumerate(steps):
                label = entry[0]
                is_sub = entry[2] if len(entry) > 2 else False
                is_timed = entry[3] if len(entry) > 3 else True
                indent = 'padding-left:1.2rem' if is_sub else ''
                t_str = (f' <span style="color:#556a8a;font-size:.7rem">'
                         f'(Execution time: {_fmt_t(step_times[i])})</span>'
                         if is_timed and i in step_times else "")
                rows += (f'<div style="display:flex;align-items:center;gap:.5rem;margin:.1rem 0;{indent}">'
                         f'<span style="color:#10b981;font-size:.75rem">✓</span>'
                         f'<span style="color:#10b981;font-size:.{("72" if is_sub else "75")}rem">{label}</span>{t_str}'
                         f'</div>')
            t_str_total = (f' <span style="color:#556a8a;font-size:.7rem">'
                           f'(Total execution time: {_fmt_t(total_elapsed)})</span>')
            uid = "prog_done"
            return (
                f'<details style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;'                f'padding:.4rem 1rem;margin-bottom:1.6rem">'                f'<summary style="cursor:pointer;color:#10b981;font-weight:700;font-size:.78rem;'                f'list-style:none;display:flex;align-items:center;gap:.4rem">'                f'✅ Computation complete{t_str_total}'                f'<span id="eh" style="color:#c0c8d8;font-size:.7rem;margin-left:auto"></span>'                f'</summary>'                f'<style>details[open] #eh{{content:"none"}} details[open] #eh::after{{content:"▲ click to collapse"}} #eh::after{{content:"▼ click to expand"}}</style>'                f'<div style="margin-top:.4rem">' + rows + f'</div></details>'
            )

        # Steps: (label, desc, is_substep, is_timed)
        # is_timed=True  → shows execution time when done
        # is_substep=True → indented, no time shown
        _steps_base = [
            ("Mean-variance frontier",         "Markowitz MV sweep over λ",          False, True),
            ("Behavioural frontier (no deriv.)","Building state space & optimising",  False, True),
            ("  ↳ Covariance matrix",           "Building Σ from securities data",    True,  False),
            (f"  ↳ State space",                f"Grid {m_val}³ — {m_val**3:,} states", True, False),
            ("  ↳ Gaussian copula",             "Joint distribution via copula",      True,  False),
            (f"  ↳ Grid search",                f"{mp_val} weight combinations",      True,  False),
            ("  ↳ COBYLA refinement — P(1)",    "Local optimisation",                 True,  False),
        ]
        _steps_der = [
            ("Derivative frontier",             "Building state space & optimising",  False, True),
            ("  ↳ Payoff computation",          "Black-Scholes pricing over states",  True,  False),
            (f"  ↳ State space (with deriv.)",  f"Grid {m_val}³×{mp_val} states",    True,  False),
            ("  ↳ Gaussian copula",             "Joint distribution with derivative", True,  False),
            (f"  ↳ Grid search",                f"{mp_val} weight combinations",      True,  False),
            ("  ↳ COBYLA refinement — P(2)",    "Local optimisation",                 True,  False),
            ("  ↳ Portfolio (3) interpolation", "Interpolate at P(1) std dev",        True,  False),
        ]
        _all_steps = _steps_base + (_steps_der if der_config else []) + [
            ("Chart rendering",                "Plotly efficient frontier chart",     False, True),
            ("Results computation",            "Final portfolio metrics",             False, True),
        ]

        # Three portfolio perspectives note
        with st.expander("**Up to four optimised portfolios — plus benchmark references**", expanded=False, icon=":material/insights:"):
            st.markdown('''
<div style="background:#ffffff;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;color:#111111;font-size:.82rem">
<b style="color:#a855f7">Portfolio (0)</b> — Markowitz mean-variance optimum (no derivative): the minimum-variance portfolio at Portfolio (1)'s expected return. It coincides with Portfolio (1) when Portfolio (1) is mean-variance efficient — directly demonstrating the MVT/MAT equivalence (shown whenever Portfolio (1) exists)<br>
<b style="color:#10b981">Portfolio (1)</b> — Behavioural optimum without derivatives at the chosen constraint (H, α): mean-variance efficient via the mental-accounting framework, and coincides with Portfolio (0) when the implied λ equals 3.795 (the MVT/MAT equivalence)<br>
<b style="color:#f59e0b">Portfolio (2)</b> — Behavioural optimum with derivative, same mental-accounting &amp; risk-aversion constraint (H, α ↔ λ): may reach higher expected returns by exploiting asymmetric derivative payoffs<br>
<b style="color:#e76f51">Portfolio (3)</b> — Portfolio with derivative and with the same variance as Portfolio (1): interpolated from the derivative frontier at equivalent risk level (indicative only)<br>
<b style="color:#556a8a">Equal-weight (1/N)</b> — naive diversification: every security weighted equally; an assumption-free reference that uses no estimates (sits inside the Markowitz frontier — mean-variance-dominated)<br>
<b style="color:#556a8a">Minimum-variance</b> — the long-only securities portfolio with the lowest variance: the left tip of the Markowitz frontier<br>
<b style="color:#556a8a">Max-Sharpe (tangency)</b> — the highest Sharpe-ratio securities portfolio: the classical mean-variance optimum, sitting on the Markowitz frontier
<div style="margin-top:.5rem;color:#555">Portfolios (0)–(3) are the optimisation's own outputs. The last three are <b>benchmark references</b> (securities only, long-only), plotted as slate diamonds on the chart for context — they are not optimisation results.</div>
</div>
''', unsafe_allow_html=True)

        _prog_box = st.empty()
        _step_times = {}
        _sim_start = _time.time()
        _step_start = _time.time()
        _prog_box.markdown(_step_html(_all_steps, 0, _step_times, 0), unsafe_allow_html=True)

        _prog_box.markdown(_step_html(_all_steps, 0, _step_times, 0), unsafe_allow_html=True)
        mv_x,mv_y,mv_eq=compute_mv_frontier(
            tuple(means_in),tuple(map(tuple,cov_mat.tolist())))

        _step_times[0] = _time.time()-_step_start; _step_start = _time.time()
        _prog_box.markdown(_step_html(_all_steps, 1, _step_times, _time.time()-_sim_start), unsafe_allow_html=True)
        try:
            nd_xs,nd_ys,nd_lbls=build_frontier(
                means_arr,sigs_arr,cov_mat,None,_alpha,m_val,mp_val,
                constraint_type=_ctype,L=_L)
        except Exception as e:
            st.error(f"Optimizer failed: {e}")
            nd_xs,nd_ys,nd_lbls=[],[],[]

        _step_times[1] = _time.time()-_step_start; _step_start = _time.time()
        der_xs,der_ys,der_lbls=[],[],[]
        if der_config:
            _prog_box.markdown(_step_html(_all_steps, 6, _step_times, _time.time()-_sim_start), unsafe_allow_html=True)
            try:
                der_xs,der_ys,der_lbls=build_frontier(
                    means_arr,sigs_arr,cov_mat,der_config,_alpha,m_val,mp_val,
                    constraint_type=_ctype,L=_L)
            except Exception as e:
                st.warning(f"Derivative frontier failed: {e}")

        if der_config:
            _step_times[7] = _time.time()-_step_start
        _prog_box.markdown(_step_html(_all_steps, len(_all_steps)-2, _step_times, _time.time()-_sim_start), unsafe_allow_html=True)

        _step_start = _time.time()
        # ── Pre-compute nd_res — retry up to 3 times for robustness ────────────
        _nd_res_pre = None
        for _retry in range(3):
            try:
                _r, _ = run_opt(means_arr, sigs_arr, cov_mat, None, H_val, _alpha,
                                m_val, mp_val, constraint_type=_ctype, L=_L)
                if _r is not None:
                    _nd_res_pre = _r
                    break
            except Exception:
                pass

        # Turbo's coarse grid can miss a thin feasible region near the
        # feasibility boundary and wrongly report no eligible portfolio. If
        # Turbo was selected and returned nothing, confirm with the dense
        # High-precision grid before concluding the problem is infeasible.
        if _nd_res_pre is None and mp_val == 'turbo':
            try:
                _r, _ = run_opt(means_arr, sigs_arr, cov_mat, None, H_val, _alpha,
                                51, 99, constraint_type=_ctype, L=_L)
                if _r is not None:
                    _nd_res_pre = _r
            except Exception:
                pass

        # Results computation (final P1 metrics) is now complete; capture its time
        _results_t = _time.time() - _step_start
        _step_start = _time.time()
        with st.spinner("Rendering chart..."):
            # Compute Portfolio (3) point for chart overlay using pre-computed nd_res
            _p3_x, _p3_y = None, None
            if der_xs and len(der_xs) >= 2 and _nd_res_pre:
                try:
                    _target_std = _nd_res_pre['std_dev'] * 100
                    _fp = sorted(zip(der_xs, der_ys), key=lambda p: p[0])
                    _fx = [p[0] for p in _fp]
                    _fy = [p[1] for p in _fp]
                    if min(_fx) <= _target_std <= max(_fx):
                        _p3_x = _target_std
                        _p3_y = float(np.interp(_target_std, _fx, _fy))
                except Exception:
                    pass

            # Compute implied lambda for actual Portfolio (1) point
            _lam_actual = None
            try:
                from scipy.optimize import brentq as _brentq
                _cov_s = corr_to_cov(sigs_in, corr_in)
                _lam_actual = implied_lambda(H_val, alpha_val, means_in, _cov_s)
            except Exception:
                pass

            # Portfolio (0): the Markowitz counterpart to Portfolio (1). It is
            # the minimum-variance (mean-variance-efficient) portfolio at the
            # SAME expected return as Portfolio (1). By construction it sits on
            # the MV frontier; it coincides with Portfolio (1) exactly when
            # Portfolio (1) is mean-variance efficient — which is the MVT/MAT
            # equivalence. Anchoring to Portfolio (1)'s realised return (rather
            # than an analytic implied λ) keeps the two methods consistent even
            # when the constraint is non-binding or the resolution is coarse.
            # Hidden when Portfolio (1) is infeasible (nothing to match).
            _p0 = None
            _p0_lam_str = ""
            if not use_es and _nd_res_pre is not None:
                _p0 = mv_frontier_at_return(means_arr, cov_mat,
                                            float(_nd_res_pre['expected_return']))
                if _lam_actual is not None:
                    _p0_lam_str = f"λ={_lam_actual:.3f} (implied by H, α)"
                else:
                    _p0_lam_str = "matched to Portfolio (1)'s expected return"

            # Naive benchmark points (securities only) for the frontier chart
            _grid_benchmarks = []
            try:
                from core import benchmarks as _bm
                for _blbl, _bw in _bm.benchmark_set(means_arr, cov_mat):
                    _bs = _bm.stats_from_moments(_bw, means_arr, cov_mat)
                    _grid_benchmarks.append((_blbl.split(" (")[0], _bs["vol"] * 100, _bs["er"] * 100))
            except Exception:
                _grid_benchmarks = []

            fig_plotly=plot_frontier_plotly(mv_x,mv_y,_p0,nd_xs,nd_ys,nd_lbls,
                                            der_xs,der_ys,der_lbls,der_label_sel,H_val,alpha_val,
                                            p3_x=_p3_x, p3_y=_p3_y,
                                            nd_res_actual=_nd_res_pre,
                                            lam_actual=_lam_actual, L=L_val,
                                            mv_eq_lam_str=_p0_lam_str,
                                            benchmarks=_grid_benchmarks)
            st.session_state['_fig_plotly'] = fig_plotly
            # Record times for chart + results steps
            _chart_t = _time.time() - _step_start
            _step_times[len(_all_steps)-2] = _chart_t       # Chart rendering
            _step_times[len(_all_steps)-1] = _results_t     # Results computation
            # Show final collapsed summary instead of clearing
            _final_elapsed = _time.time() - _sim_start
            _prog_box.markdown(
                _step_html_done(_all_steps, _step_times, _final_elapsed),
                unsafe_allow_html=True)
            # Store frontier data for matplotlib PDF chart (kaleido not available on Streamlit Cloud)
            st.session_state['_frontier_data'] = {
                'mv_x': mv_x, 'mv_y': mv_y, 'mv_eq': _p0,
                'nd_xs': nd_xs, 'nd_ys': nd_ys,
                'der_xs': der_xs, 'der_ys': der_ys,
                'der_label': der_label_sel,
                'H_val': H_val, 'alpha_val': alpha_val,
                'nd_res_actual': _nd_res_pre,
                'lam_actual': _lam_actual,
                'p3_x': _p3_x, 'p3_y': _p3_y,
            }
            # ── Data for the unified interactive 3D landscape (Monte-Carlo) ──
            # A joint scenario set of every instrument (securities + the derivative) lets the
            # fragment compute ANY objective for ANY instrument pair. This is a scalable-style
            # MC *illustration* of the grid problem, not the exact state-space solve — see the
            # in-plot method note. The optima are the exact grid weights (Portfolio 1 / 2).
            try:
                from core.scenario import mc_generate_scenarios as _mcg, mc_build_matrix as _mcb
                _S3 = 12000
                _sig3 = np.asarray(sigs_arr, float)
                _corr3 = np.asarray(cov_mat, float) / np.outer(_sig3, _sig3)
                _Rsec = _mcg(np.asarray(means_arr, float), _sig3, _corr3, S=_S3, seed=0)
                _labels3 = list(names_in); _Rfull = _Rsec; _w2 = None
                if der_config:
                    _ui3 = int(der_params.get("underlying_idx", der_config.get("underlying_index", 0)))
                    _dspec = [{"der_type": der_type, "params": dict(der_params), "underlying_idx": _ui3,
                               "T": float(der_params.get("T", 1.0)), "vol_override": der_params.get("vol"),
                               "r": float(der_params.get("r", 0.03)), "label": der_label_sel}]
                    try:
                        _Rf, _lb, _errs = _mcb(_Rsec, _dspec, _sig3, list(names_in),
                                               r=float(der_params.get("r", 0.03)),
                                               T=float(der_params.get("T", 1.0)))
                        if not _errs:
                            _Rfull, _labels3 = _Rf, _lb
                    except Exception:
                        pass
                    try:
                        _dr2, _ = run_opt(means_arr, sigs_arr, cov_mat, der_config, H_val, _alpha,
                                          m_val, mp_val, constraint_type=_ctype, L=_L)
                        if _dr2 is not None and _dr2.get('weights') is not None:
                            _w2 = [float(x) for x in _dr2['weights']]
                    except Exception:
                        _w2 = None
                _w1 = [float(x) for x in _nd_res_pre['weights']] if (_nd_res_pre and 'weights' in _nd_res_pre) else None
                st.session_state['_grid3d_data'] = {
                    'labels': _labels3,
                    'R': np.asarray(_Rfull, dtype='float32'),
                    'n_sec': len(names_in),
                    'w1': _w1, 'w2': (_w2 if (_w2 and len(_w2) == len(_labels3)) else None),
                    'lam': float(_lam_actual) if _lam_actual else None,
                    'H': float(H_val), 'alpha': (float(_alpha) if _alpha is not None else None),
                    'use_es': bool(use_es), 'L': (float(_L) if _L is not None else None),
                    'der_label': (der_label_sel if der_config else None),
                }
            except Exception:
                st.session_state['_grid3d_data'] = None
            st.session_state['_fig_png'] = None  # will be built at PDF time

            # ── Simulation summary + chart side by side ───────────────────────
            st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;margin:0;color:#E3C77E"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>Optimisation results</div><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin:-.15rem 0 1.7rem">▼</div>', unsafe_allow_html=True)
            col_summary, col_chart = st.columns([1, 3.5])

            with col_summary:
                _SKIP_KEYS = {"type","underlying_index","S0","r","cgn_premium","vol"}
                _LABEL_MAP = {"T":"Maturity","floor":"Floor","participation":"Participation",
                              "cap":"Cap","K":"Strike","barrier":"Barrier","M":"Barrier M",
                              "call_strike":"Call strike","put_strike":"Put strike"}
                der_params_str = ""
                if der_config:
                    for k, v in der_config.items():
                        if k in _SKIP_KEYS or v is None:
                            continue
                        label = _LABEL_MAP.get(k, k.replace("_"," ").title())
                        if isinstance(v, float):
                            der_params_str += f"<br>• {label}: {v:.2%}" if abs(v) <= 5 else f"<br>• {label}: {v:.2f}"
                        else:
                            der_params_str += f"<br>• {label}: {v}"
                lam_summary = "—"
                if not use_es:
                    _cov_s = corr_to_cov(sigs_in, corr_in)
                    _lam_s = implied_lambda(H_val, alpha_val, means_in, _cov_s)
                    if _lam_s is not None:
                        lam_summary = f"{_lam_s:.4f}"
                constraint_str = (
                    f"VaR — H={H_val:.0%}, α={_alpha:.0%}"
                    if not use_es else
                    f"ES — H={H_val:.0%}, L={_L:.0%}"
                )
                _data_src = data_mode.split("(")[0].strip()
                _securities = ", ".join(names_in)
                _underlying_name = ""
                if der_config and 'underlying_index' in der_config:
                    _ui = der_config['underlying_index']
                    if _ui < len(names_in):
                        _underlying_name = f"<br>• Underlying: {names_in[_ui]}"
                _der_html = (
                    f'<span style="color:#f59e0b">{der_label_sel}</span>{_underlying_name}{der_params_str}'
                    if der_config else
                    '<span style="color:#8896a8">None</span>'
                )
                _resolution = grid_lbl.split("(")[0].strip()
                _is_live = str(data_mode).startswith("Live")
                _period = st.session_state.get('_live_period', '—')
                _freq = st.session_state.get('_live_freq', '—')
                def _lbl(t): return f'<div style="color:#7fb3e8;font-size:.72rem;margin-bottom:.2rem">{t}</div>'
                def _val(v): return f'<div style="margin-bottom:.6rem">{v}</div>'
                _html = (
                    '<div class="g3d-sumcard" style="background:#1b2330;border:1px solid #1a3a5c;border-radius:8px;min-height:560px;height:100%;box-sizing:border-box;'
                    'padding:.8rem 1rem;color:#c0c8d8;font-size:.8rem">'
                    '<div style="color:#E3C77E;font-weight:700;font-size:.85rem;'
                    'margin-bottom:.6rem;border-bottom:1px solid #1a3a5c;padding-bottom:.4rem">'
                    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.4rem"><line x1="21" y1="4" x2="14" y2="4"/><line x1="10" y1="4" x2="3" y2="4"/><line x1="21" y1="12" x2="12" y2="12"/><line x1="8" y1="12" x2="3" y2="12"/><line x1="21" y1="20" x2="16" y2="20"/><line x1="12" y1="20" x2="3" y2="20"/><line x1="14" y1="2" x2="14" y2="6"/><line x1="8" y1="10" x2="8" y2="14"/><line x1="16" y1="18" x2="16" y2="22"/></svg>Optimisation Parameters <span style="color:#556a8a;font-size:.7rem;font-weight:700">(summary)</span></div>'
                    + _lbl("DATA SOURCE") + _val(_data_src)
                    + ((_lbl("PERIOD") + _val(_period) + _lbl("FREQUENCY") + _val(_freq)) if _is_live else "")
                    + _lbl("SECURITIES") + _val(_securities)
                    + _lbl("DERIVATIVE") + _val(_der_html)
                    + _lbl("CONSTRAINT") + _val(constraint_str)
                    + _lbl("IMPLIED λ")
                    + f'<div style="margin-bottom:.6rem;color:#10b981;font-weight:600">{lam_summary}</div>'
                    + _lbl("RESOLUTION") + _val(_resolution)
                    + '</div>'
                )
                st.markdown(_html, unsafe_allow_html=True)


            with col_chart:
                _g3d_u = st.session_state.get('_grid3d_data')
                if _g3d_u and (_g3d_u.get('R') is not None) and (_g3d_u.get('w1') or _g3d_u.get('w2')):
                    with st.expander("Objective landscape in 3D — pick instruments & objective", expanded=True, icon=":material/view_in_ar:"):
                        _grid_obj_surface_view()
                else:
                    _render_frontier_synced(fig_plotly)

        # ── "Reading the chart" note now renders next to the 2D frontier, which has
        #    moved down into the Results section (below the 3D landscape). ──────────


    # ── For cache render path: show params + chart from session state ───────
    if _render_from_cache:
        _underlying_name_c = ""
        if der_config and 'underlying_index' in der_config:
            _ui_c = der_config['underlying_index']
            if _ui_c < len(names_in):
                _underlying_name_c = f"<br>• Underlying: {names_in[_ui_c]}"
        _der_html_c = (
            f'<span style="color:#f59e0b">{der_label_sel}</span>{_underlying_name_c}'
            if der_config else
            '<span style="color:#8896a8">None</span>'
        )
        def _lbl_c(t): return f'<div style="color:#7fb3e8;font-size:.72rem;margin-bottom:.2rem">{t}</div>'
        def _val_c(v): return f'<div style="margin-bottom:.6rem">{v}</div>'
        _html_c = ('<div class="g3d-sumcard" style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;min-height:560px;height:100%;box-sizing:border-box;'
                   'padding:.8rem 1rem;color:#c0c8d8;font-size:.8rem">'
                   '<div style="color:#4a9eff;font-weight:700;font-size:.85rem;'
                   'margin-bottom:.6rem;border-bottom:1px solid #1a3a5c;padding-bottom:.4rem">'
                   '📌 Optimisation Parameters <span style="color:#556a8a;font-size:.7rem;font-weight:700">(summary)</span></div>'
                   + _lbl_c('DATA SOURCE') + _val_c(data_mode.split('(')[0].strip())
                   + ((_lbl_c('PERIOD') + _val_c(st.session_state.get('_live_period','—'))
                       + _lbl_c('FREQUENCY') + _val_c(st.session_state.get('_live_freq','—')))
                      if str(data_mode).startswith('Live') else "")
                   + _lbl_c('SECURITIES') + _val_c(', '.join(names_in))
                   + _lbl_c('DERIVATIVE') + _val_c(_der_html_c)
                   + _lbl_c('CONSTRAINT') + _val_c(constraint_str)
                   + _lbl_c('IMPLIED λ')
                   + f'<div style="margin-bottom:.6rem;color:#10b981;font-weight:600">{lam_summary}</div>'
                   + _lbl_c('RESOLUTION') + _val_c(grid_lbl.split('(')[0].strip())
                   + '</div>')
        st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;margin:0;color:#E3C77E"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>Optimisation results</div><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin:-.15rem 0 1.7rem">▼</div>', unsafe_allow_html=True)
        _col_s_c, _col_ch_c = st.columns([1, 3.5])
        with _col_s_c:
            st.markdown(_html_c, unsafe_allow_html=True)
        with _col_ch_c:
            _g3d_c = st.session_state.get('_grid3d_data')
            if _g3d_c and (_g3d_c.get('R') is not None) and (_g3d_c.get('w1') or _g3d_c.get('w2')):
                with st.expander("Objective landscape in 3D — pick instruments & objective", expanded=True, icon=":material/view_in_ar:"):
                    _grid_obj_surface_view()
            elif st.session_state.get('_fig_plotly'):
                _render_frontier_synced(st.session_state['_fig_plotly'])

    if _run_active and (_needs_compute or _render_from_cache):
        # ── Results ───────────────────────────────────────────────────────────────
        _g3d = st.session_state.get('_grid3d_data')
        _g3d_shown = bool(_g3d and (_g3d.get('R') is not None) and (_g3d.get('w1') or _g3d.get('w2')))
        # Equal-height frames: stretch the parameters card to the 3D panel's height so the two
        # frames in the results-header row stay aligned (scoped to the .g3d-sumcard card only).
        st.markdown(
            "<style>"
            "div[data-testid='stHorizontalBlock']:has(.g3d-sumcard){align-items:stretch}"
            "div[data-testid='stHorizontalBlock']:has(.g3d-sumcard) > div:first-child{height:auto}"
            "div[data-testid='stHorizontalBlock']:has(.g3d-sumcard) > div:first-child *:has(.g3d-sumcard){height:100%}"
            ".g3d-sumcard{height:100%;box-sizing:border-box}"
            "</style>", unsafe_allow_html=True)
        # ── 3D landscape method note — full width, directly below the params + 3D columns ──
        # Rendered here (outside the columns) so expanding it never breaks the alignment above.
        if _g3d_shown:
            _kap = float(st.session_state.get('_g3d_kappa', 1.0))
            with st.expander("Method & approximations — 3D objective landscape", expanded=False, icon=":material/functions:"):
                st.markdown(_G3D_METHOD_NOTE_HTML.replace('__KAP__', f'{_kap:.2f}'),
                            unsafe_allow_html=True)
        st.markdown("---")
        constraint_label = f"H={H_val:.0%}, α={_alpha:.0%}" if not use_es else f"H={H_val:.0%}, L={_L:.0%}"
        _lam_suffix = f" — implied λ = {lam_summary}" if lam_summary and lam_summary != "—" else ""
        # ── 2D return / tail-risk frontier (swapped down below the 3D landscape) ──
        # The 3D landscape now occupies the results-header slot above; the classical
        # 2D frontier renders here in full width, with its reading note alongside.
        if _g3d_shown and st.session_state.get('_fig_plotly'):
            _render_frontier_synced(st.session_state['_fig_plotly'])
            with st.expander("📐 Reading the chart", expanded=False):
                st.markdown(
                    '<div style="background:#ffffff;border:none;border-radius:8px;'
                    'padding:.8rem 1rem;margin-top:.5rem;color:#111111;font-size:.82rem">'
                    'Without derivatives, the blue behavioural frontier should closely track the purple MV frontier, '
                    'confirming the MVT/MAT equivalence (Das, Markowitz, Scheid &amp; Statman, 2010). '
                    'With derivatives, the frontiers may diverge — this is expected and is the core contribution of the framework: '
                    'derivatives allow the behavioural approach to reach portfolios that mean-variance optimisation cannot. '
                    'Small gaps below the MV frontier are grid discretisation. A blue point sitting <i>well</i> below it, however, '
                    'means the optimiser missed the true optimum for that H — in the no-derivative case the behavioural optimum is '
                    'mean-variance efficient and should lie on the purple frontier. This happens most often with '
                    '<b style="color:#d97706">Turbo</b>, whose coarse grid can be unreliable near the feasibility boundary; switch to '
                    'Standard or High precision for a clean frontier. (With derivatives, genuine divergence from the MV frontier is '
                    'expected and is the point of the framework.)</div>',
                    unsafe_allow_html=True)

        st.markdown(
            f'<h3 style="color:#E3C77E;text-align:center">'
            f'Optimal portfolios with {constraint_label}{_lam_suffix}</h3>',
            unsafe_allow_html=True)

        # ── Helper to render one portfolio column ────────────────────────────
        def _render_portfolio(border_color, header_html, caption_txt,
                              weights, labels, colors, stats,
                              delta_txt=None, method_txt=None, note_html=None, show_feasibility=False):
            """Render one portfolio: header box, then metrics left / donut centre / bars right."""
            # Header box
            with st.container(border=True):
                st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
                st.markdown(header_html, unsafe_allow_html=True)
                st.caption(caption_txt)
                # Three-column layout: metrics | donut | bars
                col_m, col_d, col_b = st.columns([1.2, 1, 1.4])
                with col_m:
                    _mr1, _mr2 = st.columns(2)
                    _mr1.metric("Expected return",
                                f"{stats['expected_return']*100:.2f}%",
                                delta=delta_txt)
                    _mr2.metric("Std deviation", f"{stats['std_dev']*100:.2f}%")
                    _mr3, _mr4 = st.columns(2)
                    _mr3.metric("Skewness", f"{stats['skewness']:.3f}")
                    _mr4.metric("Realised ES" if use_es else "Realised P(r<H)",
                                f"{stats['shortfall_stat']*100:.2f}%",
                                help=("Realized expected shortfall E[r | r<H] at this optimal "
                                      "portfolio (average return in the tail below H). The ES "
                                      "constraint requires this to stay ≥ L — the limit set in the sidebar."
                                      if use_es else
                                      "Realized P(r<H) at this optimal portfolio (probability the "
                                      "return falls below the threshold H). The VaR constraint requires "
                                      "this to stay ≤ α — the limit set in the sidebar."))
                    if show_feasibility:
                        _sf2 = round(stats['shortfall_stat']*100, 2)
                        if use_es:
                            _lim2 = round(_L*100, 2) if _L is not None else None
                            _ok2 = (_lim2 is None) or (_sf2 >= _lim2)
                            _lim_txt = f"L limit {_lim2:.2f}%" if _lim2 is not None else "L limit"
                            _btxt = (f"✓ {_sf2:.2f}% ≥ {_lim_txt}" if _ok2
                                     else f"✗ {_sf2:.2f}% &lt; {_lim_txt} (drifted past limit)")
                        else:
                            _lim2 = round(_alpha*100, 2) if _alpha is not None else None
                            _ok2 = (_lim2 is None) or (_sf2 <= _lim2)
                            _lim_txt = f"α limit {_lim2:.2f}%" if _lim2 is not None else "α limit"
                            _btxt = (f"✓ {_sf2:.2f}% ≤ {_lim_txt}" if _ok2
                                     else f"✗ {_sf2:.2f}% &gt; {_lim_txt}")
                        _mr4.markdown(f'<div style="color:{"#16a34a" if _ok2 else "#dc2626"};'
                                    f'font-size:.74rem;margin-top:-.5rem;margin-bottom:.2rem">{_btxt}</div>',
                                    unsafe_allow_html=True)
                    if method_txt:
                        st.caption(f"Method: {method_txt}")
                with col_d:
                    _svg = make_donut_svg(weights, labels, colors, size=150)
                    if _svg:
                        st.markdown(f'<div style="display:flex;justify-content:center;margin-top:1.8rem">{_svg}</div>', unsafe_allow_html=True)
                with col_b:
                    st.markdown('<div style="font-weight:600;font-size:.9rem;margin-bottom:.4rem">Portfolio weights</div>', unsafe_allow_html=True)
                    for i, w in enumerate(weights):
                        _c = colors[i % len(colors)]
                        _l = labels[i]
                        st.markdown(
                            f'<div style="margin-bottom:.45rem">'
                            f'<div><span style="color:{_c};font-weight:600">{_l}</span>'
                            f'<span style="color:{_c}"> — {w*100:.1f}%</span></div>'
                            f'<div style="height:6px;background:#1a2a3a;border-radius:3px;margin-top:3px">'
                            f'<div style="height:6px;width:{w*100:.1f}%;background:{_c};border-radius:3px"></div>'
                            f'</div></div>',
                            unsafe_allow_html=True)
                if note_html:
                    st.markdown(note_html, unsafe_allow_html=True)

        # ── Compute all four portfolios ─────────────────────────────────────
        if _render_from_cache:
            # Already restored from cache at top — nd_res/dr_res already set
            pass
        else:
            # Use _nd_res_pre if available; if None, retry once more
            nd_res = _nd_res_pre
            if nd_res is None:
                for _retry2 in range(3):
                    try:
                        _r2, _ = run_opt(means_arr, sigs_arr, cov_mat, None, H_val, _alpha,
                                         m_val, mp_val, constraint_type=_ctype, L=_L)
                        if _r2 is not None:
                            nd_res = _r2
                            break
                    except Exception:
                        pass
            dr_res = None
            p3_return = None
            p3_std = None

            if der_config:
                try:
                    dr_res, _ = run_opt(means_arr, sigs_arr, cov_mat, der_config,
                                        H_val, _alpha, m_val, mp_val,
                                        constraint_type=_ctype, L=_L)
                except Exception:
                    pass

        # Compute Portfolio (3) by interpolation (only on fresh compute)
        if not _render_from_cache and nd_res and der_xs and len(der_xs) >= 2:
            try:
                _target_std = nd_res['std_dev'] * 100
                _fp = sorted(zip(der_xs, der_ys), key=lambda p: p[0])
                _fx = [p[0] for p in _fp]
                _fy = [p[1] for p in _fp]
                if min(_fx) <= _target_std <= max(_fx):
                    p3_std = _target_std
                    p3_return = float(np.interp(_target_std, _fx, _fy))
            except Exception:
                pass

        # ── Render Portfolio (1) and (2) side by side ────────────────────────
        # ── Portfolio (0) stats (Markowitz MV optimum) — for table + detail card
        _p0_stats = None
        _p0_weights = None
        if (not use_es) and nd_res:
            try:
                _p0_out = mv_frontier_at_return(
                    means_arr, cov_mat, float(nd_res['expected_return']),
                    return_weights=True)
                if _p0_out is not None:
                    _p0_std_pct, _p0_ret_pct, _p0_w = _p0_out
                    _p0_mu = _p0_ret_pct / 100.0
                    _p0_sd = _p0_std_pct / 100.0
                    _p0_pH = float(_norm.cdf((H_val - _p0_mu) / _p0_sd)) if _p0_sd > 0 else 0.0
                    _p0_stats = {'expected_return': _p0_mu, 'std_dev': _p0_sd,
                                 'skewness': 0.0, 'shortfall_stat': _p0_pH}
                    _p0_weights = _p0_w
            except Exception:
                _p0_stats = None
                _p0_weights = None

        # ── Summary overview table (quick comparison of resulting portfolios) ──
        _p1_ret_ref = nd_res['expected_return'] * 100 if nd_res else None
        def _delta_pp(ret_pct):
            if _p1_ret_ref is None or ret_pct is None:
                return "—"
            _d = ret_pct - _p1_ret_ref
            return f'{("+" if _d >= 0 else "")}{_d:.2f} pp'
        _sum_rows = []
        if _p0_stats is not None:
            _sum_rows.append(("#a855f7", "Portfolio (0) — Markowitz MV optimum (no derivative)",
                              _p0_stats['expected_return'] * 100, _p0_stats['std_dev'] * 100,
                              "0.000", _delta_pp(_p0_stats['expected_return'] * 100)))
        if nd_res:
            _sum_rows.append(("#10b981", "Portfolio (1) — Behavioural optimum, no derivative",
                              nd_res['expected_return'] * 100, nd_res['std_dev'] * 100,
                              f"{nd_res['skewness']:.3f}", "—"))
        if der_config and dr_res:
            _sum_rows.append(("#f59e0b", f"Portfolio (2) — Behavioural optimum, with {der_label_sel}",
                              dr_res['expected_return'] * 100, dr_res['std_dev'] * 100,
                              f"{dr_res['skewness']:.3f}", _delta_pp(dr_res['expected_return'] * 100)))
        if der_config and (p3_return is not None) and (p3_std is not None):
            _sum_rows.append(("#e76f51", f"Portfolio (3) — Same variance as Portfolio (1), with {der_label_sel}",
                              p3_return, p3_std, "—", _delta_pp(p3_return)))
        # Naive / classical benchmarks (long-only, same securities) — for context
        _has_benchmarks = False
        try:
            from core import benchmarks as _bm
            for _blbl, _bw in _bm.benchmark_set(means_arr, cov_mat, rf=getattr(_bm, "RF_ANNUAL", 0.03)):
                _bs = _bm.stats_from_moments(_bw, means_arr, cov_mat)
                _sum_rows.append(("#7d8aa0", f"Benchmark — {_blbl}",
                                  _bs["er"] * 100, _bs["vol"] * 100, "0.000",
                                  _delta_pp(_bs["er"] * 100)))
                _has_benchmarks = True
        except Exception:
            pass
        if _sum_rows:
            _trs = ""
            for _clr, _name, _ret, _std, _skew, _dlt in _sum_rows:
                _trs += (
                    f'<tr style="border-top:1px solid #1a2a3a">'
                    f'<td style="padding:.4rem .7rem"><span style="color:{_clr};font-weight:700">&#9679;</span> '
                    f'<span style="color:#dbe7ff">{_name}</span></td>'
                    f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_ret:.2f}%</td>'
                    f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_std:.2f}%</td>'
                    f'<td style="padding:.4rem .7rem;text-align:center;color:#9fb3d1">{_skew}</td>'
                    f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_dlt}</td>'
                    f'</tr>')
            _summary_html = (
                '<div style="background:#0d1a2e;border:none;border-radius:8px;'
                'padding:.6rem .8rem;margin:.2rem 0 1rem 0;overflow-x:auto">'
                '<div style="color:#4a9eff;font-weight:700;font-size:.95rem;margin-bottom:.4rem;text-align:center">'
                'Summary — resulting portfolios</div>'
                '<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
                '<thead><tr style="color:#9fb3d1">'
                '<th style="padding:.3rem .7rem;text-align:center">Portfolio</th>'
                '<th style="padding:.3rem .7rem;text-align:center">Expected return</th>'
                '<th style="padding:.3rem .7rem;text-align:center">Std deviation</th>'
                '<th style="padding:.3rem .7rem;text-align:center">Skewness</th>'
                '<th style="padding:.3rem .7rem;text-align:center">&Delta; vs (1)</th>'
                '</tr></thead><tbody>' + _trs + '</tbody></table>'
                '<div style="color:#6b7f99;font-size:.7rem;margin-top:.4rem">'
                '&Delta; vs (1) = expected-return gap relative to Portfolio (1). '
                'Portfolio (0) is a Gaussian mean-variance construct (skewness 0). '
                'Portfolio (3) is interpolated from the derivative frontier (indicative only).'
                + (' Benchmarks are long-only, fully-invested portfolios of the same securities '
                   '(equal-weight, minimum-variance, max-Sharpe) shown for context, not ranking.'
                   if _has_benchmarks else '')
                + '</div>'
                '</div>')
            st.markdown(_summary_html, unsafe_allow_html=True)

        # ── Portfolio (0) — Markowitz MV optimum (detailed view) ─────────────
        if _p0_stats is not None and _p0_weights is not None:
            with st.container():
                _p0_labels = [names_in[i] if i < len(names_in) else f"Asset {i+1}"
                              for i in range(len(_p0_weights))]
                _p0_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_p0_weights))]
                _render_portfolio(
                    border_color="#a855f7",
                    show_feasibility=False,
                    header_html=(
                        '<div style="background:#0d1a2e;border:1px solid #a855f7;border-radius:8px;'
                        'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                        '<span style="color:#a855f7;font-weight:700;font-size:.95rem">'
                        '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#a855f7;border:2px solid white;margin-right:.4rem;vertical-align:middle"></span>'
                        'Optimal portfolio (0) — Markowitz MV optimum (no derivative)</span></div>'
                    ),
                    caption_txt=("Minimum-variance portfolio at Portfolio (1)'s expected return — "
                                 "coincides with Portfolio (1) when it is mean-variance efficient "
                                 "(the MVT/MAT equivalence). Gaussian construct: skewness 0, tail "
                                 "probability (chance of finishing below H) from the normal model."),
                    weights=_p0_weights, labels=_p0_labels, colors=_p0_colors,
                    stats=_p0_stats, method_txt="Markowitz mean-variance (SLSQP)")

        with st.container():
            if nd_res:
                _nd_weights = nd_res["weights"]
                _nd_labels = [names_in[i] if i < len(names_in) else f"Asset {i+1}" for i in range(len(_nd_weights))]
                _nd_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_nd_weights))]
                _method = ("Exhaustive grid search + COBYLA" if nd_res.get('method_used') == "grid_search"
                           else "Differential evolution + COBYLA" if nd_res.get('method_used') == "differential_evolution"
                           else "Coarse-to-fine grid + COBYLA (Turbo)" if nd_res.get('method_used') == "coarse_to_fine"
                           else "Coarse-to-fine + ES-aware COBYLA (Rigorous ES)" if nd_res.get('method_used') == "es_rigorous"
                           else nd_res.get('method_used', '—'))
                _render_portfolio(
                    border_color="#10b981",
                    show_feasibility=True,
                    header_html=(
                        '<div style="background:#0d1a2e;border:1px solid #10b981;border-radius:8px;'
                        'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                        '<span style="color:#10b981;font-weight:700;font-size:.95rem">'
                        '<span style="color:#10b981;margin-right:.4rem">◆</span>Behavioural optimal portfolio (1) — no derivative</span></div>'
                    ),
                    caption_txt="Maximises return subject to the downside constraint — reference portfolio (equivalent to Markowitz MV optimum)",
                    weights=_nd_weights, labels=_nd_labels, colors=_nd_colors,
                    stats=nd_res, method_txt=_method)
            else:
                st.markdown("**Behavioural optimal portfolio (1) — no derivative**")
                # Suggest a wider constraint based on current securities volatility.
                # sigs_arr is set on the fresh-compute path and restored from cache
                # on rerun; guard against an older cache that lacks it.
                try:
                    _avg_sig = float(np.mean(sigs_arr)) * 100
                    _suggested_H_val = min(40, max(15, int(_avg_sig * 1.5)))
                    st.warning(
                        f"⚠️ No eligible portfolio found at H={H_val:.0%}, α={_alpha:.0%}. "
                        f"With live market data (avg volatility {_avg_sig:.1f}%), "
                        f"try a wider threshold such as H=-{_suggested_H_val}% or switch to Standard resolution.")
                except Exception:
                    st.warning(
                        f"⚠️ No eligible portfolio found at H={H_val:.0%}, α={_alpha:.0%}. "
                        f"Try a wider threshold (e.g. H=-40%) or switch to Standard resolution.")

        with st.container():
            if der_config:
                if dr_res:
                    _dr_weights = dr_res["weights"]
                    _dr_labels = [asset_labels[i] if i < len(asset_labels) else f"Asset {i+1}" for i in range(len(_dr_weights))]
                    _dr_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_dr_weights))]
                    _delta = f"+{(dr_res['expected_return']-(nd_res['expected_return'] if nd_res else 0))*100:.2f}pp"
                    _method = ("Exhaustive grid search + COBYLA" if dr_res.get('method_used') == "grid_search"
                               else "Differential evolution + COBYLA" if dr_res.get('method_used') == "differential_evolution"
                               else "Coarse-to-fine grid + COBYLA (Turbo)" if dr_res.get('method_used') == "coarse_to_fine"
                               else "Coarse-to-fine + ES-aware COBYLA (Rigorous ES)" if dr_res.get('method_used') == "es_rigorous"
                               else dr_res.get('method_used', '—'))
                    _dr_ret = dr_res['expected_return']*100
                    _nd_ret = nd_res['expected_return']*100 if nd_res else 0
                    _p2_sign = "+" if _dr_ret >= _nd_ret else ""
                    _p2_diff = _dr_ret - _nd_ret
                    _p2_note = (
                        f'<div style="background:#ffffff;border:1px solid #f59e0b;border-radius:6px;'
                        f'padding:.6rem 1rem;color:#111111;font-size:.82rem;margin-top:.6rem">'
                        f'At the same mental-accounting constraint (H={H_val:.0%}, α={_alpha:.0%} ↔ λ), '
                        f'the optimum portfolio with <b style="color:#f59e0b">{der_label_sel}</b> '
                        f'achieves <b style="color:#f59e0b">{_dr_ret:.2f}%</b> expected return '
                        f'vs <b>{_nd_ret:.2f}%</b> for portfolio (1) without derivatives — '
                        f'a <b style="color:{"#10b981" if _p2_diff>=0 else "#ef4444"}">{_p2_sign}{_p2_diff:.2f} pp '
                        f'{"gain" if _p2_diff>=0 else "reduction"}</b> '
                        f'(note: higher return may come with higher variance).</div>'
                    )
                    _render_portfolio(
                        border_color="#f59e0b",
                        show_feasibility=True,
                        header_html=(
                            f'<div style="background:#0d1a2e;border:1px solid #f59e0b;border-radius:8px;'
                            f'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                            f'<span style="color:#f59e0b;font-weight:700;font-size:.95rem">'
                            f'<span style="display:inline-block;width:12px;height:12px;background:#ff6b00;border:2px solid white;margin-right:.4rem;vertical-align:middle"></span>Behavioural optimal portfolio (2) — with {der_label_sel}</span></div>'
                        ),
                        caption_txt=f"Same mental-accounting & risk-aversion constraint (H={H_val:.0%}, α={_alpha:.0%} ↔ λ) — results may vary",
                        weights=_dr_weights, labels=_dr_labels, colors=_dr_colors,
                        stats=dr_res, delta_txt=f"{_p2_sign}{_p2_diff:.2f}pp vs portfolio (1)",
                        method_txt=_method, note_html=_p2_note)
                else:
                    st.markdown(f"**Behavioural optimal portfolio (2) — with {der_label_sel}**")
                    st.warning("⚠️ No eligible portfolio found with this derivative. Try different parameters.")
            else:
                st.info("Select a derivative in the sidebar to see Portfolio (2).")

        # ── Portfolio (3) — full width below ─────────────────────────────────
        if der_config and nd_res and p3_return is not None:
            st.markdown("---")
            _gain3 = p3_return - nd_res['expected_return'] * 100
            # Use derivative weights as approximation (closest frontier point)
            if dr_res:
                _p3_weights = dr_res["weights"]
                _p3_labels = [asset_labels[i] if i < len(asset_labels) else f"Asset {i+1}" for i in range(len(_p3_weights))]
                _p3_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_p3_weights))]
                _p3_donut = make_donut_svg(_p3_weights, _p3_labels, _p3_colors, size=150)
            _gain3_sign = "+" if _gain3 >= 0 else ""
            _gain3_word = "gain" if _gain3 >= 0 else "reduction"
            _gain3_color = "#10b981" if _gain3 >= 0 else "#ef4444"
            _p3_interp_note = (
                f'<div style="background:#ffffff;border:1px solid #e76f51;border-radius:6px;'
                f'padding:.6rem 1rem;color:#111111;font-size:.82rem;margin-top:.4rem">'
                f'At the <b style="color:#e76f51">same variance as portfolio (1)</b> ({p3_std:.1f}% std dev), '
                f'the derivative frontier achieves <b style="color:#e76f51">{p3_return:.2f}%</b> expected return '
                f'vs <b>{nd_res["expected_return"]*100:.2f}%</b> without derivatives — '
                f'a <b style="color:{_gain3_color}">{_gain3_sign}{_gain3:.2f} pp {_gain3_word}</b> '
                f'(indicative — interpolated from derivative frontier, not directly optimised).</div>'
            )
            _gain3_sign = "+" if _gain3 >= 0 else ""
            if dr_res:
                _p3_stats = {
                    'expected_return': p3_return / 100,
                    'std_dev': p3_std / 100,
                    'skewness': dr_res.get('skewness', 0),
                    'shortfall_stat': dr_res.get('shortfall_stat', 0)
                }
                _render_portfolio(
                    border_color="#e76f51",
                    header_html=(
                        f'<div style="background:#0d1a2e;border:1px solid #e76f51;border-radius:8px;'
                        f'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                        f'<span style="color:#e76f51;font-weight:700;font-size:.95rem">'
                        f'<svg width="14" height="14" viewBox="0 0 24 24" style="margin-right:.4rem;vertical-align:middle" xmlns="http://www.w3.org/2000/svg"><polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" fill="#e76f51" stroke="white" stroke-width="1.5"/></svg>Optimal portfolio (3) — same variance as Portfolio (1), with {der_label_sel}'
                        f'</span> <span style="color:#c0c8d8;font-size:.78rem">(interpolated)</span>'
                        f'</div>'
                    ),
                    caption_txt=f"Interpolated from the derivative frontier at the same std deviation as portfolio (1) — indicative only",
                    weights=_p3_weights, labels=_p3_labels, colors=_p3_colors,
                    stats=_p3_stats,
                    delta_txt=f"{_gain3_sign}{_gain3:.2f}pp vs portfolio (1)",
                    method_txt="Interpolated from derivative frontier — weights shown are from the closest optimised frontier point",
                    note_html=_p3_interp_note)
        elif der_config and nd_res and len(der_xs) >= 2:
            st.markdown("---")
            st.markdown(
                f'<div style="background:#0d1a2e;border:1px solid #e76f51;border-radius:8px;'
                f'padding:.8rem 1rem;color:#c0c8d8;font-size:.85rem">'
                f'<b style="color:#e76f51">Portfolio (3) — not available for this derivative</b><br><br>'
                f'Portfolio (3) requires the no-derivative portfolio std dev (<b>{nd_res["std_dev"]*100:.1f}%</b>) '
                f'to fall within the derivative frontier range '
                f'(<b>{min(der_xs):.1f}%–{max(der_xs):.1f}%</b>). '
                f'With a <b>{der_label_sel}</b>, the derivative portfolio always carries higher variance than the '
                f'no-derivative portfolio — so no same-variance comparison is possible at this constraint level.<br><br>'
                f'<b>To see Portfolio (3):</b> try a <b>put option</b> or <b>collar</b>, which have lower variance '
                f'impact and whose frontier may overlap with the no-derivative portfolio range.</div>',
                unsafe_allow_html=True)
        elif der_config and nd_res and len(der_xs) < 2:
            st.markdown("---")
            st.info(f"Portfolio (3) not available — derivative frontier has {len(der_xs)} point(s). Try Standard resolution.")

        # ── PDF Export — generate and store in session_state ────────────────────
        st.markdown("---")
        if nd_res:
            try:
                _lam_s = lam_summary if 'lam_summary' in dir() else "—"
                _nd_lbls_pdf = [names_in[i] if i<len(names_in) else f"Asset {i+1}" for i in range(len(nd_res["weights"]))]
                _nd_wts_pdf  = list(nd_res["weights"])
                _nd_cols_pdf = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_nd_wts_pdf))]
                _dr_lbls_pdf = [asset_labels[i] if i<len(asset_labels) else f"Asset {i+1}" for i in range(len(dr_res["weights"]))] if dr_res else []
                _dr_wts_pdf  = list(dr_res["weights"]) if dr_res else []
                _dr_cols_pdf = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_dr_wts_pdf))] if dr_res else []
                _p3r = p3_return if p3_return is not None else None
                _p3s = p3_std if p3_std is not None else None
                # Build chart PNG using matplotlib (kaleido not available on Streamlit Cloud)
                _fig_png_for_pdf = None
                _fd = st.session_state.get('_frontier_data', {})
                if _fd:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import matplotlib.patches as mpatches
                        import io as _io
                        _fig_m, _ax = plt.subplots(figsize=(10, 5.5))
                        _fig_m.patch.set_facecolor('#0d1117')
                        _ax.set_facecolor('#0d1117')
                        _ax.tick_params(colors='#c0c8d8', labelsize=8)
                        _ax.spines[:].set_color('#1a3a5c')
                        _ax.xaxis.label.set_color('#c0c8d8')
                        _ax.yaxis.label.set_color('#c0c8d8')
                        _ax.set_xlabel('Portfolio Risk — Standard Deviation (%)', fontsize=9, color='#c0c8d8')
                        _ax.set_ylabel('Expected Return (%)', fontsize=9, color='#c0c8d8')
                        _ax.set_title('Mean-Variance vs Behavioural Portfolio Efficient Frontier',
                                      fontsize=10, color='#ffffff', pad=8)
                        _ax.grid(True, color='#1a3a5c', linewidth=0.5, alpha=0.5)
                        if _fd.get('mv_x'):
                            _ax.plot(_fd['mv_x'], _fd['mv_y'], '--', color='#a855f7',
                                     linewidth=1.2, label='MV frontier (Markowitz)', alpha=0.8)
                        if _fd.get('nd_xs'):
                            _ax.plot(_fd['nd_xs'], _fd['nd_ys'], 'o-', color='#4a9eff',
                                     linewidth=1.2, markersize=5, label='Behavioural — no derivative', alpha=0.85)
                        if _fd.get('der_xs'):
                            _ax.scatter(_fd['der_xs'], _fd['der_ys'], marker='s', color='#f59e0b',
                                        s=40, label=f"Behavioural — {_fd.get('der_label','derivative')}", alpha=0.8, zorder=5)
                        _nr = _fd.get('nd_res_actual')
                        if _nr:
                            _ax.scatter([_nr['std_dev']*100], [_nr['expected_return']*100],
                                        marker='D', color='#10b981', s=100, zorder=10,
                                        label=f"Portfolio (1) H={_fd['H_val']:.0%} α={_fd['alpha_val']:.0%}")
                        if _fd.get('p3_x') and _fd.get('p3_y'):
                            _ax.scatter([_fd['p3_x']], [_fd['p3_y']], marker='*',
                                        color='#e76f51', s=200, zorder=10,
                                        edgecolors='white', linewidths=0.8,
                                        label='Portfolio (3) — interpolated')
                        # ── Portfolio (0) — Markowitz MV optimum (purple circle) ──
                        _p0pt = _fd.get('mv_eq')
                        if _p0pt:
                            _ax.scatter([_p0pt[0]], [_p0pt[1]], marker='o', color='#a855f7',
                                        s=90, zorder=11, edgecolors='white', linewidths=0.8,
                                        label='Portfolio (0) — Markowitz MV optimum')
                        # ── Portfolio (2) — behavioural optimum with derivative (orange square) ──
                        _p2pt = (dr_res['std_dev']*100, dr_res['expected_return']*100) if dr_res else None
                        if _p2pt:
                            _ax.scatter([_p2pt[0]], [_p2pt[1]], marker='s', color='#f59e0b',
                                        s=130, zorder=11, edgecolors='white', linewidths=0.8,
                                        label='Portfolio (2) — optimum with derivative')
                        # ── Labelled call-outs (squares linked to each portfolio with text) ──
                        def _callout(pt, txt, color, dx, dy):
                            if not pt:
                                return
                            _ax.annotate(txt, xy=(pt[0], pt[1]), xytext=(pt[0]+dx, pt[1]+dy),
                                         fontsize=6.3, color='white', ha='left', va='center', zorder=20,
                                         bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='white', lw=0.5, alpha=0.95),
                                         arrowprops=dict(arrowstyle='->', color=color, lw=1.0))
                        _p1pt = (_nr['std_dev']*100, _nr['expected_return']*100) if _nr else None
                        _p3pt = (_fd.get('p3_x'), _fd.get('p3_y')) if (_fd.get('p3_x') and _fd.get('p3_y')) else None
                        if _p0pt:
                            _callout(_p0pt, f"Portfolio (0)\nMarkowitz MV\n{_p0pt[1]:.1f}%  |  sd {_p0pt[0]:.1f}%", '#a855f7', -9, 3.2)
                        if _p1pt:
                            _callout(_p1pt, f"Portfolio (1)\nbehavioural, no deriv\n{_p1pt[1]:.1f}%  |  sd {_p1pt[0]:.1f}%", '#10b981', -10, -4.2)
                        if _p2pt:
                            _callout(_p2pt, f"Portfolio (2)\nwith {_fd.get('der_label','derivative')}\n{_p2pt[1]:.1f}%  |  sd {_p2pt[0]:.1f}%", '#f59e0b', 2.5, 3.2)
                        if _p3pt:
                            _callout(_p3pt, f"Portfolio (3)\nsame sd as (1)\n{_p3pt[1]:.1f}%  |  sd {_p3pt[0]:.1f}%", '#e76f51', 2.5, -4.2)
                        _ax.legend(fontsize=7, facecolor='#0d1a2e', edgecolor='#1a3a5c',
                                   labelcolor='#c0c8d8', loc='upper left')
                        plt.tight_layout(pad=1.5)
                        _buf_m = _io.BytesIO()
                        _fig_m.savefig(_buf_m, format='png', dpi=150,
                                       bbox_inches='tight', facecolor='#0d1117')
                        plt.close(_fig_m)
                        _fig_png_for_pdf = _buf_m.getvalue()
                    except Exception as _mpl_err:
                        _fig_png_for_pdf = None
                _pdf_bytes = generate_pdf_report(
                    constraint_label=constraint_label,
                    nd_res=nd_res, dr_res=dr_res,
                    p3_return=_p3r, p3_std=_p3s,
                    nd_labels=_nd_lbls_pdf, nd_weights=_nd_wts_pdf, nd_colors=_nd_cols_pdf,
                    dr_labels=_dr_lbls_pdf, dr_weights=_dr_wts_pdf, dr_colors=_dr_cols_pdf,
                    der_label_sel=der_label_sel,
                    H_val=H_val, _alpha=_alpha, use_es=use_es, _L=_L,
                    data_mode=data_mode, names_in=names_in,
                    grid_lbl=grid_lbl, lam_summary=_lam_s,
                    p0_stats=_p0_stats,
                    p0_labels=([names_in[i] if i < len(names_in) else f"Asset {i+1}" for i in range(len(_p0_weights))] if _p0_weights is not None else None),
                    p0_weights=(list(_p0_weights) if _p0_weights is not None else None),
                    p0_colors=([DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_p0_weights))] if _p0_weights is not None else None),
                    fig_png=_fig_png_for_pdf
                )
                # Store PDF bytes in session_state so download button doesn't trigger rerun loss
                st.session_state['_pdf_bytes'] = _pdf_bytes
                st.session_state['_pdf_filename'] = f"portfolio_optimisation_{H_val:.0%}_{_alpha:.0%}.pdf"
            except Exception as _pdf_err:
                st.caption(f"PDF generation failed: {_pdf_err}")

        # Render download button from session_state — st.download_button does NOT trigger rerun
        if st.session_state.get('_pdf_bytes'):
            _col_l, _col_c, _col_r = st.columns([1, 2, 1])
            with _col_c:
                st.download_button(
                    label=":material/download: Export & Download PDF Report",
                    data=st.session_state['_pdf_bytes'],
                    file_name=st.session_state.get('_pdf_filename', 'report.pdf'),
                    mime="application/pdf",
                    type="primary",
                    key="pdf_download",
                    use_container_width=True
                )

        # ── Cache results in session_state ───────────────────────────────────
        st.session_state['_cached_results'] = {
            'nd_res': nd_res, 'dr_res': dr_res,
            'p3_return': p3_return, 'p3_std': p3_std,
            'constraint_label': constraint_label,
            'der_config': der_config, 'der_label_sel': der_label_sel,
            'H_val': H_val, '_alpha': _alpha, 'use_es': use_es, '_ctype': _ctype, '_L': _L,
            'data_mode': data_mode,
            'lam_summary': lam_summary,
            'names_in': names_in,
            'sigs_arr': sigs_arr,
            'means_arr': means_arr, 'cov_mat': cov_mat,
            'asset_labels': asset_labels,
            'constraint_str': constraint_str,
            'grid_lbl': grid_lbl,
            'underlying_idx': der_params.get('underlying_idx', 0) if der_config else 0,
            'der_params_fp': str(sorted(der_params.items())) if der_type is not None else '',
        }
        st.session_state['_needs_compute'] = False

        # ── How to read these results ────────────────────────────────────────
        if der_config and nd_res:
            st.markdown("---")
            st.markdown(
                '<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
                'padding:.8rem 1rem;margin-top:.5rem;color:#111111;font-size:.85rem">'
                '<b style="color:#1a3a6b">📌 How to read these results</b><br>'
                'Portfolio (1) and (2) are compared at the <b>same mental-accounting & risk-aversion constraint</b> '
            f'(H={H_val:.0%}, α={_alpha:.0%} — same risk-aversion λ). '
            'Depending on the derivative chosen, portfolio (2) may achieve a higher or lower expected return '
            'and may show higher variance. '
            'Portfolio (3) shows the derivative frontier return at the <b>same variance as Portfolio (1)</b>, '
            'providing a complementary risk-adjusted perspective.</div>',
                unsafe_allow_html=True)




    # Always visible — portfolio data and contact
    st.markdown("---")
    show_portfolio_data(names_in, means_in, sigs_in, corr_in)

    st.markdown("---")



    st.markdown(
        '<div style="text-align:center;color:#556a8a;font-size:.75rem;margin-top:2rem;padding:.6rem">'
        '⚠️ For educational &amp; research purposes only — not financial advice. '
        'Based on Das &amp; Statman (2009), Das, Markowitz, Scheid &amp; Statman (2010) and Jeddou (2012). '
        'See <b>About</b> tab for full disclaimer.<br>© 2026 Sami Jeddou · All rights reserved.</div>',
        unsafe_allow_html=True)

elif _view == "scalable":
    import datetime as _dt
    with st.container():
        _bb_l, _bb_mid, _bb_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _bb_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _bb_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em">Scalable Optimiser — Monte-Carlo + CVaR</h2></div>', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;align-items:flex-start;gap:.6rem;background:transparent;border:1px solid rgba(231,236,244,0.2);border-radius:.5rem;padding:.7rem .95rem;margin:.2rem 0 1rem;color:#c0c8d8;font-size:.9rem;line-height:1.55"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;margin-top:3px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg><span>A <strong>scenario-based</strong> engine for <strong>large portfolios</strong> and <strong>several derivatives at once</strong> — the case the exact grid optimiser cannot reach. It samples joint return scenarios and solves <em>maximise expected return subject to an α-CVaR (Expected-Shortfall at level α) floor</em> as a linear program. Cost grows <strong>linearly</strong> in the number of assets, so it scales to many securities; and any number of derivatives just add columns.</span></div>', unsafe_allow_html=True)
    st.warning("**Beta — approximate engine.** Results carry Monte-Carlo sampling error and "
               "depend on scenario quality. It *complements* the exact grid Optimiser (which "
               "remains the thesis-faithful reference for small portfolios), it does not "
               "replace it.")

    st.markdown("""
<div class="info-box" style="color:#ffffff !important;background:linear-gradient(165deg,#1b2330,#161b22)">

<details>
<summary style="color:#E3C77E;font-size:1.4rem;font-weight:700;cursor:pointer"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.5rem"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>How to use this tool: Scalable Portfolio Optimiser</summary>
<div style="margin:.6rem 0">Set up the run in the sections below, then click Run:</div>
<table style="width:100%;border-collapse:collapse;color:#ffffff">
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">1</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Data &amp; estimation</strong> — Choose a data source: live market tickers (estimated over a window) or the 3-asset thesis sample case (Das–Statman)</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">2</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Scenarios</strong> — Set the number of Monte-Carlo scenarios S and the copula (Gaussian, or Student-t for tail dependence)</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">3</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Derivatives</strong> (optional) — Add one or more derivatives (call, put, straddle, strangle, vertical spreads), each on any security</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">4</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Constraint</strong> — Set the tail probability α, the α-CVaR floor L (mean of the worst α% of outcomes ≥ L), and optional min/max weight bounds per security</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">5</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Validation</strong> (optional) — Run validation checks against closed-form values (Gaussian copula)</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">6</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Run</strong> — Click <strong>▶ Run scalable optimiser</strong></td>
</tr>
</table>
</details>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class="info-box" style="color:#ffffff !important;background:linear-gradient(165deg,#1b2330,#161b22)">

<details>
<summary style="color:#E3C77E;font-size:1.4rem;font-weight:700;cursor:pointer"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.45rem"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>Output chart content</summary>
<div style="margin:.6rem 0">After a run, the results show a details box, colour-coded weight bars, and an interactive return / tail-risk frontier:</div>
<table style="width:100%;border-collapse:collapse;color:#ffffff;margin-top:.5rem">
<tr><td colspan="2" style="padding:.3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Frontier chart</td></tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">🔵 <strong>Blue line + dots</strong></td>
  <td style="padding:.3rem .5rem">The return / tail-risk frontier — each dot is the <strong>maximum expected return</strong> achievable for one Expected-Shortfall floor L</td>
</tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">⭐ <strong>Gold star</strong></td>
  <td style="padding:.3rem .5rem"><strong>Scalable CVaR optimum</strong> (your resulting portfolio) at the chosen floor L — hover for its expected return, L and realised ES</td>
</tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">🖱️ <strong>Interactivity</strong></td>
  <td style="padding:.3rem .5rem">Hover any point for its coordinates; drag to zoom, double-click to reset</td>
</tr>
<tr><td colspan="2" style="padding:.5rem .5rem .3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Weights &amp; details</td></tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">📊 <strong>Weight bars</strong></td>
  <td style="padding:.3rem .5rem">Each security has its own colour; <span style="color:#f59e0b;font-weight:700">amber</span> bars are derivatives</td>
</tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">🔷 <strong>Details box</strong></td>
  <td style="padding:.3rem .5rem">Expected return, volatility, skewness and realised ES of the optimal portfolio, with a feasibility badge against the floor L</td>
</tr>
</table>
</details>
</div>
""", unsafe_allow_html=True)

    with st.expander("How this engine works", expanded=False, icon=":material/settings:"):
        st.markdown(
            "1. **Scenarios.** Draw S joint return scenarios for all securities at once via a "
            "copula — Gaussian (the thesis's multivariate-Normal assumption) or Student-t "
            "(tail dependence: assets crash together). The data is an S×N matrix, so memory "
            "and cost are O(S·N) — *linear* in the number of securities, not exponential.\n"
            "2. **Derivatives.** Each derivative is priced from its Black-Scholes legs inside "
            "*every* scenario (same definition as the engine and the Backtest tab), adding one "
            "return column. Several derivatives — even on different underlyings — cost nothing "
            "extra structurally.\n"
            "3. **Optimisation.** With weights w, the portfolio return in scenario s is w·Rₛ. "
            "Using the Rockafellar–Uryasev identity, *maximise E[r] subject to "
            "ES_α(r) ≥ L* becomes a **linear program** in (w, ζ, z) that solves in well under "
            "a second even for hundreds of weights and tens of thousands of scenarios — and "
            "it is convex, so no heuristics.\n"
            "4. **Frontier.** Sweeping the floor L traces the return / tail-risk frontier.\n\n"
            "Full derivation is in the accompanying addendum to the thesis."
        )
    with st.expander("Assumptions & limitations", icon=":material/warning:"):
        st.markdown(
            "- **Approximate.** Scenario estimates carry error of order S^(−1/2); tails need "
            "more scenarios than the body. Raise S for stability.\n"
            "- **One-year horizon.** Losses are simulated over a single one-year period "
            "(μ and σ are annualised), so the CVaR floor is a one-year tail limit. Each "
            "derivative's *maturity* is set per row and can differ from this horizon; the horizon "
            "itself — when the portfolio is valued — is fixed at one year.\n"
            "- **Scenario quality is everything.** The copula choice and the estimated means, "
            "vols and correlations drive the result; estimation error worsens as N grows "
            "(shrinkage/robust estimators are the natural next step).\n"
            "- **α-CVaR — a different ES from the Optimiser tab.** The constraint here is "
            "α-CVaR, the mean of the worst α% of outcomes, which is convex and so solvable as "
            "a linear program. The Optimiser tab's *Realised ES* is instead the thesis's "
            "fixed-threshold mean E[r | r < H]; the two coincide only when H equals the "
            "α-quantile, so the ES figures shown on the two tabs are not directly comparable. "
            "α-CVaR is also the coherent, regulatory-standard measure, and α-CVaR ≥ VaR, so a "
            "CVaR floor conservatively honours the thesis's P(r<H) ≤ α intent.\n"
            "- **Complements the grid.** The grid optimiser is exact for small N and remains "
            "the reference; this engine is the route for large, multi-derivative portfolios. "
            "The validation panel below cross-checks the engine against closed-form values.\n"
            "- **Derivatives in this tab:** the full terminal-payoff library — calls, puts, "
            "straddles, strangles, vertical spreads, safety & aggressive collars, long butterfly, "
            "call condor, reverse convertible, discount & outperformance certificates, and "
            "capital-guaranteed notes (capped/uncapped). The only instrument reserved for the exact "
            "grid engine is the **barrier-M note**: it is path-dependent, and this engine simulates "
            "the horizon return only, not the full price path. (Capital-guaranteed notes here use "
            "100% participation; for a custom participation rate, use the grid engine.)"
        )

    st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;margin:1.4rem 0 0;color:#e7ecf4">⚙️ <span style="color:#E3C77E">Optimisation Parameters</span></div><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin:-.15rem 0 1.2rem">▼</div>', unsafe_allow_html=True)

    _MC_RULE = "<hr style='border:none;border-top:1px solid #30363d;margin:1.2rem 0 0.55rem'>"
    def _mc_head(t, n=None, rule=True):
        badge = ("<span style='display:inline-block;background:#E3C77E;color:#0d1117;border-radius:50%;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;font-size:.82rem;font-weight:700'>" + str(n) + "</span>") if n else ""
        st.markdown("<div style='text-align:center;margin:-.6rem 0 1rem'><span style='display:inline-block;width:290px;box-sizing:border-box;border:1px solid #30363d;background:#0e1521;padding:.28rem 1rem;border-radius:10px;color:#E3C77E;font-weight:600;font-size:.95rem;letter-spacing:.02em;text-align:center'>" + badge + "<span style='display:block;margin-top:.1rem'>" + t + "</span></span></div>", unsafe_allow_html=True)
    def _mc_ai(label, body):
        st.markdown("<div style='display:flex;justify-content:center'><details style='width:290px;box-sizing:border-box;background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;padding:.4rem .7rem;margin:.2rem 0 .8rem;font-size:.78rem'><summary style='cursor:pointer;color:#79b6ff;font-weight:600;list-style:none'>✨ AI-powered: " + label + "</summary><div style='color:#aebccd;margin-top:.4rem;line-height:1.45'>" + body + "</div></details></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.4rem 0 1.4rem'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _mc_head("Data & estimation", 1, rule=False)
        _mc_ai("How these inputs are built", "From your tickers' prices over the chosen window, the app estimates each security's expected return, volatility and the correlation matrix — the moments that drive every Monte-Carlo scenario. The sample case loads the Das &amp; Statman base case — Means: 5%, 10%, 25% | Std devs: 5%, 20%, 50%.")
        mc_source = st.radio(
            "Data source",
            ["Live tickers", "Default (3-asset sample case)"],
            horizontal=True, key="mc_source")
        if mc_source.startswith("Live"):
            mc_tickers_raw = st.text_input(
                "Tickers (comma-separated — add as many as you like)",
                value="AAPL, MSFT, JPM, TLT, XLE, GLD", key="mc_tickers",
                help="The scalable engine is built for large universes. Means, volatilities and "
                     "correlations are estimated from this window.")
            _mc_tk = [t.strip().upper() for t in mc_tickers_raw.split(",") if t.strip()]
            cme1, cme2, cme3 = st.columns(3)
            with cme1:
                mc_start = st.date_input("Estimate from", value=_dt.date(2018, 1, 1), key="mc_start")
            with cme2:
                mc_end = st.date_input("Estimate to", value=_dt.date(2023, 12, 31), key="mc_end")
            with cme3:
                mc_freq = st.selectbox("Frequency", ["Daily", "Monthly"], index=0, key="mc_freq")
            _mc_undl_opts = _mc_tk
        else:
            st.caption("Thesis 3-asset case — expected returns [5%, 10%, 25%], volatilities "
                       "[5%, 20%, 50%], correlation(Mid, High) = 0.4. No prices are fetched; this "
                       "reproduces the **Optimiser tab's default sample**, so you can compare the "
                       "two engines directly on the same inputs.")
            _mc_tk, mc_start, mc_end, mc_freq = [], None, None, "Daily"
            _mc_undl_opts = list(DEFAULT_NAMES)

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.6rem 0 1.6rem'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _mc_head("Scenarios", 2)
        _mc_ai("What do S and the copula mean?", "The engine draws <b>S</b> joint return scenarios from those moments. The <b>copula</b> sets how assets move together in the tails: Gaussian for normal dependence, Student-t for fatter joint crashes. More scenarios cut sampling noise but take longer.")
        cms1, cms2 = st.columns(2)
        with cms1:
            mc_S = st.select_slider("Number of scenarios S",
                                    options=[2000, 5000, 8000, 10000, 15000, 20000, 25000],
                                    value=10000, key="mc_S")
        with cms2:
            mc_cop = st.selectbox("Copula", ["Gaussian (Normal)", "Student-t (tail dependence)"],
                                  index=0, key="mc_cop")
        mc_dof = 5
        if mc_cop.startswith("Student"):
            mc_dof = st.slider("Student-t degrees of freedom (lower = fatter tails)",
                               3, 15, 5, 1, key="mc_dof")

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.6rem 0 1.6rem'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _mc_head("Derivatives  (optional — add multiple)", 3)
        _mc_ai("How derivatives enter the model", "Each row is priced by Black-Scholes on its underlying and added as an extra payoff column evaluated across all S scenarios — so non-normal payoffs (options, spreads, collars, structured notes and certificates) flow straight into the optimisation. Add as many as you like; the only type not available here is the path-dependent barrier-M note (use the Grid optimiser).")
        st.caption("Add one row per derivative — pick a Type and an Underlying. New rows pre-fill "
                   "strike 1.00 (at-the-money), maturity 1 year (settled at intrinsic at the horizon; "
                   "raise it to mark the option to market with its remaining life) and rate 3% — all "
                   "editable. Implied vol stays \"auto\" (the underlying's own volatility, consistent "
                   "with the scenarios); the resolved value per row is listed below the table, where you "
                   "can confirm each derivative's settings. All strikes/levels are × of spot (S₀=1).")
        st.markdown(
            "<details style='background:rgba(74,158,255,.08);border:1px solid #34527a;border-radius:6px;"
            "padding:.4rem .8rem;margin:.1rem 0 .7rem;font-size:.82rem'>"
            "<summary style='cursor:pointer;color:#79b6ff;font-weight:600;list-style:none'>"
            "ℹ️ Which columns each type uses</summary>"
            "<div style='color:#aebccd;margin-top:.45rem;line-height:1.6'>"
            "<b>Strike</b> only: Call, Put, Straddle, Reverse convertible (put), Discount certificate (cap call), "
            "Outperformance certificate, Capital-guaranteed note (guarantee level, e.g. 1.00 = 100%).<br>"
            "<b>Strike + Strike-2</b>: Strangle / spreads / collars (the two strikes), "
            "Long butterfly (center, width), CGN-capped (guarantee level, cap level — e.g. 1.30 = +30%).<br>"
            "<b>Strike + Strike-2 + Strike-3</b>: Call condor (center, inner width, outer width).<br>"
            "Capital-guaranteed notes use 100% participation here; for a custom participation rate or a "
            "<b>Barrier-M note</b> (path-dependent, not priceable from horizon-only scenarios) use the "
            "<b>Grid optimiser</b>.</div></details>", unsafe_allow_html=True)
        import pandas as _pd
        _mc_der_template = _pd.DataFrame(
            {"Type": _pd.Series(dtype="str"), "Underlying": _pd.Series(dtype="str"),
             "Strike": _pd.Series(dtype="float"), "Strike2": _pd.Series(dtype="float"),
             "Strike3": _pd.Series(dtype="float"),
             "Maturity": _pd.Series(dtype="float"), "ImplVol": _pd.Series(dtype="float"),
             "Rate": _pd.Series(dtype="float")})
        mc_der_table = st.data_editor(
            _mc_der_template, num_rows="dynamic", hide_index=True, key="mc_der_table",
            use_container_width=True,
            column_config={
                "Type": st.column_config.SelectboxColumn("Type", options=list(MC_DER_TYPES.keys()),
                                                         width="medium"),
                "Underlying": st.column_config.SelectboxColumn(
                    "Underlying", options=(_mc_undl_opts if _mc_undl_opts else ["(enter tickers)"]), width="small"),
                "Strike": st.column_config.NumberColumn("Strike (×)", min_value=0.1, max_value=3.0,
                                                        step=0.05, format="%.2f", default=1.0),
                "Strike2": st.column_config.NumberColumn("Strike-2 (×)", min_value=0.1, max_value=3.0,
                                                         step=0.05, format="%.2f",
                                                         help="Second strike / level for multi-leg products "
                                                              "(spread, collar, strangle, butterfly width, "
                                                              "condor inner width, CGN cap level)."),
                "Strike3": st.column_config.NumberColumn("Strike-3 (×)", min_value=0.1, max_value=3.0,
                                                         step=0.05, format="%.2f",
                                                         help="Third input — only the Call condor uses it "
                                                              "(outer width)."),
                "Maturity": st.column_config.NumberColumn(
                    "Maturity (yr)", min_value=1.0, max_value=5.0, step=0.25, format="%.2f", default=1.0,
                    help="Option maturity in years. 1.0 = expires at the 1-year horizon (settled at "
                         "intrinsic). Above 1, the option is marked to market at the horizon using its "
                         "remaining life."),
                "ImplVol": st.column_config.NumberColumn(
                    "Impl. vol % (auto)", min_value=1.0, max_value=200.0, step=1.0, format="%.0f",
                    help="Implied volatility for pricing. Blank uses the underlying's own volatility "
                         "(arbitrage-consistent with the scenarios)."),
                "Rate": st.column_config.NumberColumn(
                    "Rate (%)", min_value=0.0, max_value=20.0, step=0.25, format="%.2f", default=3.0,
                    help="Risk-free rate for option pricing. Blank = 3%."),
            })

        # Read-out of the resolved parameters per row (blank cells use the defaults shown), plus a
        # clear flag when a Type is chosen with no underlying. No table mutation here, so a selection
        # shows immediately — no reload/flicker.
        def _mc_isblank(v):
            return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v.strip() == "")
        _mc_preview = []
        _mc_sig_map = dict(zip(DEFAULT_NAMES, DEFAULT_SIGS))
        _mc_tbl_ro = mc_der_table.dropna(how="all") if hasattr(mc_der_table, "dropna") else mc_der_table
        for _, _r in _mc_tbl_ro.iterrows():
            _ty = _r.get("Type")
            if _mc_isblank(_ty) or _ty not in MC_DER_TYPES:
                continue
            if _mc_isblank(_r.get("Underlying")):
                _mc_preview.append(f"⚠️ **{_ty}** — pick an underlying (ignored until you do)")
                continue
            _un = _r.get("Underlying")
            _t  = "1.00" if _mc_isblank(_r.get("Maturity")) else f"{float(_r.get('Maturity')):.2f}"
            _rt = "3.00" if _mc_isblank(_r.get("Rate"))     else f"{float(_r.get('Rate')):.2f}"
            if _mc_isblank(_r.get("ImplVol")):
                _iv = (f"auto = {_mc_sig_map[_un]*100:.0f}% (underlying \u03c3)" if _un in _mc_sig_map
                       else "auto (underlying \u03c3, set on run)")
            else:
                _iv = f"{float(_r.get('ImplVol')):.0f}%"
            _pk1 = None if _mc_isblank(_r.get("Strike"))  else float(_r.get("Strike"))
            _pk2 = None if _mc_isblank(_r.get("Strike2")) else float(_r.get("Strike2"))
            _pk3 = None if _mc_isblank(_r.get("Strike3")) else float(_r.get("Strike3"))
            _pparams, _pdesc, _pwarn = _mc_der_struct(MC_DER_TYPES[_ty], _pk1, _pk2, _pk3)
            if _pparams is None:
                _mc_preview.append(f"\u26a0\ufe0f **{_ty}** on **{_un}** \u2014 {_pwarn}")
            else:
                _mc_preview.append(f"\u2022 **{_ty}** on **{_un}** \u2014 {_pdesc}, maturity {_t} y, vol {_iv}, rate {_rt}%")
        if _mc_preview:
            st.caption("Resolved settings (blank cells use the defaults shown):")
            st.markdown("  \n".join(_mc_preview))

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.6rem 0 1.6rem'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _mc_head("Constraint", 4)
        _mc_ai("What do α and the CVaR floor mean?", "<b>Tail probability α</b> sets how deep into the downside you look — the worst <b>α%</b> of scenarios (α = 5% means the worst 1-in-20 outcomes).<br><br><b>α-CVaR</b> is the average loss across that tail; the linear program maximises expected return while keeping it at or above the floor <b>L</b>, and within each security's min/max weight bounds.")
        cmc1, cmc2 = st.columns(2)
        with cmc1:
            st.session_state.setdefault("mc_alpha", 5)
            mc_alpha = st.slider("Tail probability α", 1, 25, step=1, format="%d%%", key="mc_alpha") / 100.0
        with cmc2:
            st.session_state.setdefault("mc_L", -20)
            mc_L = st.slider("α-CVaR floor L  (mean of worst α% ≥ L)",
                             -40, 0, step=1, format="%d%%", key="mc_L") / 100.0

        # --- Per-security weight bounds (replaces the old single global max-weight slider) ---
        st.markdown("<div style='margin:1rem 0 .15rem;font-weight:600;color:#E3C77E'>"
                    "Per-security weight bounds</div>", unsafe_allow_html=True)
        st.caption("Set a minimum and maximum % for each security. The optimum is always fully "
                   "invested (weights sum to 100%), so each security is held between its floor and "
                   "its cap. Leave a row at 0–100% to leave it unconstrained. Derivative legs are "
                   "not bounded here — the α and L conditions already govern them.")
        _mc_secs = (list(dict.fromkeys(_mc_tk)) if mc_source.startswith("Live")
                    else list(DEFAULT_NAMES))
        mc_bounds_table = None
        if _mc_secs:
            _mc_bounds_template = pd.DataFrame({
                "Security": _mc_secs,
                "Min %": [0] * len(_mc_secs),
                "Max %": [100] * len(_mc_secs),
            })
            mc_bounds_table = st.data_editor(
                _mc_bounds_template, hide_index=True, use_container_width=True,
                disabled=["Security"],
                key="mc_bounds::" + "|".join(_mc_secs),   # fresh editor when the universe changes
                column_config={
                    "Security": st.column_config.TextColumn("Security", width="medium"),
                    "Min %": st.column_config.NumberColumn(
                        "Min %", min_value=0, max_value=100, step=1, format="%d%%"),
                    "Max %": st.column_config.NumberColumn(
                        "Max %", min_value=0, max_value=100, step=1, format="%d%%"),
                })
        else:
            st.caption("Enter at least one ticker above to set per-security bounds.")

        # Resolve bounds (fractions, keyed by security) and run a live feasibility check.
        mc_sec_max, mc_sec_min, mc_bounds_ok = {}, {}, True
        if mc_bounds_table is not None and len(mc_bounds_table):
            _bt = mc_bounds_table.fillna({"Min %": 0, "Max %": 100})
            _secn = _bt["Security"].tolist()
            _mins = [float(x) for x in _bt["Min %"].tolist()]
            _maxs = [float(x) for x in _bt["Max %"].tolist()]
            for _s, _lo, _hi in zip(_secn, _mins, _maxs):
                mc_sec_min[_s] = _lo / 100.0
                mc_sec_max[_s] = _hi / 100.0
            _row_bad = [_secn[i] for i in range(len(_secn)) if _mins[i] > _maxs[i]]
            _sum_max, _sum_min = sum(_maxs), sum(_mins)
            _msgs = []
            if _row_bad:
                _msgs.append("Min exceeds Max for: " + ", ".join(_row_bad) + ".")
            if _sum_max < 100:
                _msgs.append(f"Max weights add up to {_sum_max:.0f}% — below 100%, so no "
                             f"fully-invested portfolio fits. Raise some caps.")
            if _sum_min > 100:
                _msgs.append(f"Min weights add up to {_sum_min:.0f}% — above 100%, which "
                             f"over-invests the portfolio. Lower some floors.")
            if _msgs:
                mc_bounds_ok = False
                st.error("  \n".join("• " + m for m in _msgs))

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.6rem 0 1.6rem'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _mc_head("Validation", 5)
        _mc_ai("What does validation check?", "When enabled, the scenario-based optimum is cross-checked against the closed-form Gaussian (analytic) values — a quick gauge of Monte-Carlo sampling error. Large gaps suggest raising the number of scenarios <b>S</b> for a steadier result.")
        mc_validate = st.checkbox("Run validation checks against closed-form values "
                                  "(Gaussian copula)", value=True, key="mc_validate")

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.6rem 0 2rem'>", unsafe_allow_html=True)
    st.markdown("<style>.st-key-mc_run button{font-size:1.1rem;font-weight:700;"
                "padding:0.85rem 1rem;border-radius:8px;}</style>", unsafe_allow_html=True)
    _mcb = st.columns([1, 2, 1])
    with _mcb[1]:
        mc_run = st.button("▶  Run scalable optimiser", type="primary", key="mc_run",
                           use_container_width=True, disabled=not mc_bounds_ok)
    if not mc_bounds_ok:
        st.caption("Adjust the per-security weight bounds above to enable the run.")

    if mc_run:
        try:
            copula = "gaussian" if mc_cop.startswith("Gaussian") else "t"
            if mc_source.startswith("Live"):
                tickers = list(dict.fromkeys(_mc_tk))
                if len(tickers) < 2:
                    raise RuntimeError("Please enter at least two tickers.")
                if not (mc_start < mc_end):
                    raise RuntimeError("Estimation 'from' must precede 'to'.")
                with st.spinner("Estimating from prices…"):
                    px, err = fetch_close_prices(tickers, mc_start, mc_end)
                    if err:
                        raise RuntimeError(f"Price data: {err}")
                    means, sigs, corr, names, _ = stats_from_prices(px, mc_freq)
                    if len(names) < 2:
                        raise RuntimeError("Fewer than two usable securities after cleaning.")
            else:
                means, sigs = list(DEFAULT_MEANS), list(DEFAULT_SIGS)
                corr = [r[:] for r in DEFAULT_CORR]; names = list(DEFAULT_NAMES)
            N = len(names)
            cov = corr_to_cov(sigs, corr)
            with st.spinner("Generating scenarios…"):
                R_sec = mc_generate_scenarios(means, sigs, corr, S=int(mc_S),
                                              copula=copula, dof=int(mc_dof), seed=1)

                # parse derivative rows
                der_specs = []
                der_warn = []
                tbl = mc_der_table.dropna(how="all") if hasattr(mc_der_table, "dropna") else mc_der_table
                for _, row in tbl.iterrows():
                    typ_lbl = row.get("Type"); undl = row.get("Underlying")
                    if _mc_isblank(typ_lbl) or typ_lbl not in MC_DER_TYPES:
                        continue
                    if _mc_isblank(undl):
                        der_warn.append(f"{typ_lbl}: no underlying selected \u2014 not included"); continue
                    if undl not in names:
                        der_warn.append(f"{typ_lbl} on {undl} (ticker unavailable)"); continue
                    dt = MC_DER_TYPES[typ_lbl]
                    k1 = row.get("Strike"); k2 = row.get("Strike2"); k3 = row.get("Strike3")
                    k1 = float(k1) if k1 == k1 and k1 is not None else None
                    k2 = float(k2) if k2 == k2 and k2 is not None else None
                    k3 = float(k3) if k3 == k3 and k3 is not None else None
                    params, _pdesc, _pwarn = _mc_der_struct(dt, k1, k2, k3)
                    if params is None:
                        der_warn.append(f"{typ_lbl}: {_pwarn}"); continue
                    _mat = row.get("Maturity"); _iv = row.get("ImplVol"); _rr = row.get("Rate")
                    _mat = float(_mat) if _mat == _mat and _mat is not None else 1.0
                    _iv  = float(_iv)  if _iv  == _iv  and _iv  is not None else None
                    _rr  = float(_rr)  if _rr  == _rr  and _rr  is not None else None
                    der_specs.append({"der_type": dt, "params": params, "desc": _pdesc,
                                      "underlying_idx": names.index(undl),
                                      "label": f"{typ_lbl}·{undl}",
                                      "T": _mat,
                                      "vol_override": (_iv / 100.0) if _iv is not None else None,
                                      "r": (_rr / 100.0) if _rr is not None else 0.03})

                R_full, labels, errs = mc_build_matrix(R_sec, der_specs, np.array(sigs), names)
                der_warn += [f"{e} (pricing failed)" for e in errs]

            with st.spinner("Solving the CVaR linear program…"):
                # Map per-security bounds onto the full column vector (securities first,
                # then derivative columns which stay unconstrained at [0, 1]).
                _ncols = R_full.shape[1]
                w_max_vec = [mc_sec_max.get(nm, 1.0) for nm in names] + [1.0] * (_ncols - len(names))
                w_min_vec = [mc_sec_min.get(nm, 0.0) for nm in names] + [0.0] * (_ncols - len(names))
                _has_bounds = (any(v < 1.0 for v in w_max_vec[:len(names)])
                               or any(v > 0.0 for v in w_min_vec[:len(names)]))
                w, er, es, res = mc_max_return_cvar(R_full, mc_alpha, mc_L,
                                                    w_max=w_max_vec, w_min=w_min_vec)

            st.markdown("---")
            st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;margin:0;color:#E3C77E"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>Optimisation results</div><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin:-.15rem 0 1.7rem">▼</div>',
                        unsafe_allow_html=True)
            if der_warn:
                st.warning("Skipped derivative rows: " + "; ".join(der_warn))
            if der_specs:
                _dp = []
                _sig_arr = np.array(sigs, dtype=float)
                for d in der_specs:
                    _ui = d["underlying_idx"]; _ov = d.get("vol_override")
                    _vol = _ov if _ov is not None else float(_sig_arr[_ui])
                    _voltxt = (f"{_vol*100:.1f}%" if _ov is not None
                               else f"{_vol*100:.1f}% (auto \u2014 {names[_ui]} \u03c3)")
                    _ks = d.get("desc") or "\u2014"
                    _dp.append(f"\u2022 **{d['label']}** \u2014 {_ks}, maturity {d['T']:.2f} y, "
                               f"vol {_voltxt}, rate {d['r']*100:.1f}%")
                with st.expander(f"Derivative pricing used ({len(der_specs)})", expanded=True):
                    st.markdown("  \n".join(_dp))
            if w is None:
                st.error(f"No feasible portfolio reaches an ES of {mc_L:.0%} for this universe. "
                         f"Loosen the floor (more negative L), widen the per-security weight "
                         f"bounds, or change the inputs.")
            else:
                K = len(labels) - N
                # portfolio stats for the details box
                _port = R_full @ w
                _sig = float(_port.std())
                _mn = float(_port.mean())
                _skew = float((((_port - _mn) / _sig) ** 3).mean()) if _sig > 0 else 0.0
                # The CVaR LP drives the tail-average return onto the floor L,
                # so at the optimum ES binds L exactly; tiny solver float noise
                # can leave it a hair past the floor. Compare at the 2-dp
                # display precision (as the grid badge does) so the badge can
                # never contradict the figures shown — e.g. ES -20.00% vs
                # L -20.00% reads as feasible (binding), not "✗ < L".
                _es2 = round(es * 100, 2)
                _L2 = round(mc_L * 100, 2)
                _feas = _es2 >= _L2
                _badge = (f'<span style="color:#16a34a">✓ feasible — α-CVaR {_es2:.2f}% ≥ L {_L2:.2f}%</span>'
                          if _feas else
                          f'<span style="color:#dc2626">✗ α-CVaR {_es2:.2f}% &lt; L {_L2:.2f}%</span>')
                _wmax_txt = " · per-asset weight bounds applied" if _has_bounds else ""
                _univ = (f"{N} securit{'y' if N == 1 else 'ies'}"
                         + (f" + {K} derivative{'s' if K != 1 else ''}" if K else ""))
                _detbox = (
                    '<div style="background:#1b2330;border:1px solid #30363d;border-radius:8px;'
                    'padding:.8rem 1.1rem;flex:1;min-height:0;box-sizing:border-box;overflow:auto">'
                    '<div style="color:#E3C77E;font-weight:700;font-size:.98rem;margin-bottom:.5rem;text-align:center">'
                    '<span style="color:#f59e0b;font-size:1.05rem;margin-right:.4rem;vertical-align:middle">★</span>' 
                    'Scalable CVaR optimum — Monte-Carlo</div>'
                    '<div style="color:#c9d1d9;font-size:.86rem;line-height:1.75">'
                    f'<b>Expected return:</b> {er*100:.2f}% &nbsp;·&nbsp; <b>Volatility:</b> {_sig*100:.2f}% '
                    f'&nbsp;·&nbsp; <b>Skewness:</b> {_skew:.3f}<br>'
                    f'<b>Realised α-CVaR</b> (mean of worst α = {mc_alpha*100:.0f}%)<b>:</b> '
                    f'{es*100:.2f}% &nbsp; {_badge}<br>'
                    f'<b>Universe:</b> {_univ}<br>'
                    f'<b>Scenarios:</b> {int(mc_S):,} ({copula} copula){_wmax_txt}<br>'
                    f'<b>Objective:</b> maximise E[r] subject to α-CVaR ≥ {mc_L*100:.0f}%<br>'
                    '<span style="color:#8b949e;font-size:.8rem">Approximate scenario-based optimum '
                    '(CVaR linear program) — complements the exact grid engine.</span>'
                    '</div></div>')
                _metbox = (
                    '<div style="background:#0d1a2e;border:1px solid #30363d;border-radius:8px;'
                    'padding:.85rem .95rem">'
                    '<div style="display:flex;gap:.5rem;text-align:center">'
                    f'<div style="flex:1"><div style="color:#E3C77E;font-weight:700;font-size:.95rem;line-height:1.2">'
                    f'Expected return</div><div style="color:#fafafa;font-size:1.4rem;font-weight:600;'
                    f'margin-top:.25rem">{er*100:.2f}%</div></div>'
                    f'<div style="flex:1"><div style="color:#E3C77E;font-weight:700;font-size:.95rem;line-height:1.2">'
                    f'Realised α-CVaR</div><div style="color:#fafafa;font-size:1.4rem;font-weight:600;'
                    f'margin-top:.25rem">{es*100:.2f}%</div></div>'
                    f'<div style="flex:1"><div style="color:#E3C77E;font-weight:700;font-size:.95rem;line-height:1.2">'
                    f'Securities / derivatives</div><div style="color:#fafafa;font-size:1.4rem;font-weight:600;'
                    f'margin-top:.25rem">{N} / {K}</div></div>'
                    '</div></div>')

                # frontier computed once
                with st.spinner("Tracing the return / tail-risk frontier…"):
                    _floors = sorted(set([-0.30, -0.25, -0.20, -0.15, -0.10, -0.05]
                                         + [round(float(mc_L), 4)]))
                    fr = mc_frontier(R_full, mc_alpha, _floors, w_max=w_max_vec, w_min=w_min_vec)
                _okfr = [r for r in fr if r["ok"]]

                # Benchmark portfolios (computed once — used by both the frontier chart and the table)
                _bm_points = []
                _rf_bm = 0.03
                try:
                    from core import benchmarks as _bm
                    _rf_bm = getattr(_bm, "RF_ANNUAL", 0.03)
                    _mu_sec = R_sec.mean(axis=0)
                    _cov_sec = np.cov(R_sec, rowvar=False)
                    for _blbl, _bw in _bm.benchmark_set(_mu_sec, _cov_sec, rf=_rf_bm):
                        _bs = _bm.stats_from_scenarios(R_sec, _bw, alpha=mc_alpha, rf=_rf_bm)
                        _bm_points.append((_blbl, _bs["er"], _bs["vol"], _bs["sharpe"], _bs["es"]))
                    # Markowitz mean-variance portfolio matched to the optimum's expected return
                    _mvw = _bm.return_matched_mv_weights(_mu_sec, _cov_sec, er)
                    if _mvw is not None:
                        _bs = _bm.stats_from_scenarios(R_sec, _mvw, alpha=mc_alpha, rf=_rf_bm)
                        _bm_points.append(("Mean-variance (Markowitz)", _bs["er"], _bs["vol"],
                                           _bs["sharpe"], _bs["es"]))
                except Exception:
                    _bm_points = []

                # Markowitz MV efficient frontier in (volatility, return) space
                _mv_curve = []
                try:
                    for _tr, _w in _bm.mv_frontier_weights(_mu_sec, _cov_sec, n_points=18):
                        _s = _bm.stats_from_scenarios(R_sec, _w, alpha=mc_alpha, rf=_rf_bm)
                        _mv_curve.append((_s["vol"], _s["er"]))
                    _mv_curve.sort(key=lambda p: p[0])
                except Exception:
                    _mv_curve = []

                # Row: left = metrics box on top of details box (height matched to chart);
                #      right = frontier chart
                colA_l, colA_r = st.columns([1, 1])
                with colA_l:
                    st.markdown(
                        '<div style="display:flex;flex-direction:column;gap:12px;height:400px">'
                        + _metbox + _detbox + '</div>', unsafe_allow_html=True)
                with colA_r:
                    if _okfr:
                        _drawn = False
                        try:
                            import plotly.graph_objects as _go
                            xs = [r["L"] * 100 for r in _okfr]
                            ys = [r["er"] * 100 for r in _okfr]
                            es_pct = [r["es"] * 100 for r in _okfr]
                            fig = _go.Figure()
                            fig.add_trace(_go.Scatter(
                                x=xs, y=ys, mode="lines+markers", name="Frontier",
                                line=dict(color="#4a9eff", width=2.5),
                                marker=dict(size=8, color="#4a9eff"),
                                customdata=es_pct,
                                hovertemplate="α-CVaR floor L: %{x:.1f}%<br>Max E[r]: %{y:.2f}%"
                                              "<br>Realised α-CVaR: %{customdata:.2f}%<extra></extra>"))
                            fig.add_trace(_go.Scatter(
                                x=[mc_L * 100], y=[er * 100], mode="markers",
                                name="Portfolio (1) — Scalable CVaR optimum",
                                marker=dict(size=18, color="#f59e0b", symbol="star",
                                            line=dict(color="#ffffff", width=1.2)),
                                customdata=[es * 100],
                                hovertemplate="<b>Portfolio (1) — Scalable CVaR optimum</b><br>α-CVaR floor L: %{x:.1f}%"
                                              "<br>E[r]: %{y:.2f}%<br>Realised α-CVaR: %{customdata:.2f}%<extra></extra>"))
                            if _bm_points:
                                fig.add_trace(_go.Scatter(
                                    x=[p[4] * 100 for p in _bm_points],
                                    y=[p[1] * 100 for p in _bm_points],
                                    mode="markers+text", name="Benchmarks (securities only)",
                                    marker=dict(size=11, color="#9aa7bd", symbol="diamond",
                                                line=dict(color="#0d1117", width=1)),
                                    text=[("(0) Markowitz" if p[0].startswith("Mean-variance")
                                           else p[0].split(" (")[0]) for p in _bm_points],
                                    textposition="top center", textfont=dict(color="#9aa7bd", size=9),
                                    customdata=[[p[2] * 100, p[3]] for p in _bm_points],
                                    hovertemplate="<b>%{text}</b><br>E[r]: %{y:.2f}%<br>"
                                                  "Realised α-CVaR: %{x:.2f}%<br>Vol: %{customdata[0]:.2f}%"
                                                  "<br>Sharpe: %{customdata[1]:.2f}<extra></extra>"))
                            fig.update_layout(
                                template="plotly_dark", paper_bgcolor="#1b2330",
                                plot_bgcolor="#0e1521", height=400,
                                title=dict(text="Return / Tail-Risk Frontier (Monte-Carlo + CVaR)",
                                           font=dict(color="#E3C77E", size=15),
                                           x=0.5, xanchor="center", xref="paper"),
                                margin=dict(l=10, r=10, t=52, b=40), hovermode="closest",
                                xaxis=dict(title=dict(text="α-CVaR floor L (%)",
                                                      font=dict(color="#c0c8d8", size=12)),
                                           color="#c0c8d8", gridcolor="#1e2130", zerolinecolor="#2a2a3a"),
                                yaxis=dict(title=dict(text="Max expected return (%)",
                                                      font=dict(color="#c0c8d8", size=12)),
                                           color="#c0c8d8", gridcolor="#1e2130", zerolinecolor="#2a2a3a"),
                                legend=dict(bgcolor="rgba(26,26,46,0.9)", bordercolor="#3a3a5a",
                                            borderwidth=1, font=dict(color="white", size=9), x=0.01, y=0.99),
                                hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#1a6bbf",
                                                font=dict(color="white", size=11)))
                            fig.update_xaxes(showgrid=True, gridcolor="#27344e", gridwidth=1, griddash="dot",
                                             showline=True, linecolor="#46566f", linewidth=1, mirror=True,
                                             showspikes=True, spikethickness=1)
                            fig.update_yaxes(showgrid=True, gridcolor="#27344e", gridwidth=1, griddash="dot",
                                             showline=True, linecolor="#46566f", linewidth=1, mirror=True,
                                             showspikes=True, spikethickness=1)
                            _dtxt = f"{N} securities" + (f" + {K} deriv." if K else "")
                            fig.add_annotation(
                                x=mc_L * 100, y=er * 100, ax=46, ay=-58,
                                xref="x", yref="y", axref="pixel", ayref="pixel",
                                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#f59e0b",
                                text=("<b>Portfolio (1)</b><br><b>Scalable CVaR optimum</b><br>"
                                      f"E[r] = {er*100:.1f}%&nbsp; | &nbsp;Vol = {_sig*100:.1f}%<br>"
                                      f"Skew = {_skew:.2f}<br>"
                                      f"Realised α-CVaR = {es*100:.1f}%&nbsp; (L = {mc_L*100:.0f}%)<br>"
                                      f"{_dtxt} · {'✓ feasible' if _feas else '✗ infeasible'}"),
                                font=dict(color="#f59e0b", size=9),
                                bgcolor="rgba(13,17,23,0.92)", bordercolor="#f59e0b", borderwidth=1,
                                align="left", xanchor="left")
                            st.plotly_chart(fig, use_container_width=True,
                                            config={'edits': {'annotationPosition': True, 'annotationTail': True, 'legendPosition': True}, 'displayModeBar': True})
                            st.caption("⭐ marks the Scalable CVaR optimum (your resulting portfolio); "
                                       "◆ grey diamonds are the securities-only benchmarks (plotted at their "
                                       "expected return vs realised α-CVaR) — they sit below/right of the "
                                       "frontier, which dominates them. Hover any point for its coordinates; drag to zoom.")
                            _drawn = True
                        except Exception:
                            pass
                        if not _drawn:
                            try:
                                import altair as alt
                                dff = _pd.DataFrame({
                                    "α-CVaR floor L (%)": [r["L"] * 100 for r in _okfr],
                                    "Max expected return (%)": [r["er"] * 100 for r in _okfr],
                                    "Realised α-CVaR (%)": [r["es"] * 100 for r in _okfr]})
                                _base = alt.Chart(dff).encode(
                                    x=alt.X("α-CVaR floor L (%):Q", scale=alt.Scale(zero=False)),
                                    y=alt.Y("Max expected return (%):Q", scale=alt.Scale(zero=False)))
                                _line = _base.mark_line(color="#4a9eff", strokeWidth=2.5)
                                _pts = _base.mark_circle(color="#4a9eff", size=95).encode(
                                    tooltip=[alt.Tooltip("α-CVaR floor L (%):Q", format=".1f"),
                                             alt.Tooltip("Max expected return (%):Q", format=".2f"),
                                             alt.Tooltip("Realised α-CVaR (%):Q", format=".2f")])
                                _star = alt.Chart(_pd.DataFrame({
                                    "α-CVaR floor L (%)": [mc_L * 100],
                                    "Max expected return (%)": [er * 100]})).mark_point(
                                    shape="diamond", size=260, color="#f59e0b", filled=True).encode(
                                    x="α-CVaR floor L (%):Q", y="Max expected return (%):Q")
                                st.altair_chart((_line + _pts + _star).interactive().properties(height=360),
                                                use_container_width=True)
                                st.caption("◆ marks the Scalable CVaR optimum. Hover points for coordinates.")
                            except Exception as _fe:
                                st.caption(f"(frontier chart unavailable: {_fe})")
                    else:
                        st.caption("No feasible frontier points for these settings.")

                # ── Classical Markowitz frontier (return vs volatility) ──
                try:
                    if _mv_curve and _bm_points:
                        import plotly.graph_objects as _go2
                        _mvf = _go2.Figure()
                        _mvf.add_trace(_go2.Scatter(
                            x=[p[0] * 100 for p in _mv_curve], y=[p[1] * 100 for p in _mv_curve],
                            mode="lines", name="Mean-variance frontier",
                            line=dict(color="#a855f7", width=2.5, dash="dash"),
                            hovertemplate="Markowitz MV frontier<br>Vol: %{x:.2f}%<br>E[r]: %{y:.2f}%<extra></extra>"))
                        _mvf.add_trace(_go2.Scatter(
                            x=[p[2] * 100 for p in _bm_points], y=[p[1] * 100 for p in _bm_points],
                            mode="markers+text", name="Benchmarks (securities only)",
                            marker=dict(size=11, color="#9aa7bd", symbol="diamond",
                                        line=dict(color="#0d1117", width=1)),
                            text=[("(0) Markowitz" if p[0].startswith("Mean-variance")
                                   else p[0].split(" (")[0]) for p in _bm_points],
                            textposition="top center", textfont=dict(color="#9aa7bd", size=9),
                            customdata=[[p[3], p[4] * 100] for p in _bm_points],
                            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>E[r]: %{y:.2f}%"
                                          "<br>Sharpe: %{customdata[0]:.2f}<br>α-CVaR: %{customdata[1]:.2f}%<extra></extra>"))
                        _mvf.add_trace(_go2.Scatter(
                            x=[_sig * 100], y=[er * 100], mode="markers",
                            name="Portfolio (1) — Scalable CVaR optimum",
                            marker=dict(size=18, color="#f59e0b", symbol="star",
                                        line=dict(color="#ffffff", width=1.2)),
                            customdata=[es * 100],
                            hovertemplate="<b>Portfolio (1) — Scalable CVaR optimum</b><br>Vol: %{x:.2f}%"
                                          "<br>E[r]: %{y:.2f}%<br>α-CVaR: %{customdata:.2f}%<extra></extra>"))
                        _mvf.update_layout(
                            template="plotly_dark", paper_bgcolor="#1b2330", plot_bgcolor="#0e1521",
                            height=380, hovermode="closest", margin=dict(l=10, r=10, t=52, b=44),
                            title=dict(text="Markowitz Mean-Variance Frontier (return vs volatility)",
                                       font=dict(color="#E3C77E", size=15), x=0.5, xanchor="center", xref="paper"),
                            xaxis=dict(title=dict(text="Volatility (annualised, %)",
                                                  font=dict(color="#c0c8d8", size=12)),
                                       color="#c0c8d8", gridcolor="#1e2130", zerolinecolor="#2a2a3a"),
                            yaxis=dict(title=dict(text="Expected return (%)",
                                                  font=dict(color="#c0c8d8", size=12)),
                                       color="#c0c8d8", gridcolor="#1e2130", zerolinecolor="#2a2a3a"),
                            legend=dict(bgcolor="rgba(26,26,46,0.9)", bordercolor="#3a3a5a", borderwidth=1,
                                        font=dict(color="white", size=9), x=0.01, y=0.99))
                        _mvf.update_xaxes(showgrid=True, gridcolor="#27344e", gridwidth=1, griddash="dot",
                                          showline=True, linecolor="#46566f", linewidth=1, mirror=True)
                        _mvf.update_yaxes(showgrid=True, gridcolor="#27344e", gridwidth=1, griddash="dot",
                                          showline=True, linecolor="#46566f", linewidth=1, mirror=True)
                        _dtxt2 = f"{N} securities" + (f" + {K} deriv." if K else "")
                        _mvf.add_annotation(
                            x=_sig * 100, y=er * 100, ax=58, ay=52,
                            xref="x", yref="y", axref="pixel", ayref="pixel",
                            showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#f59e0b",
                            text=("<b>Portfolio (1)</b><br><b>Scalable CVaR optimum</b><br>"
                                  f"E[r] = {er*100:.1f}%&nbsp; | &nbsp;Vol = {_sig*100:.1f}%<br>"
                                  f"Skew = {_skew:.2f}<br>"
                                  f"Realised α-CVaR = {es*100:.1f}%<br>"
                                  f"{_dtxt2}"),
                            font=dict(color="#f59e0b", size=9),
                            bgcolor="rgba(13,17,23,0.92)", bordercolor="#f59e0b", borderwidth=1,
                            align="left", xanchor="left")
                        st.plotly_chart(_mvf, use_container_width=True,
                                        config={'edits': {'annotationPosition': True, 'annotationTail': True,
                                                          'legendPosition': True}, 'displayModeBar': True})
                        st.caption("Classical Markowitz bullet (return vs volatility): the dashed frontier is the "
                                   "highest return for each volatility using the securities only (long-only). "
                                   "Minimum-variance sits at the left tip; minimum-variance and max-Sharpe lie on "
                                   "the frontier, while equal-weight sits inside it (mean-variance-dominated). "
                                   "Portfolio (1) — the ⭐ — is optimised for tail risk (α-CVaR), not variance, so "
                                   "it need not lie on this frontier: a return-shaping derivative can push it "
                                   "above the frontier, while a tail-hedging derivative (e.g. a strangle) pushes "
                                   "it below — the hedge adds variance to buy a better tail, which a variance-only "
                                   "view counts as 'worse'. That trade-off is best judged on the tail-risk chart "
                                   "above, not here. Securities-only benchmarks; not investment advice.")
                except Exception:
                    pass

                # ── Benchmark comparison (naive / classical reference portfolios) ──
                try:
                    if not _bm_points:
                        raise RuntimeError("benchmark points unavailable")
                    _opt_sharpe = (er - _rf_bm) / _sig if _sig > 1e-12 else float("nan")

                    def _bp(_lbl):
                        for _p in _bm_points:
                            if _p[0] == _lbl:
                                return _p  # (label, er, vol, sharpe, es)
                        return None
                    # Order: Portfolio (0) Markowitz, Portfolio (1) optimum, then named references
                    _spec = [
                        ("Portfolio (0) — Mean-variance (Markowitz)", _bp("Mean-variance (Markowitz)"), False, "#c9a6f5"),
                        ("Portfolio (1) — Scalable CVaR optimum", ("opt", er, _sig, _opt_sharpe, es), True, "#f5b942"),
                        ("Equal-weight (1/N)", _bp("Equal-weight (1/N)"), False, "#dbe7ff"),
                        ("Minimum-variance", _bp("Minimum-variance"), False, "#dbe7ff"),
                        ("Max-Sharpe (tangency)", _bp("Max-Sharpe (tangency)"), False, "#dbe7ff"),
                    ]
                    _bm_rows = []
                    for _nm, _tp, _hot, _col in _spec:
                        if _tp is None:
                            continue
                        _bm_rows.append((_nm, _tp[1], _tp[2], _tp[3], _tp[4], _hot, _col))

                    def _bmf(x, pct=True):
                        if x is None or not np.isfinite(x):
                            return "—"
                        return f"{x*100:.2f}%" if pct else f"{x:.2f}"
                    _btr = ""
                    for _lbl, _r, _v, _sh, _e, _hot, _col in _bm_rows:
                        _bg = "background:#13233b;" if _hot else ""
                        _nmc = _col
                        _fw = "700" if _hot else "500"
                        _btr += (f'<tr style="border-top:1px solid #1a2a3a;{_bg}">'
                                 f'<td style="padding:.4rem .7rem;color:{_nmc};font-weight:{_fw}">{_lbl}</td>'
                                 f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_bmf(_r)}</td>'
                                 f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_bmf(_v)}</td>'
                                 f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_bmf(_sh, pct=False)}</td>'
                                 f'<td style="padding:.4rem .7rem;text-align:center;color:#dbe7ff">{_bmf(_e)}</td></tr>')
                    st.markdown(
                        '<div style="background:#0d1a2e;border:none;border-radius:8px;'
                        'padding:.6rem .8rem;margin:1.1rem 0 1rem 0;overflow-x:auto">'
                        '<div style="color:#4a9eff;font-weight:700;font-size:.95rem;margin-bottom:.4rem;text-align:center">'
                        'Benchmark comparison — same securities</div>'
                        '<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
                        '<thead><tr style="color:#9fb3d1">'
                        '<th style="padding:.3rem .7rem;text-align:left">Portfolio</th>'
                        '<th style="padding:.3rem .7rem;text-align:center">Expected return</th>'
                        '<th style="padding:.3rem .7rem;text-align:center">Volatility</th>'
                        '<th style="padding:.3rem .7rem;text-align:center">Sharpe</th>'
                        '<th style="padding:.3rem .7rem;text-align:center">&alpha;-CVaR</th>'
                        '</tr></thead><tbody>' + _btr + '</tbody></table>'
                        '<div style="color:#6b7f99;font-size:.7rem;margin-top:.4rem">'
                        'Benchmarks are long-only, fully-invested portfolios of the same securities '
                        '(no derivatives): equal-weight, minimum-variance and max-Sharpe (tangency), '
                        'evaluated on the same Monte-Carlo scenarios. Sharpe uses r<sub>f</sub> = '
                        f'{_rf_bm*100:.0f}% p.a. The CVaR optimum may include derivatives, so it is not '
                        'directly comparable on every metric — this table is for context, not ranking. '
                        'Not investment advice.</div>'
                        '</div>', unsafe_allow_html=True)

                    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                    with st.expander("**What each of these portfolios is**", expanded=False,
                                     icon=":material/insights:"):
                        st.markdown('''
<div style="background:#ffffff;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;color:#111111;font-size:.82rem">
<b style="color:#a855f7">Portfolio (0) — Mean-variance (Markowitz)</b> — the Markowitz frontier point <i>matched to Portfolio (1)'s expected return</i>: the lowest-variance securities portfolio earning the same expected return as the CVaR optimum. The cleanest like-for-like — same return, compare the tail risk (the grid optimiser's Portfolio (0))<br>
<b style="color:#f59e0b">Portfolio (1) — Scalable CVaR optimum</b> — your resulting portfolio (the ⭐): it maximises expected return subject to the α-CVaR floor L. It may include derivatives, so it is not directly comparable to the securities-only benchmarks on every metric<br>
<b style="color:#7d8aa0">Equal-weight (1/N)</b> — naive diversification: every security receives the same weight. An assumption-free reference that needs no estimated inputs<br>
<b style="color:#7d8aa0">Minimum-variance</b> — the long-only, fully-invested portfolio with the lowest variance (no return target): the lowest-volatility benchmark of the set<br>
<b style="color:#7d8aa0">Max-Sharpe (tangency)</b> — the classical mean-variance optimum: the long-only portfolio with the highest Sharpe ratio (excess return per unit of volatility) at the risk-free rate shown
<div style="margin-top:.55rem;color:#555">The <b style="color:#a855f7">purple Markowitz frontier</b> (the return-vs-volatility chart above) has minimum-variance at its left tip; minimum-variance and max-Sharpe sit on it, while equal-weight sits inside it (dominated). Equal-weight, minimum-variance and max-Sharpe are kept as named references rather than numbered, since the grid optimiser reserves Portfolio (2)/(3) for its with-derivative portfolios. All are long-only and fully invested (weights sum to 100%, no shorting), built from the same securities on the same Monte-Carlo scenarios. Shown for context, not as recommendations.</div>
</div>
''', unsafe_allow_html=True)
                except Exception as _bme:
                    st.caption(f"(benchmark comparison unavailable: {_bme})")

                # ── P&L distribution of the optimal portfolio (worst-α tail shaded) ──
                st.markdown('#### <span style="color:#E3C77E">Portfolio return distribution</span>', unsafe_allow_html=True)
                st.caption("Simulated return of the chosen portfolio across every scenario. "
                           "The shaded tail is the worst α; its average is the realised "
                           "α-CVaR, held at or above the floor L — so the shaded region "
                           "is exactly what the CVaR constraint controls. The dashed lines overlay the "
                           "same-return Markowitz portfolio (0) and equal-weight (securities only) on the "
                           "same scenarios. Note: a taller, narrower curve just means returns are more "
                           "*concentrated* (lower variance) — not better. Portfolio (1) is deliberately more "
                           "spread out (the hedge widens the body and stretches the right tail), so its bars "
                           "sit lower; judge it by the **left tail** and the CVaR floor, not by peak height — "
                           "there, the hedged optimum pulls in its worst outcomes versus the same-return "
                           "mean-variance portfolio.")
                # Overlay same-return Markowitz (0) + equal-weight distributions for tail-shape comparison
                _pnl_overlays = []
                try:
                    from core import benchmarks as _bm2
                    _mu_s = R_sec.mean(axis=0); _cov_s = np.cov(R_sec, rowvar=False)
                    _mk_w = _bm2.return_matched_mv_weights(_mu_s, _cov_s, er)
                    if _mk_w is not None:
                        _pnl_overlays.append(("Portfolio (0) — Markowitz (same return)",
                                              R_sec @ _mk_w, "#fb923c"))
                    _ew_w = _bm2.equal_weight(R_sec.shape[1])
                    _pnl_overlays.append(("Equal-weight (1/N)", R_sec @ _ew_w, "#34d399"))
                except Exception:
                    _pnl_overlays = []
                st.plotly_chart(_mc_pnl_distribution(_port, mc_alpha, es, mc_L, er, overlays=_pnl_overlays),
                                use_container_width=True, config={'displayModeBar': True})

                # ── Joint return scenarios (copula scatter) — actual MC scenarios used ──
                _mc_joint_scatter_view(R_sec, names, w, mc_alpha)

                # Portfolio weights box (full width, below the row)
                _rows = []
                for i in range(len(labels)):
                    is_der = i >= N
                    _col = "#f59e0b" if is_der else DONUT_COLORS[i % len(DONUT_COLORS)]
                    _rows.append((labels[i], float(w[i]), is_der, _col))
                _rows.sort(key=lambda r: r[1], reverse=True)
                _bar = ""
                for lbl, wi, is_der, _c in _rows:
                    pct = wi * 100.0
                    width = max(0.0, min(100.0, pct))
                    _bar += (
                        f'<div style="margin-bottom:.45rem">'
                        f'<div><span style="color:{_c};font-weight:600">{lbl}</span>'
                        f'<span style="color:{_c}"> — {pct:.1f}%</span></div>'
                        f'<div style="height:7px;background:#1a2a3a;border-radius:3px;margin-top:3px">'
                        f'<div style="height:7px;width:{width:.1f}%;background:{_c};border-radius:3px"></div>'
                        f'</div></div>')
                _dcolors = ["#f59e0b" if i >= N else DONUT_COLORS[i % len(DONUT_COLORS)]
                            for i in range(len(labels))]
                _donut = make_donut_svg(list(w), list(labels), _dcolors, size=170)
                st.markdown(
                    '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                    'padding:.75rem .95rem;margin-top:14px">'
                    '<div style="color:#ffffff;font-weight:700;font-size:.95rem;margin-bottom:.6rem">'
                    'Portfolio weights</div>'
                    '<div style="display:flex;gap:22px;align-items:center">'
                    '<div style="flex:none">' + _donut + '</div>'
                    '<div style="flex:1;min-width:0">' + _bar + '</div>'
                    '</div></div>',
                    unsafe_allow_html=True)

                # ── PDF export — same button style as the grid optimiser ──
                try:
                    _mc_fig_png = None
                    try:
                        import matplotlib; matplotlib.use('Agg')
                        import matplotlib.pyplot as _pltmc; import io as _iomc
                        if _okfr:
                            _fx = [r["L"] * 100 for r in _okfr]
                            _fy = [r["er"] * 100 for r in _okfr]
                            _fm, _axm = _pltmc.subplots(figsize=(10, 5.2))
                            _fm.patch.set_facecolor('#0d1117'); _axm.set_facecolor('#0d1117')
                            _axm.tick_params(colors='#c0c8d8', labelsize=8)
                            _axm.spines[:].set_color('#1a3a5c')
                            _axm.set_xlabel('alpha-CVaR floor L (%)', fontsize=9, color='#c0c8d8')
                            _axm.set_ylabel('Max expected return (%)', fontsize=9, color='#c0c8d8')
                            _axm.set_title('Return / Tail-Risk Frontier (Monte-Carlo + CVaR)',
                                           fontsize=10, color='white', pad=8)
                            _axm.grid(True, color='#1a3a5c', linewidth=0.5, alpha=0.5)
                            _axm.plot(_fx, _fy, 'o-', color='#4a9eff', linewidth=1.6,
                                      markersize=6, label='Frontier')
                            _axm.scatter([mc_L * 100], [er * 100], marker='*', color='#f59e0b',
                                         s=240, edgecolors='white', linewidths=0.8, zorder=10,
                                         label='Scalable CVaR optimum')
                            _axm.legend(fontsize=8, facecolor='#0d1a2e', edgecolor='#1a3a5c',
                                        labelcolor='#c0c8d8', loc='upper left')
                            _pltmc.tight_layout(pad=1.2)
                            _bmc = _iomc.BytesIO()
                            _fm.savefig(_bmc, format='png', dpi=150, facecolor='#0d1117',
                                        bbox_inches='tight')
                            _pltmc.close(_fm); _mc_fig_png = _bmc.getvalue()
                    except Exception:
                        _mc_fig_png = None
                    _wmax_ascii = " | per-asset weight bounds" if _has_bounds else ""
                    _mc_meta = {
                        "subtitle": "Monte-Carlo scenarios + alpha-CVaR linear program",
                        "er": f"{er * 100:.2f}%", "es": f"{es * 100:.2f}%",
                        "vol": f"{_sig * 100:.2f}%", "skew": f"{_skew:.3f}",
                        "summary_html": (
                            f"<b>Universe:</b> {_pdf_safe(_univ)} &nbsp;&nbsp; "
                            f"<b>Scenarios:</b> {int(mc_S):,} ({copula} copula){_wmax_ascii}<br/>"
                            f"<b>Objective:</b> maximise E[r] subject to alpha-CVaR &gt;= "
                            f"{mc_L * 100:.0f}% (alpha = {mc_alpha * 100:.0f}%)<br/>"
                            f"<b>Feasibility:</b> "
                            f"{'feasible (floor binding)' if _feas else 'infeasible'}"),
                        "chart_caption": ("Each point is the maximum expected return achievable for "
                                          "one Expected-Shortfall floor; the star marks the chosen optimum."),
                        "footer_caption": ("Approximate scenario-based optimum (CVaR linear program) "
                                           "- complements the exact grid engine. Research & "
                                           "educational project; not investment advice."),
                    }
                    _mc_der_lines = ([f"- <b>{_pdf_safe(d['label'])}</b> - maturity {d['T']:.2f} y, "
                                      f"rate {d['r'] * 100:.1f}%" for d in der_specs]
                                     if der_specs else None)
                    _mc_wr = [(str(_lbl), float(_wi), str(_c))
                              for (_lbl, _wi, _isd, _c) in _rows if float(_wi) > 1e-6][:30]
                    _mc_pdf = generate_mc_pdf_report(_mc_meta, _mc_wr,
                                                     der_lines=_mc_der_lines, fig_png=_mc_fig_png)
                    st.session_state['_mc_pdf_bytes'] = _mc_pdf
                    st.session_state['_mc_pdf_name'] = f"scalable_cvar_optimiser_{mc_L * 100:.0f}pct.pdf"
                except Exception as _mc_pdf_err:
                    st.session_state['_mc_pdf_bytes'] = None
                    st.caption(f"PDF generation failed: {_mc_pdf_err}")

                if st.session_state.get('_mc_pdf_bytes'):
                    st.markdown("---")
                    _mcl, _mcc, _mcr = st.columns([1, 2, 1])
                    with _mcc:
                        st.download_button(
                            label=":material/download: Export & Download PDF Report",
                            data=st.session_state['_mc_pdf_bytes'],
                            file_name=st.session_state.get('_mc_pdf_name', 'scalable_results.pdf'),
                            mime="application/pdf", type="primary",
                            key="mc_pdf_download", use_container_width=True)

            # ---- validation panel ----
            if mc_validate:
                st.markdown("---")
                m_err = float(np.max(np.abs(R_sec.mean(0) - np.array(means))))
                s_err = float(np.max(np.abs(R_sec.std(0) - np.array(sigs))))
                rows_v = [
                    ["Scenario means vs target (max abs err)", f"{m_err:.4f}", "→ 0 as S grows"],
                    ["Scenario vols vs target (max abs err)",  f"{s_err:.4f}", "→ 0 as S grows"],
                ]
                if copula == "gaussian":
                    w_eq = np.full(N, 1.0 / N)
                    mu_p = float(w_eq @ np.array(means))
                    sig_p = float(np.sqrt(w_eq @ cov @ w_eq))
                    es_mc = mc_realised_es(R_sec @ w_eq, mc_alpha)
                    es_an = mc_analytical_es(mu_p, sig_p, mc_alpha)
                    rows_v.append([
                        f"Equal-weight ES: MC {es_mc:.4f} vs Normal {es_an:.4f}",
                        f"|Δ| = {abs(es_mc-es_an):.4f}",
                        "✓ match" if abs(es_mc - es_an) < 0.01 else "raise S"])
                else:
                    rows_v.append(["Closed-form ES check", "n/a",
                                   "Gaussian copula only (t-portfolio is non-Normal)"])
                _vhead = ('<tr>'
                          '<th style="background:rgba(16,185,129,.12);color:#86e0b0;font-weight:bold;padding:6px 10px;text-align:center">Check</th>'
                          '<th style="background:rgba(16,185,129,.12);color:#86e0b0;font-weight:bold;padding:6px 10px;text-align:center">Result</th>'
                          '<th style="background:rgba(16,185,129,.12);color:#86e0b0;font-weight:bold;padding:6px 10px;text-align:center">Note</th></tr>')
                _vrows = ""
                for _c0, _c1, _c2 in rows_v:
                    _vrows += ('<tr style="border-bottom:1px solid #1b2230">'
                               f'<td style="padding:.4rem .5rem">{_c0}</td>'
                               f'<td style="padding:.4rem .5rem;color:#fafafa;font-weight:600;text-align:center">{_c1}</td>'
                               f'<td style="padding:.4rem .5rem;color:#8b949e;text-align:center">{_c2}</td></tr>')
                _vcap = ('<div style="color:#8b949e;font-size:.78rem;margin-top:.6rem;line-height:1.5">'
                         'The scenario sample reproduces the target means and volatilities, and '
                         '(under the Gaussian copula) its Expected Shortfall matches the closed-form '
                         'Normal value — confirming the generator and the tail estimate are faithful. '
                         'The minimum-CVaR portfolio is mean-variance efficient but not the '
                         'global-minimum-variance portfolio (CVaR carries a return tilt). The exact '
                         'grid-agreement test on a 3–4-asset case is documented in the addendum.</div>')
                st.markdown(
                    '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                    'padding:.85rem 1rem">'
                    '<div style="color:#E3C77E;font-weight:600;font-size:16px;margin-bottom:.6rem">'
                    'Validation against closed-form values</div>'
                    '<table style="width:100%;border-collapse:collapse;font-size:.85rem;color:#c9d1d9">'
                    + _vhead + _vrows + '</table>' + _vcap + '</div>',
                    unsafe_allow_html=True)

        except Exception as _e:
            st.error(str(_e))
            import traceback as _tb
            with st.expander("Traceback"):
                st.code(_tb.format_exc())



elif _view == "backtest":
    import datetime as _dt
    with st.container():
        _bb_l, _bb_mid, _bb_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _bb_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _bb_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em">Out-of-Sample Backtest</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "This tab is **self-contained and independent of the Optimiser tab** — it has its "
        "own inputs. It builds the optimal portfolio on a **construction period**, then "
        "holds those fixed weights through a later **evaluation period**, and compares what "
        "the model *expected* with what actually *happened* — for both a no-derivative "
        "portfolio (P1) and a with-derivative portfolio (P2). It also reports the realised "
        "**alpha and beta** (and R\u00b2) of each security and of both portfolios against a "
        "benchmark you choose, with an optional expected-market-return input for a CAPM "
        "(ex-ante) alpha."
    )

    with st.expander("How this backtest works — and why the derivative is marked to market", expanded=False, icon=":material/settings:"):
        st.markdown(
            "**Which engine does this test?** This back-test builds, holds and marks the "
            "portfolios produced by the exact **grid** optimiser — the thesis-validated reference "
            "method. The scalable **Monte-Carlo + CVaR** engine is an approximate complement for "
            "large portfolios; putting *it* through the same walk-forward is a planned extension, "
            "so today's back-test reflects the grid optimiser's output, not the MC engine's.\n\n"
            "**The procedure (walk-forward, single horizon):**\n"
            "1. **Construction period** — download prices, estimate annualised means, "
            "volatilities and correlations, and run the optimiser to get the optimal weights "
            "*with* and *without* the derivative, subject to your mental-account constraint "
            "P(return < H) ≤ α.\n"
            "2. **Evaluation period** — freeze those weights and **buy-and-hold** them (no "
            "rebalancing) over a later, completely separate window.\n"
            "3. Compare **expected vs realised** return, volatility and the loss-threshold "
            "outcome, for P1 and P2.\n\n"
            "**Why the derivative is marked to market mid-life (not just held to maturity):**\n\n"
            "You asked the backtest to measure **risk**, not just return — and risk is a "
            "property of the *path*, not the endpoint. The securities naturally give a return "
            "*series* over the evaluation window (one return per period), from which realised "
            "volatility follows. A held-to-maturity option, by contrast, gives only a *single* "
            "number — its payoff at expiry versus the entry price — which has no volatility and "
            "no distribution.\n\n"
            "To compute the **portfolio's** realised return series, every component must have a "
            "value at every date so they can be summed into one portfolio value each period. "
            "The securities have a price each period; the option therefore also needs a value "
            "each period. **Marking to market** — repricing the option with Black-Scholes at the "
            "current spot, the shrinking remaining maturity, and the volatility assumption — is "
            "exactly what supplies that intermediate value, giving the derivative its own "
            "period-return series.\n\n"
            "This matters *more* for options than for the stocks, because options are "
            "non-linear and time-decaying: their value swings with the underlying (delta/gamma) "
            "and bleeds with time (theta). A held-to-maturity view hides all of that "
            "intra-window behaviour; marking to market is what lets you see whether the realised "
            "return distribution matched the modelled one. And since the mental-account "
            "constraint is itself distributional (a probability of finishing below H), it can "
            "only be checked against a distribution of realised returns.\n\n"
            "In short: held-to-maturity answers *“did it end up where we expected?”*; "
            "mark-to-market answers *“did it **behave** — return **and** risk — the way we "
            "expected along the way?”*\n\n"
            "**One instrument is excluded — Barrier-M.** Marking to market only gives a "
            "comparable expected-vs-realised picture when both sides use the same return "
            "convention. Every instrument here defines its return *net* — (payoff − cost) / cost "
            "— except the Barrier-M note, whose thesis return is *gross* — payoff / cost, with "
            "no −1. If it were dropped in unchanged, its realised (net) value path would land "
            "about 100 points below its gross *expected* figure, purely from the convention "
            "mismatch rather than performance. (It is otherwise European and perfectly markable; "
            "the issue is the gross-vs-net basis, not path dependence.) Reconciling that is a "
            "deliberate next step, so Barrier-M is left out for now."
        )

    with st.expander("Assumptions & limitations", icon=":material/warning:"):
        st.markdown(
            "- **Model mark, not market quotes.** The derivative leg is valued with "
            "Black-Scholes on the *realised underlying price path* — Yahoo Finance gives the "
            "stock path, not the option's traded quotes. This is the standard approach for an "
            "academic out-of-sample test, but the derivative leg is **model-valued**, not "
            "marked against observed option prices.\n"
            "- **Pricing inputs.** The option is priced and marked with the **construction-period "
            "volatility** of the underlying (the model's own assumption), risk-free rate r = 3%, "
            "entry spot normalised to 1.0, strikes as fractions of that spot. Holding the vol "
            "assumption fixed isolates the effect of the realised *price path*.\n"
            "- **Underlying = highest-volatility security**, following the thesis convention "
            "(the derivative is written on the riskiest holding).\n"
            "- **Option life = evaluation window.** The option is entered at the start of the "
            "evaluation window and expires at its end, so the buy-and-hold position is held to "
            "expiry. Remaining maturity shrinks linearly across the window.\n"
            "- **Buy-and-hold, no rebalancing.** Weights are fixed at construction; the value "
            "path lets the weights drift naturally (true buy-and-hold). No transaction costs, "
            "no dividends beyond what auto-adjusted prices already reflect.\n"
            "- **One-year construction horizon.** The weights being tested were optimised for a "
            "*one-year* holding period (μ and σ are annualised), which lines up with an annual "
            "mandate, annual risk budget, or roughly yearly rebalancing. This back-test then holds "
            "those weights over the evaluation window you choose — which may be shorter or longer "
            "than a year — to show how the one-year-optimised portfolio behaves when actually held.\n"
            "- **Single horizon.** The evaluation window is one option horizon, so the "
            "loss-threshold check is a *single realised outcome* versus the modelled probability "
            "α — not a frequency. A multi-window walk-forward (a richer realised P(r<H)) is a "
            "planned extension.\n"
            "- **Instruments.** Vanilla options, straddle, strangle, both collars, the "
            "capital-guaranteed notes (uncapped and capped), and the spreads/certificates — "
            "15 of the 16. Each is marked to market from its Black-Scholes legs and its return "
            "matches the engine's definition exactly (collars normalise by the gross premium "
            "P0+C0; the notes by their replication cost). **Barrier-M is the one exclusion**: "
            "in the thesis its return is defined *gross* (payoff ÷ cost, with no −1), unlike "
            "every other instrument's *net* return, so a mark-to-market value path would not be "
            "comparable to its expected figure without reconciling that convention first.\n"
            "- **Constraint.** You can build under Value-at-Risk (P(r<H) ≤ α), thesis "
            "Expected Shortfall, or Rigorous ES (E[r|r<H] ≥ L). These change the *construction* "
            "weights; the *realised* tail check stays a single-window outcome (a true realised "
            "ES needs the multi-window walk-forward extension).\n"
            "- **Grid resolution.** Fast / Standard / High only. Unlike the Optimiser tab, "
            "**Turbo is not offered here** even under VaR: Turbo is VaR-only and becomes "
            "unreliable once a derivative is in the portfolio — and the backtest always builds "
            "one — so it would silently fall back to the slower solver. (Rigorous ES likewise "
            "runs at a fixed m=51 and is chosen in the Risk measure section, not as a resolution.)"
        )

    st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;margin:1.4rem 0 0;color:#e7ecf4">⚙️ <span style="color:#E3C77E">Optimisation Parameters</span></div><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin:-.15rem 0 1.2rem">▼</div>', unsafe_allow_html=True)

    _BT_RULE = "<hr style='border:none;border-top:1px solid #E3C77E;margin:.6rem 0 1.6rem'>"
    def _bt_head(t, n=None, rule=True):
        badge = ("<span style='display:inline-block;background:#E3C77E;color:#0d1117;border-radius:50%;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;font-size:.82rem;font-weight:700'>" + str(n) + "</span>") if n else ""
        st.markdown("<div style='text-align:center;margin:-.6rem 0 1rem'><span style='display:inline-block;width:290px;box-sizing:border-box;border:1px solid #30363d;background:#0e1521;padding:.28rem 1rem;border-radius:10px;color:#E3C77E;font-weight:600;font-size:.95rem;letter-spacing:.02em;text-align:center'>" + badge + "<span style='display:block;margin-top:.1rem'>" + t + "</span></span></div>", unsafe_allow_html=True)
    def _bt_ai(label, body):
        st.markdown("<div style='display:flex;justify-content:center'><details style='width:290px;box-sizing:border-box;background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;padding:.4rem .7rem;margin:.2rem 0 .8rem;font-size:.78rem'><summary style='cursor:pointer;color:#79b6ff;font-weight:600;list-style:none'>✨ AI-powered: " + label + "</summary><div style='color:#aebccd;margin-top:.4rem;line-height:1.45'>" + body + "</div></details></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #E3C77E;margin:.4rem 0 1.4rem'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _bt_head("Securities & settings", 1, rule=False)
        _bt_ai("What goes here", "Yahoo Finance tickers and the return frequency. Prices over the construction window below feed the optimiser's estimates of means, volatilities and correlations.")
        bt_tickers_raw = st.text_input(
            "Tickers (comma-separated)", value="AAPL, MSFT, JPM", key="bt_tickers",
            help="Yahoo Finance symbols. Pick the option's underlying in the Derivative section below.")
        bt_freq = st.selectbox("Return frequency", ["Daily", "Monthly"], index=0, key="bt_freq")

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
            _bt_head("Construction period", 2, rule=False)
            _bt_ai("What is this window?", "The optimiser estimates the inputs from this window and solves for the optimal weights — with and without the derivative — under your constraint.")
            st.caption("the portfolio is built using data from this period")
            bt_con_start = st.date_input("From", value=_dt.date(2012, 1, 1), key="bt_con_start")
            bt_con_end   = st.date_input("To",   value=_dt.date(2016, 12, 31), key="bt_con_end")
    with c2:
        with st.container(border=True):
            st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
            _bt_head("Evaluation period", 3, rule=False)
            _bt_ai("What is this window?", "Those fixed weights are bought and held — no rebalancing — across this separate, later window; the derivative is marked to market and realised return, risk and the loss-threshold outcome are measured.")
            st.caption("the fixed portfolio is then held and measured over this period")
            bt_eval_start = st.date_input("From", value=_dt.date(2017, 1, 1), key="bt_eval_start")
            bt_eval_end   = st.date_input("To",   value=_dt.date(2017, 12, 31), key="bt_eval_end")

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _bt_head("Walk-forward (optional)", None, rule=False)
        _bt_ai("What is walk-forward?",
               "Instead of a single window, repeat the build-and-hold across several consecutive, "
               "non-overlapping windows that roll forward in time — re-estimating and re-optimising "
               "each time. The construction and evaluation windows keep their current lengths and "
               "step forward together. This turns the single loss-threshold outcome into a "
               "<b>realised P(r&lt;H) across many windows</b> and average realised return/volatility — "
               "the multi-window check the single-window test can only approximate.")
        bt_walk = st.checkbox("Run a rolling multi-window walk-forward (in addition to the single window)",
                              value=False, key="bt_walk")
        bt_n_windows = st.slider("Number of windows", 2, 6, 4, 1, key="bt_nwin",
                                 help="Consecutive evaluation windows, each the same length as the "
                                      "evaluation period above, rolling forward from its start date. "
                                      "Windows that would run past today are skipped.")

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _bt_head("Derivative", 4)
        bt_labels = [lbl for lbl, t in PREDEFINED_DERIVATIVES.items() if t in _BT_SUPPORTED]
        _bt_cur = st.session_state.get("bt_der") or ("Put option" if "Put option" in bt_labels else bt_labels[0])
        _bt_ai("How the derivative is handled",
               "Priced by Black-Scholes and marked to market across the evaluation window "
               "(shrinking maturity), giving the P2 portfolio its own realised return path."
               "<br><br><b>" + _bt_cur + ".</b> " + get_explanation(_bt_cur))
        bt_label = st.selectbox("Instrument", bt_labels, index=bt_labels.index("Put option")
                                if "Put option" in bt_labels else 0, key="bt_der")
        bt_dtype = PREDEFINED_DERIVATIVES[bt_label]

        def _bt_param_inputs(dtype):
            p = {}
            if dtype in ("put", "call", "straddle"):
                default = 0.9 if dtype == "put" else 1.2
                p["strike"] = st.slider("Strike (× entry spot)", 0.50, 1.60, default, 0.05, key="bt_k")
            elif dtype == "strangle":
                p["strike_kp"] = st.slider("Put strike (×)", 0.50, 1.00, 0.85, 0.05, key="bt_kp")
                p["strike_kc"] = st.slider("Call strike (×)", 1.00, 1.60, 1.15, 0.05, key="bt_kc")
            elif dtype in ("safety_collar", "aggressive_collar"):
                p["strike_p"] = st.slider("Put strike (×)", 0.50, 1.50, 1.20, 0.05, key="bt_clp")
                p["strike_c"] = st.slider("Call strike (×)", 1.00, 2.00, 1.60, 0.05, key="bt_clc")
            elif dtype in ("cgn_uncapped", "cgn_capped"):
                p["floor"]         = st.slider("Floor (%)", 0.0, 10.0, 1.0, 0.5, key="bt_cgf") / 100.0
                p["participation"] = st.slider("Participation (%)", 50, 150, 100, 10, key="bt_cgp") / 100.0
                if dtype == "cgn_capped":
                    p["cap"] = st.slider("Cap (%)", 5.0, 50.0, 20.0, 5.0, key="bt_cgc") / 100.0
                p["premium"] = st.slider("Issuer premium", 0.00, 0.10, 0.00, 0.01, key="bt_cgpr")
            elif dtype == "bull_call_spread":
                p["k1"] = st.slider("Long call strike (×)", 0.70, 1.20, 1.00, 0.05, key="bt_b1")
                p["k2"] = st.slider("Short call strike (×, higher)", 1.00, 1.60, 1.20, 0.05, key="bt_b2")
            elif dtype == "bear_put_spread":
                p["k1"] = st.slider("Long put strike (×, higher)", 0.90, 1.30, 1.10, 0.05, key="bt_bp1")
                p["k2"] = st.slider("Short put strike (×, lower)", 0.60, 1.00, 0.90, 0.05, key="bt_bp2")
            elif dtype == "butterfly_call":
                p["center"] = st.slider("Centre strike (×)", 0.80, 1.30, 1.00, 0.05, key="bt_fc")
                p["width"]  = st.slider("Wing width (×)", 0.10, 0.40, 0.20, 0.05, key="bt_fw")
            elif dtype == "condor_call":
                p["center"] = st.slider("Centre (×)", 0.80, 1.30, 1.00, 0.05, key="bt_cc")
                p["w_in"]   = st.slider("Inner half-width (×)", 0.05, 0.30, 0.10, 0.05, key="bt_ci")
                p["w_out"]  = st.slider("Outer half-width (×)", 0.20, 0.60, 0.30, 0.05, key="bt_co")
            elif dtype == "reverse_convertible":
                p["kp"] = st.slider("Short put strike (×)", 0.60, 1.00, 0.90, 0.05, key="bt_rc")
            elif dtype == "discount_certificate":
                p["kc"] = st.slider("Cap strike (×)", 1.00, 1.60, 1.20, 0.05, key="bt_dc")
                p["premium"] = st.slider("Issuer premium", 0.00, 0.10, 0.00, 0.01, key="bt_dcp")
            elif dtype == "outperformance_certificate":
                p["k"] = st.slider("Participation strike (×)", 0.80, 1.30, 1.00, 0.05, key="bt_oc")
                p["premium"] = st.slider("Issuer premium", 0.00, 0.10, 0.00, 0.01, key="bt_ocp")
            return p

        # Payoff diagram (left); underlying + derivative parameters (right), side by side
        _bt_tk = [t.strip().upper() for t in bt_tickers_raw.split(",") if t.strip()]
        _pc = st.columns([1.15, 1])
        with _pc[1]:
            bt_undl_choice = st.selectbox(
                "Underlying security (for the derivative)",
                ["Auto — highest volatility"] + _bt_tk, index=0, key="bt_undl",
                help="Which security the option is written on. Auto picks the highest-volatility "
                     "holding (thesis convention); or pin it to a specific ticker.")
            st.markdown('<div style="font-weight:600;font-size:.9rem;margin:.45rem 0 .3rem">'
                        'Parameters</div>', unsafe_allow_html=True)
            bt_params = _bt_param_inputs(bt_dtype)
        with _pc[0]:
            try:
                _dp = dict(bt_params); _dp["r"] = 0.03; _dp["T"] = 1.0
                _dcfg = build_der_config(bt_dtype, _dp, [0.25], 0)
                if _dcfg:
                    _dcfg["r"] = 0.03; _dcfg["T"] = 1.0
                    _fig_bt = plot_named_payoff(_dcfg, "underlying")
                    st.pyplot(_fig_bt, use_container_width=True)
                    plt.close(_fig_bt)
                    st.caption("Illustrative payoff: derivative return vs underlying return "
                               "(priced at ~25% vol). The backtest prices and marks the option "
                               "with the construction-period volatility of the chosen underlying.")
            except Exception:
                pass

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _bt_head("Risk measure", 5)
        _bt_ai("What does this control?", "The constraint used to <b>build</b> the weights: VaR caps the probability of finishing below H (P(r&lt;H) ≤ α); Expected-Shortfall caps the average loss in the tail (E[r|r&lt;H] ≥ L).")
        bt_method = st.selectbox(
            "Constraint",
            ["Value-at-Risk (VaR)", "Expected Shortfall — thesis", "Rigorous ES — beyond thesis"],
            index=0, key="bt_method",
            help="VaR caps the probability of finishing below H; ES floors the average loss in "
                 "that tail. Rigorous ES enforces the ES floor in the optimisation itself "
                 "(runs at high precision, m=51).")
        bt_ct = {"Value-at-Risk (VaR)": "var",
                 "Expected Shortfall — thesis": "es",
                 "Rigorous ES — beyond thesis": "es_rigorous"}[bt_method]

        bt_H = st.slider("Loss threshold H (horizon return)", -40, 0, -10, 1,
                         format="%d%%", key="bt_H") / 100.0
        if bt_ct == "var":
            bt_alpha = st.slider("Target shortfall probability α  (P(r < H) ≤ α)",
                                 1, 25, 5, 1, format="%d%%", key="bt_alpha") / 100.0
            bt_L = None
        else:
            bt_L = st.slider("Minimum Expected Shortfall L  (E[r | r < H] ≥ L)",
                             -40, 0, -15, 1, format="%d%%", key="bt_L") / 100.0
            bt_alpha = 0.05

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _bt_head("Grid resolution", 6)
        _bt_ai("What does resolution mean?", "A finer state grid gives more accurate (but slower) weights. Fast / Standard / High only — Turbo is disabled here because the backtest always builds a derivative, where Turbo becomes unreliable.")
        bt_res = st.selectbox(
            "Grid precision", ["Fast", "Standard", "High"], index=0, key="bt_res",
            help="Weight-grid precision for the construction optimiser "
                 "(Fast m=21 / Standard m=35 / High m=51). Turbo is omitted here — it is "
                 "unreliable when a derivative is in the portfolio, which the backtest always "
                 "builds — and Rigorous ES is selected in the Risk measure section above.")
        _bt_grid_key = next((k for k in GRID_EXPLANATIONS
                             if bt_res in k and "Turbo" not in k and "Rigorous" not in k), bt_res)
        st.markdown(
            f'<details style="background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;'
            f'padding:.4rem .8rem;margin:.3rem 0;font-size:.82rem">'
            f'<summary style="cursor:pointer;color:#79b6ff;font-weight:600;list-style:none">'
            f'✨ AI-powered: What does this resolution mean?</summary>'
            f'<div style="color:#aebccd;margin-top:.4rem">'
            f'{GRID_EXPLANATIONS.get(_bt_grid_key, "No explanation available.")}</div></details>',
            unsafe_allow_html=True)
        if bt_res == "High":
            st.markdown('<div class="warn-box">⚠️ The heaviest mode — and a backtest re-optimises at every walk-forward window, so on the free hosted demo it is likely to stall. Use Fast here and keep the universe small, or <a href="https://github.com/SamiJeddou/behavioral-portfolio-optimizer/blob/main/Run_Locally_Guide.pdf" target="_blank" rel="noopener" style="color:#fde68a;text-decoration:underline;font-weight:600">run High locally</a>.</div>',
                        unsafe_allow_html=True)
        elif bt_res == "Standard":
            st.markdown('<div class="warn-box">⏱️ Heavy on the free hosted demo — a backtest re-optimises at every window, so this may not finish beyond a few assets. Use Fast for a quick run, or <a href="https://github.com/SamiJeddou/behavioral-portfolio-optimizer/blob/main/Run_Locally_Guide.pdf" target="_blank" rel="noopener" style="color:#fde68a;text-decoration:underline;font-weight:600">run Standard locally</a>.</div>',
                        unsafe_allow_html=True)

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
        _bt_head("Benchmark (for α / β)", 7)
        _bt_ai("What is the benchmark for?", "The market index used to compute realised <b>alpha</b> and <b>beta</b> (and R²) for each security and for both portfolios. Optionally add an expected market return for a CAPM ex-ante alpha.")
        bt_bench_choice = st.selectbox(
            "Benchmark — the \"market\" for alpha & beta",
            ["S&P 500 (^GSPC)", "Global ACWI (ACWI)", "60/40 SPY-AGG blend", "Type my own ticker"],
            index=0, key="bt_bench",
            help="Alpha and beta are measured against this benchmark over the evaluation window. "
                 "For a multi-asset or crypto portfolio, a broad or blended benchmark is more "
                 "meaningful than a single equity index.")
        if bt_bench_choice.startswith("Type"):
            bt_bench_custom = st.text_input("Benchmark ticker (Yahoo symbol)", value="^GSPC",
                                            key="bt_bench_tk")
        else:
            bt_bench_custom = ""
        _bcol1, _bcol2 = st.columns(2)
        with _bcol1:
            bt_rf = st.number_input("Risk-free rate (annual, %)", min_value=0.0, max_value=20.0,
                                    value=3.0, step=0.25, key="bt_rf",
                                    help="Used to form excess returns for the regression.") / 100.0
        with _bcol2:
            bt_erm_raw = st.text_input(
                "Expected market return E[Rₘ] (annual %, optional)", value="", key="bt_erm",
                help="Leave blank to report only realised α / β. Enter a value to add a CAPM "
                     "required-return and an ex-ante (expected) alpha column, using the realised beta.")
        st.markdown(
            f'<details style="background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;'
            f'padding:.4rem .8rem;margin:.3rem 0;font-size:.82rem">'
            f'<summary style="cursor:pointer;color:#79b6ff;font-weight:600;list-style:none">'
            f'✨ AI-powered: What are alpha &amp; beta, and which benchmark?</summary>'
            f'<div style="color:#aebccd;margin-top:.4rem">{BENCHMARK_EXPLANATION}</div></details>',
            unsafe_allow_html=True)

    st.markdown(_BT_RULE, unsafe_allow_html=True)
    st.markdown(
        "<style>.st-key-bt_run button{font-size:1.1rem;font-weight:700;"
        "padding:0.85rem 1rem;border-radius:8px;}</style>", unsafe_allow_html=True)
    _rb = st.columns([1, 2, 1])
    with _rb[1]:
        run_bt = st.button("▶  Run backtest", type="primary", key="bt_run",
                           use_container_width=True)

    if run_bt:
        try:
            tickers = [t.strip().upper() for t in bt_tickers_raw.split(",") if t.strip()]
            if len(tickers) < 2:
                raise RuntimeError("Please enter at least two tickers.")
            if not (bt_con_start < bt_con_end <= bt_eval_start < bt_eval_end):
                raise RuntimeError("Dates must satisfy: construction start < construction end "
                                   "<= evaluation start < evaluation end.")

            res_map = {"Fast": (21, 15), "Standard": (35, 50), "High": (51, 99)}
            m_bt, mp_bt = res_map[bt_res]
            T_years = (bt_eval_end - bt_eval_start).days / 365.25
            factor_eval = 252 if bt_freq == "Daily" else 12

            with st.spinner("Building portfolio on the construction period…"):
                con_px, err = fetch_close_prices(tickers, bt_con_start, bt_con_end)
                if err:
                    raise RuntimeError(f"Construction data: {err}")
                means, sigs, corr, names, _ = stats_from_prices(con_px, bt_freq)
                if len(names) < 2:
                    raise RuntimeError("Fewer than two usable securities after cleaning.")
                cov = corr_to_cov(sigs, corr)
                if bt_undl_choice.startswith("Auto"):
                    u_idx = int(np.argmax(sigs))
                elif bt_undl_choice in names:
                    u_idx = names.index(bt_undl_choice)
                else:
                    u_idx = int(np.argmax(sigs))
                    st.warning(f"{bt_undl_choice} unavailable after cleaning — using "
                               f"highest-volatility security {names[u_idx]} instead.")
                vol_u = float(sigs[u_idx])
                bt_params["vol"] = vol_u
                bt_params["r"] = 0.03
                bt_params["T"] = T_years
                der_cfg = build_der_config(bt_dtype, bt_params, sigs, u_idx)

                nd_res, _ = run_opt(means, sigs, cov, None,    bt_H, bt_alpha, m_bt, mp_bt, bt_ct, L=bt_L)
                dr_res, _ = run_opt(means, sigs, cov, der_cfg, bt_H, bt_alpha, m_bt, mp_bt, bt_ct, L=bt_L)

            with st.spinner("Holding the fixed portfolio over the evaluation period…"):
                ev_px, err = fetch_close_prices(tickers, bt_eval_start, bt_eval_end)
                if err:
                    raise RuntimeError(f"Evaluation data: {err}")
                missing = [nm for nm in names if nm not in ev_px.columns]
                if missing:
                    raise RuntimeError(f"No evaluation-period data for: {missing}")
                ev_px = ev_px[names].ffill().dropna()
                if bt_freq == "Monthly":
                    ev_px = ev_px.resample('ME').last().dropna()
                if len(ev_px) < 3:
                    raise RuntimeError("Insufficient evaluation-period observations.")

                sec_gross = ev_px.values / ev_px.values[0]
                spot_path = ev_px[names[u_idx]].values
                legs, _norm_mode, prem = _bt_legs(bt_dtype, bt_params)
                g_path = mtm_gross_path(legs, _norm_mode, prem, spot_path, T_years, vol_u, 0.03)

                w1 = np.asarray(nd_res["weights"], dtype=float)
                w2 = np.asarray(dr_res["weights"], dtype=float)
                w2_sec, w2_der = w2[:-1], float(w2[-1])
                pv1 = _bt_portfolio_path(sec_gross, w1)
                pv2 = _bt_portfolio_path(sec_gross, w2_sec, der_gross=g_path, w_der=w2_der)

                cum1, ann1, vol1, br1 = _bt_metrics(pv1, factor_eval, bt_H, T_years)
                cum2, ann2, vol2, br2 = _bt_metrics(pv2, factor_eval, bt_H, T_years)

            st.markdown("---")
            st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;margin:0;color:#E3C77E"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:.45rem"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>Optimisation results</div><div style="text-align:center;color:#E3C77E;font-size:.85rem;line-height:1;margin:-.15rem 0 1.7rem">▼</div>', unsafe_allow_html=True)
            _con_txt = (f"P(r < {bt_H:.0%}) ≤ {bt_alpha:.0%}" if bt_ct == "var"
                        else f"E[r | r < {bt_H:.0%}] ≥ {bt_L:.0%}  ({bt_method})")
            _undl_note = "highest volatility" if bt_undl_choice.startswith("Auto") else "selected"
            st.caption(
                f"Underlying for the derivative: **{names[u_idx]}** ({_undl_note}, "
                f"σ = {vol_u:.1%}).  Optimal derivative weight in P2: **{w2_der:.1%}**.  "
                f"Option life T = {T_years:.2f}y.  Constraint: {_con_txt}.")

            _tail_label = "Model P(r<H)" if bt_ct == "var" else "Model E[r|r<H] (tail avg)"
            _rows_bt = [
                ("Expected annual return",     f"{nd_res['expected_return']:.2%}", f"{dr_res['expected_return']:.2%}"),
                ("Realised annual return",     f"{ann1:.2%}",                      f"{ann2:.2%}"),
                ("Expected annual volatility", f"{nd_res['std_dev']:.2%}",         f"{dr_res['std_dev']:.2%}"),
                ("Realised annual volatility", f"{vol1:.2%}",                      f"{vol2:.2%}"),
                (_tail_label,                  f"{nd_res['shortfall_stat']:.2%}",  f"{dr_res['shortfall_stat']:.2%}"),
                (f"Window return (vs H = {bt_H:.0%})",
                                               f"{cum1:.2%}  {(chr(0x26a0)+' below H') if br1 else chr(0x2713)}",
                                               f"{cum2:.2%}  {(chr(0x26a0)+' below H') if br2 else chr(0x2713)}"),
            ]
            _bt_thead = ('<tr style="color:#9fb3d1;border-bottom:1px solid #30363d">'
                         '<td style="text-align:left;padding:.4rem .5rem;font-weight:700">Metric</td>'
                         '<td style="text-align:center;padding:.4rem .5rem;font-weight:700;color:#4a9eff">No derivative (P1)</td>'
                         '<td style="text-align:center;padding:.4rem .5rem;font-weight:700;color:#f5b942">With derivative (P2)</td></tr>')
            _bt_trows = ""
            for _m, _v1, _v2 in _rows_bt:
                _bt_trows += ('<tr style="border-bottom:1px solid #1b2230">'
                              f'<td style="padding:.4rem .5rem;color:#c9d1d9">{_m}</td>'
                              f'<td style="padding:.4rem .5rem;color:#fafafa;font-weight:600;text-align:center">{_v1}</td>'
                              f'<td style="padding:.4rem .5rem;color:#fafafa;font-weight:600;text-align:center">{_v2}</td></tr>')
            _bt_table_html = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                'padding:.85rem 1rem;height:440px;display:flex;flex-direction:column;box-sizing:border-box">'
                '<div style="color:#E3C77E;font-weight:700;font-size:.98rem;margin-bottom:.6rem">'
                'Expected vs realised \u2014 out-of-sample</div>'
                '<div style="flex:1;overflow:auto">'
                '<table style="width:100%;height:100%;border-collapse:collapse;font-size:.85rem;color:#c9d1d9">'
                + _bt_thead + _bt_trows + '</table></div></div>')

            # Benchmark paths (same tickers, built on construction window) \u2014 for chart + table
            _bt_bench_paths = []
            try:
                from core import benchmarks as _bm
                _btb_colors = {"Equal-weight (1/N)": "#34d399", "Minimum-variance": "#9aa7bd",
                               "Max-Sharpe (tangency)": "#c084fc"}
                for _blbl, _bw in _bm.benchmark_set(means, cov, rf=bt_rf):
                    _pvb = _bt_portfolio_path(sec_gross, np.asarray(_bw, float))
                    _bt_bench_paths.append((_blbl, _pvb, _btb_colors.get(_blbl, "#9aa7bd")))
            except Exception:
                _bt_bench_paths = []

            _bt_left, _bt_right = st.columns([1, 1.2])
            with _bt_left:
                st.html(_bt_table_html)
            with _bt_right:
                try:
                    _fig_bt_p = plot_backtest_paths_plotly(
                        ev_px.index, pv1, pv2, f"With derivative (P2) \u2014 {bt_label}")
                    for _blbl, _pvb, _bcol in _bt_bench_paths:
                        _fig_bt_p.add_trace(go.Scatter(
                            x=list(ev_px.index), y=(100.0 * np.asarray(_pvb, float)),
                            mode="lines", name=_blbl,
                            line=dict(color=_bcol, width=1.4, dash="dot"),
                            hovertemplate='<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>Value: %{y:.2f}<extra></extra>'))
                    st.plotly_chart(_fig_bt_p, use_container_width=True,
                                    config={'edits': {'legendPosition': True}, 'displayModeBar': True})
                except Exception as _ce:
                    st.warning(f"Chart unavailable: {_ce}")

            # ── Naive benchmark portfolios (same tickers, built on construction window) ──
            try:
                _bt_bm_rows = [
                    ("Portfolio P1 (no derivative)",   ann1, vol1, cum1, br1, "#4a9eff", True),
                    ("Portfolio P2 (with derivative)", ann2, vol2, cum2, br2, "#f5b942", True),
                ]
                for _blbl, _pvb, _bcol in _bt_bench_paths:
                    _cb, _ab2, _vb, _brb = _bt_metrics(_pvb, factor_eval, bt_H, T_years)
                    _bt_bm_rows.append((f"Benchmark — {_blbl}", _ab2, _vb, _cb, _brb, "#9aa7bd", False))
                _bmtr = ""
                for _lbl, _ar, _vr, _cr, _brk, _clr, _hot in _bt_bm_rows:
                    _fw = "700" if _hot else "500"
                    _wr = f"{_cr:.2%}  " + ("⚠ below H" if _brk else "✓")
                    _bmtr += (f'<tr style="border-bottom:1px solid #1b2230">'
                              f'<td style="padding:.4rem .5rem;color:{_clr};font-weight:{_fw}">{_lbl}</td>'
                              f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_ar:.2%}</td>'
                              f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_vr:.2%}</td>'
                              f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_wr}</td></tr>')
                st.markdown(
                    '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                    'padding:.85rem 1rem;margin-top:1rem">'
                    '<div style="color:#E3C77E;font-weight:700;font-size:.98rem;margin-bottom:.2rem">'
                    'Benchmark comparison — realised out-of-sample</div>'
                    '<div style="color:#8b949e;font-size:.78rem;margin-bottom:.6rem">'
                    'Long-only, fully-invested portfolios of the same tickers, built on the construction '
                    'window and held over the evaluation window (no rebalancing). Equal-weight, '
                    'minimum-variance and max-Sharpe are securities only (no derivative). '
                    'For context, not ranking.</div>'
                    '<table style="width:100%;border-collapse:collapse;font-size:.85rem;color:#c9d1d9">'
                    '<tr style="color:#9fb3d1">'
                    '<th style="text-align:left;padding:.4rem .5rem;font-weight:600">Portfolio</th>'
                    '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Realised annual return</th>'
                    '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Realised annual volatility</th>'
                    f'<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Window return (vs H = {bt_H:.0%})</th>'
                    '</tr>' + _bmtr + '</table></div>', unsafe_allow_html=True)
            except Exception as _btbme:
                st.caption(f"(benchmark comparison unavailable: {_btbme})")

            # ── Alpha / beta of securities and portfolios vs a benchmark ────────
            _bench_label = ""
            try:
                if bt_bench_choice.startswith("Type"):
                    _btk = [t.strip().upper() for t in (bt_bench_custom or "").split(",") if t.strip()]
                    _bench_tickers = _btk[:1] or ["^GSPC"]
                    _bench_label = _bench_tickers[0]
                elif bt_bench_choice.startswith("60/40"):
                    _bench_tickers = ["SPY", "AGG"]
                    _bench_label = "60/40 SPY-AGG"
                else:
                    _bench_tickers = {"S&P 500 (^GSPC)": ["^GSPC"],
                                      "Global ACWI (ACWI)": ["ACWI"]}[bt_bench_choice]
                    _bench_label = bt_bench_choice

                _bpx, _berr = fetch_close_prices(_bench_tickers, bt_eval_start, bt_eval_end)
                if _berr:
                    raise RuntimeError(_berr)
                _bpx = _bpx.reindex(ev_px.index).ffill().bfill()
                if bt_bench_choice.startswith("60/40") and {"SPY", "AGG"}.issubset(_bpx.columns):
                    _bench_ret = (0.6 * _bpx["SPY"].pct_change()
                                  + 0.4 * _bpx["AGG"].pct_change()).values[1:]
                else:
                    _bcol = next((c for c in _bench_tickers if c in _bpx.columns), None)
                    if _bcol is None:
                        raise RuntimeError("benchmark series unavailable after alignment")
                    _bench_ret = _bpx[_bcol].pct_change().values[1:]

                rf_per = bt_rf / factor_eval
                _sec_ret = ev_px[names].pct_change().values[1:]
                _pv1_ret = np.diff(pv1) / pv1[:-1]
                _pv2_ret = np.diff(pv2) / pv2[:-1]
                _Lmin = min(len(_bench_ret), _sec_ret.shape[0], len(_pv1_ret), len(_pv2_ret))
                _bench_ret = _bench_ret[:_Lmin]
                _sec_ret = _sec_ret[:_Lmin]
                _pv1_ret = _pv1_ret[:_Lmin]
                _pv2_ret = _pv2_ret[:_Lmin]
                if _Lmin < 3:
                    raise RuntimeError("too few aligned observations for a regression")

                _erm = None
                _erm_s = (bt_erm_raw or "").strip().replace("%", "")
                if _erm_s:
                    try:
                        _erm = float(_erm_s) / 100.0
                    except ValueError:
                        _erm = None

                def _ab_row(nm, r_series, exp_ret, accent="#c9d1d9", strong=False):
                    _b, _a, _r2 = _capm_alpha_beta(r_series, _bench_ret, rf_per, factor_eval)
                    _fw = "700" if strong else "500"
                    _b_s = f"{_b:.2f}" if np.isfinite(_b) else "—"
                    _a_s = f"{_a:+.2%}" if np.isfinite(_a) else "—"
                    _r2_s = f"{_r2:.2f}" if np.isfinite(_r2) else "—"
                    _c = (f'<td style="padding:.4rem .5rem;color:{accent};font-weight:{_fw}">{nm}</td>'
                          f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_b_s}</td>'
                          f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_a_s}</td>'
                          f'<td style="padding:.4rem .5rem;text-align:center;color:#9fb3d1">{_r2_s}</td>')
                    if _erm is not None:
                        if np.isfinite(_b) and exp_ret is not None:
                            _req = bt_rf + _b * (_erm - bt_rf)
                            _ea = exp_ret - _req
                            _c += (f'<td style="padding:.4rem .5rem;text-align:right;color:#fafafa">{_req:.2%}</td>'
                                   f'<td style="padding:.4rem .5rem;text-align:right;color:#fafafa">{_ea:+.2%}</td>')
                        else:
                            _c += ('<td style="padding:.4rem .5rem;text-align:right;color:#6b7a96">—</td>'
                                   '<td style="padding:.4rem .5rem;text-align:right;color:#6b7a96">—</td>')
                    return f'<tr style="border-bottom:1px solid #1b2230">{_c}</tr>'

                _ab_rows = ""
                for _i, _nm in enumerate(names):
                    _ab_rows += _ab_row(_nm, _sec_ret[:, _i],
                                        float(means[_i]) if _erm is not None else None)
                _ab_rows += _ab_row("Portfolio P1 (no derivative)", _pv1_ret,
                                    nd_res['expected_return'], accent="#4a9eff", strong=True)
                _ab_rows += _ab_row("Portfolio P2 (with derivative)", _pv2_ret,
                                    dr_res['expected_return'], accent="#f5b942", strong=True)

                _ab_hdr = ('<th style="text-align:left;padding:.4rem .5rem;font-weight:600">Holding</th>'
                           '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">β</th>'
                           '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Realised α (ann.)</th>'
                           '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">R²</th>')
                if _erm is not None:
                    _ab_hdr += ('<th style="text-align:right;padding:.4rem .5rem;font-weight:600">CAPM req. (ann.)</th>'
                                '<th style="text-align:right;padding:.4rem .5rem;font-weight:600">Expected α (ann.)</th>')

                _ab_note = (
                    f'Realised over the evaluation window ({_Lmin} {bt_freq.lower()} periods), '
                    f'risk-free {bt_rf:.2%} p.a. β is the regression slope of excess returns, '
                    f'α the annualised Jensen intercept, R² its fit. Portfolio β is measured '
                    f'directly from the realised buy-and-hold path (so the derivative\u2019s '
                    f'non-linearity is captured).')
                if _erm is not None:
                    _ab_note += (f' Expected α = model E[r] − CAPM required return, using your '
                                 f'E[Rₘ] = {_erm:.2%} and the realised β.')

                _ab_html = (
                    '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                    'padding:.85rem 1rem;margin-top:1rem">'
                    '<div style="color:#E3C77E;font-weight:700;font-size:.98rem;margin-bottom:.2rem">'
                    f'Alpha / Beta \u2014 vs {_bench_label}</div>'
                    '<div style="color:#8b949e;font-size:.78rem;margin-bottom:.6rem">' + _ab_note + '</div>'
                    '<table style="width:100%;border-collapse:collapse;font-size:.85rem;color:#c9d1d9">'
                    f'<tr style="color:#9fb3d1">{_ab_hdr}</tr>{_ab_rows}</table></div>')
                st.markdown(_ab_html, unsafe_allow_html=True)

                if _erm is None:
                    st.caption("Tip: enter an expected market return E[Rₘ] in the inputs above to add "
                               "a CAPM required-return and an ex-ante (expected) alpha column.")
            except Exception as _abe:
                st.caption(f"α / β unavailable — {_abe}")

            # Rule-based verdict (no LLM)
            verdict = []
            d_ret = ann2 - ann1
            d_vol = vol2 - vol1
            if d_ret > 0.005:
                verdict.append(f"The derivative **added {d_ret:.2%}** of realised annual return "
                               f"versus the no-derivative portfolio.")
            elif d_ret < -0.005:
                verdict.append(f"The derivative **cost {-d_ret:.2%}** of realised annual return "
                               f"this window.")
            else:
                verdict.append("Realised returns of the two portfolios were broadly similar.")
            if not np.isnan(d_vol):
                if d_vol < -0.005:
                    verdict.append(f"It also **reduced realised volatility** by {-d_vol:.2%}.")
                elif d_vol > 0.005:
                    verdict.append(f"It **raised realised volatility** by {d_vol:.2%}.")
            if br1 and not br2:
                verdict.append(f"P1 breached the {bt_H:.0%} loss threshold while **P2 did not** — "
                               f"the protection held this window.")
            elif br2 and not br1:
                verdict.append(f"P2 breached the {bt_H:.0%} threshold while P1 did not.")
            elif br1 and br2:
                verdict.append(f"Both portfolios finished below the {bt_H:.0%} threshold.")
            else:
                _tgt = (f" (model target: ≤ {bt_alpha:.0%} chance)" if bt_ct == "var" else "")
                verdict.append(f"Neither portfolio finished below the {bt_H:.0%} threshold{_tgt}.")
            for which, exp_r, real_r in [("P1", nd_res['expected_return'], ann1),
                                         ("P2", dr_res['expected_return'], ann2)]:
                gap = real_r - exp_r
                tone = "above" if gap > 0 else "below"
                verdict.append(f"{which} realised {abs(gap):.2%} {tone} its expected annual return.")

            st.markdown("#### Verdict")
            _es_note = ("" if bt_ct == "var" else
                        " A realised Expected Shortfall (a tail average) needs a distribution of "
                        "outcomes, so it isn't estimable from one window — the *construction* "
                        "objective is ES here, but the realised check stays single-draw.")
            st.info("  ".join(verdict) +
                    "\n\n*One evaluation window is a single draw — read the loss-threshold "
                    "outcome as one realisation, not a probability." + _es_note + "*")

            # ── PDF export — same button style as the grid optimiser ──
            try:
                _bt_fig_png = None
                try:
                    import matplotlib; matplotlib.use('Agg')
                    import matplotlib.pyplot as _pltbt; import io as _iobt
                    _fb, _axb = _pltbt.subplots(figsize=(10, 5.0))
                    _fb.patch.set_facecolor('#0d1117'); _axb.set_facecolor('#0d1117')
                    _axb.tick_params(colors='#c0c8d8', labelsize=8)
                    _axb.spines[:].set_color('#1a3a5c')
                    _axb.set_ylabel('Portfolio value (indexed)', fontsize=9, color='#c0c8d8')
                    _axb.set_title('Out-of-sample portfolio paths', fontsize=10,
                                   color='white', pad=8)
                    _axb.grid(True, color='#1a3a5c', linewidth=0.5, alpha=0.5)
                    _bx = list(ev_px.index)
                    _axb.plot(_bx, pv1, color='#10b981', linewidth=1.6, label='No derivative (P1)')
                    _axb.plot(_bx, pv2, color='#f59e0b', linewidth=1.6, label='With derivative (P2)')
                    _axb.legend(fontsize=8, facecolor='#0d1a2e', edgecolor='#1a3a5c',
                                labelcolor='#c0c8d8', loc='upper left')
                    _fb.autofmt_xdate(); _pltbt.tight_layout(pad=1.2)
                    _bbt = _iobt.BytesIO()
                    _fb.savefig(_bbt, format='png', dpi=150, facecolor='#0d1117',
                                bbox_inches='tight')
                    _pltbt.close(_fb); _bt_fig_png = _bbt.getvalue()
                except Exception:
                    _bt_fig_png = None
                _bt_sum = [["Metric", "No derivative (P1)", "With derivative (P2)"]]
                for _m, _v1, _v2 in _rows_bt:
                    _bt_sum.append([_m, _v1, _v2])
                _bt_meta = {
                    "subtitle": (f"Construction {bt_con_start} -> {bt_con_end}  |  "
                                 f"Evaluation {bt_eval_start} -> {bt_eval_end}"),
                    "period_html": (
                        f"<b>Derivative underlying:</b> {names[u_idx]} "
                        f"(sigma {vol_u * 100:.1f}%) &nbsp;&nbsp; "
                        f"<b>Optimal derivative weight in P2:</b> {w2_der * 100:.1f}% &nbsp;&nbsp; "
                        f"<b>Option life:</b> {T_years:.2f} y<br/>"
                        f"<b>Constraint:</b> {_pdf_safe(_con_txt)}"),
                    "chart_caption": ("Fixed weights from the construction window, held through the "
                                      "evaluation window; the derivative is marked to market each step."),
                    "footer_caption": ("Out-of-sample buy-and-hold of the grid optimiser's portfolios. "
                                       "One evaluation window is a single draw. Research & educational "
                                       "project; not investment advice."),
                }
                _bt_verdict = [_md_bold(v) for v in verdict]
                _bt_pdf = generate_backtest_pdf_report(_bt_meta, _bt_sum,
                                                       verdict_lines=_bt_verdict, fig_png=_bt_fig_png)
                st.session_state['_bt_pdf_bytes'] = _bt_pdf
                st.session_state['_bt_pdf_name'] = "backtest_results.pdf"
            except Exception as _bt_pdf_err:
                st.session_state['_bt_pdf_bytes'] = None
                st.caption(f"PDF generation failed: {_bt_pdf_err}")

            if st.session_state.get('_bt_pdf_bytes'):
                st.markdown("---")
                _btl, _btc, _btr = st.columns([1, 2, 1])
                with _btc:
                    st.download_button(
                        label=":material/download: Export & Download PDF Report",
                        data=st.session_state['_bt_pdf_bytes'],
                        file_name=st.session_state.get('_bt_pdf_name', 'backtest_results.pdf'),
                        mime="application/pdf", type="primary",
                        key="bt_pdf_download", use_container_width=True)

            # ── Walk-forward: rolling multi-window out-of-sample ──────────────────
            if bt_walk:
                st.markdown("---")
                st.markdown('<div style="text-align:center;font-size:18px;font-weight:600;'
                            'margin:0;color:#E3C77E">Walk-forward — rolling windows</div>'
                            '<div style="text-align:center;color:#E3C77E;font-size:.85rem;'
                            'line-height:1;margin:-.15rem 0 1.2rem">▼</div>', unsafe_allow_html=True)
                try:
                    from core import benchmarks as _bm

                    def _wf_one_window(_cs, _ce, _es2, _ee):
                        """Build on (_cs,_ce), hold over (_es2,_ee). Returns a dict
                        {'metrics':{strategy:(cum,ann,vol,breach)}, 'paths':{strategy:pv}, 'dates':[...]}
                        or None if the window is unusable."""
                        _cpx, _e1 = fetch_close_prices(tickers, _cs, _ce)
                        if _e1 or _cpx is None or getattr(_cpx, "empty", True):
                            return None
                        _mu, _sg, _cr, _nm, _ = stats_from_prices(_cpx, bt_freq)
                        if len(_nm) < 2:
                            return None
                        _cv = corr_to_cov(_sg, _cr)
                        if bt_undl_choice.startswith("Auto"):
                            _ui = int(np.argmax(_sg))
                        elif bt_undl_choice in _nm:
                            _ui = _nm.index(bt_undl_choice)
                        else:
                            _ui = int(np.argmax(_sg))
                        _Ty = max((_ee - _es2).days / 365.25, 1e-6)
                        _vu = float(_sg[_ui])
                        _p = dict(bt_params); _p["vol"] = _vu; _p["r"] = 0.03; _p["T"] = _Ty
                        _dc = build_der_config(bt_dtype, _p, _sg, _ui)
                        _nd, _ = run_opt(_mu, _sg, _cv, None, bt_H, bt_alpha, m_bt, mp_bt, bt_ct, L=bt_L)
                        _dr, _ = run_opt(_mu, _sg, _cv, _dc, bt_H, bt_alpha, m_bt, mp_bt, bt_ct, L=bt_L)
                        if not _nd or not _dr:
                            return None
                        _epx, _e2 = fetch_close_prices(tickers, _es2, _ee)
                        if _e2 or _epx is None:
                            return None
                        if any(x not in _epx.columns for x in _nm):
                            return None
                        _epx = _epx[_nm].ffill().dropna()
                        if bt_freq == "Monthly":
                            _epx = _epx.resample('ME').last().dropna()
                        if len(_epx) < 3:
                            return None
                        _sg2 = _epx.values / _epx.values[0]
                        _spot = _epx[_nm[_ui]].values
                        _legs, _nmode, _prem = _bt_legs(bt_dtype, _p)
                        _gp = mtm_gross_path(_legs, _nmode, _prem, _spot, _Ty, _vu, 0.03)
                        _factor = 252 if bt_freq == "Daily" else 12
                        _w1 = np.asarray(_nd["weights"], float)
                        _w2 = np.asarray(_dr["weights"], float); _w2s, _w2d = _w2[:-1], float(_w2[-1])
                        _pv1 = _bt_portfolio_path(_sg2, _w1)
                        _pv2 = _bt_portfolio_path(_sg2, _w2s, der_gross=_gp, w_der=_w2d)
                        _out = {}; _paths = {}
                        _out["P1 (no derivative)"] = _bt_metrics(_pv1, _factor, bt_H, _Ty)
                        _paths["P1 (no derivative)"] = _pv1
                        _out["P2 (with derivative)"] = _bt_metrics(_pv2, _factor, bt_H, _Ty)
                        _paths["P2 (with derivative)"] = _pv2
                        for _bl, _bw in _bm.benchmark_set(_mu, _cv, rf=bt_rf):
                            _pvb = _bt_portfolio_path(_sg2, np.asarray(_bw, float))
                            _out["Benchmark — " + _bl] = _bt_metrics(_pvb, _factor, bt_H, _Ty)
                            _paths["Benchmark — " + _bl] = _pvb
                        return {"metrics": _out, "paths": _paths, "dates": list(_epx.index)}

                    _eval_len = max((bt_eval_end - bt_eval_start).days, 1)
                    _con_len = max((bt_con_end - bt_con_start).days, 1)
                    _today = _dt.date.today()
                    _wins = []
                    for _i in range(int(bt_n_windows)):
                        _es2 = bt_eval_start + _dt.timedelta(days=_i * _eval_len)
                        _ee = _es2 + _dt.timedelta(days=_eval_len)
                        if _ee > _today:
                            break
                        _ce = _es2 - _dt.timedelta(days=1)
                        _cs = _ce - _dt.timedelta(days=_con_len)
                        _wins.append((_cs, _ce, _es2, _ee))

                    _per_win = []
                    with st.spinner(f"Running {len(_wins)} walk-forward window(s)…"):
                        for _w in _wins:
                            try:
                                _r = _wf_one_window(*_w)
                            except Exception:
                                _r = None
                            _per_win.append((_w, _r))
                    _ok = [(w, r) for (w, r) in _per_win if r]

                    if not _ok:
                        st.warning("Walk-forward produced no usable windows — the tickers may lack "
                                   "data over the rolled-forward dates. Try fewer windows or an earlier start.")
                    else:
                        _strats = list(_ok[0][1]["metrics"].keys())
                        _accent = {"P1 (no derivative)": "#4a9eff", "P2 (with derivative)": "#f5b942"}
                        _wfcolors = {"P1 (no derivative)": "#4a9eff", "P2 (with derivative)": "#f5b942",
                                     "Benchmark — Equal-weight (1/N)": "#34d399",
                                     "Benchmark — Minimum-variance": "#9aa7bd",
                                     "Benchmark — Max-Sharpe (tangency)": "#c084fc"}
                        _wfc = lambda _s: _wfcolors.get(_s, "#9aa7bd")
                        _wfname = lambda _s: _s.replace("Benchmark — ", "")
                        # Aggregate per strategy
                        _agg_tr = ""
                        for _s in _strats:
                            _arr = np.array([r["metrics"][_s] for (_, r) in _ok], float)  # cols: (cum, ann, vol, breach)
                            _mr = float(np.nanmean(_arr[:, 1])); _mv = float(np.nanmean(_arr[:, 2]))
                            _br = float(np.nanmean(_arr[:, 3]))  # breach rate = realised P(r<H)
                            _clr = _accent.get(_s, "#9aa7bd"); _fw = "700" if _s in _accent else "500"
                            _agg_tr += (f'<tr style="border-bottom:1px solid #1b2230">'
                                        f'<td style="padding:.4rem .5rem;color:{_clr};font-weight:{_fw}">{_s}</td>'
                                        f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_mr:.2%}</td>'
                                        f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_mv:.2%}</td>'
                                        f'<td style="padding:.4rem .5rem;text-align:center;color:#fafafa">{_br:.0%}</td>'
                                        f'<td style="padding:.4rem .5rem;text-align:center;color:#9fb3d1">{len(_ok)}</td></tr>')
                        st.markdown(
                            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                            'padding:.85rem 1rem;margin-top:.4rem">'
                            '<div style="color:#E3C77E;font-weight:700;font-size:.98rem;margin-bottom:.2rem">'
                            'Walk-forward summary — averaged across windows</div>'
                            '<div style="color:#8b949e;font-size:.78rem;margin-bottom:.6rem">'
                            f'{len(_ok)} of {len(_wins)} rolling windows produced usable data. Each window '
                            're-estimates inputs, re-optimises, and holds out-of-sample. <b>Realised P(r&lt;H)</b> '
                            f'is the share of windows finishing below H = {bt_H:.0%} — the multi-window tail '
                            'check (compare with your target α). Benchmarks are securities-only. Not advice.</div>'
                            '<table style="width:100%;border-collapse:collapse;font-size:.85rem;color:#c9d1d9">'
                            '<tr style="color:#9fb3d1">'
                            '<th style="text-align:left;padding:.4rem .5rem;font-weight:600">Strategy</th>'
                            '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Mean realised ann. return</th>'
                            '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Mean realised ann. volatility</th>'
                            f'<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Realised P(r&lt;H)</th>'
                            '<th style="text-align:center;padding:.4rem .5rem;font-weight:600">Windows</th>'
                            '</tr>' + _agg_tr + '</table></div>', unsafe_allow_html=True)

                        # ── Walk-forward equity curve (stitched growth of $1) ──
                        try:
                            import plotly.graph_objects as _gowf
                            _eqfig = _gowf.Figure()
                            for _s in _strats:
                                _xs = []; _ys = []; _run = 1.0
                                for (_w, _r) in _ok:
                                    _p = _r["paths"].get(_s); _d = _r.get("dates")
                                    if _p is None or not _d or len(_p) == 0:
                                        continue
                                    _seg = (_run * np.asarray(_p, float)).tolist()
                                    _xs += list(_d); _ys += _seg
                                    _run = _seg[-1]
                                if _xs:
                                    _strong = _s in _accent
                                    _eqfig.add_trace(_gowf.Scatter(
                                        x=_xs, y=_ys, mode="lines", name=_wfname(_s),
                                        line=dict(color=_wfc(_s), width=2.4 if _strong else 1.6,
                                                  dash=None if _strong else "dot")))
                            _eqfig.add_hline(y=1.0, line=dict(color="#46566f", width=1, dash="dash"))
                            _eqfig.update_layout(
                                template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                height=420, margin=dict(l=10, r=10, t=52, b=40), hovermode="x unified",
                                title=dict(text="Walk-forward equity curve — growth of $1 (stitched out-of-sample)",
                                           x=0.5, font=dict(color="#E3C77E", size=15)),
                                xaxis=dict(title="Date", gridcolor="#1e2130"),
                                yaxis=dict(title="Growth of $1 (×)", gridcolor="#1e2130"),
                                legend=dict(bgcolor="rgba(13,17,23,0.7)", bordercolor="#3a3a5a", borderwidth=1,
                                            font=dict(color="#e7ecf4", size=9), x=0.01, y=0.99,
                                            xanchor="left", yanchor="top"))
                            st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
                            st.plotly_chart(_eqfig, use_container_width=True, config={'displayModeBar': True})
                            st.caption("Each window's out-of-sample path stitched end-to-end into one growth-of-$1 "
                                       "curve. P1/P2 are solid; the securities-only benchmarks are dotted. "
                                       "Re-optimised at the start of every window; no rebalancing within a window.")

                            # ── Per-window grouped bar chart of realised annual return ──
                            _barfig = _gowf.Figure()
                            _wlabels = [f"{_w[2]:%Y-%m}" for (_w, _r) in _ok]
                            for _s in _strats:
                                _vals = [r["metrics"][_s][1] * 100 for (_, r) in _ok]  # ann return %
                                _barfig.add_trace(_gowf.Bar(name=_wfname(_s), x=_wlabels, y=_vals,
                                                            marker_color=_wfc(_s),
                                                            text=[f"{_v:.0f}%" for _v in _vals],
                                                            textposition="outside", textangle=0,
                                                            textfont=dict(size=8, color="#c9d1d9"),
                                                            cliponaxis=False))
                            _barfig.update_layout(
                                template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                height=400, barmode="group", margin=dict(l=10, r=10, t=52, b=40),
                                title=dict(text="Realised annual return by window",
                                           x=0.5, font=dict(color="#E3C77E", size=15)),
                                xaxis=dict(title="Evaluation window (start)", gridcolor="#1e2130"),
                                yaxis=dict(title="Realised annual return (%)", gridcolor="#1e2130"),
                                legend=dict(bgcolor="rgba(13,17,23,0.7)", bordercolor="#3a3a5a", borderwidth=1,
                                            font=dict(color="#e7ecf4", size=9), x=0.01, y=0.99,
                                            xanchor="left", yanchor="top"))
                            st.plotly_chart(_barfig, use_container_width=True, config={'displayModeBar': True})
                            st.caption("Realised annualised return of each strategy in each evaluation window — "
                                       "shows how consistent (or not) each strategy is across regimes.")
                        except Exception as _wfce:
                            st.caption(f"(walk-forward charts unavailable: {_wfce})")

                        # Per-window window returns (P1 vs P2)
                        _pw_tr = ""
                        for (_w, _r) in _ok:
                            _lbl = f"{_w[2]:%Y-%m-%d} → {_w[3]:%Y-%m-%d}"
                            _p1 = _r["metrics"]["P1 (no derivative)"]; _p2 = _r["metrics"]["P2 (with derivative)"]
                            _m1 = f"{_p1[0]:.2%} " + ("⚠" if _p1[3] else "✓")
                            _m2 = f"{_p2[0]:.2%} " + ("⚠" if _p2[3] else "✓")
                            _pw_tr += (f'<tr style="border-bottom:1px solid #1b2230">'
                                       f'<td style="padding:.35rem .5rem;color:#c9d1d9">{_lbl}</td>'
                                       f'<td style="padding:.35rem .5rem;text-align:center;color:#4a9eff">{_m1}</td>'
                                       f'<td style="padding:.35rem .5rem;text-align:center;color:#f5b942">{_m2}</td></tr>')
                        st.markdown(
                            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                            'padding:.85rem 1rem;margin-top:1rem">'
                            '<div style="color:#E3C77E;font-weight:700;font-size:.98rem;margin-bottom:.6rem">'
                            'Per-window outcomes — window return vs H</div>'
                            '<table style="width:100%;border-collapse:collapse;font-size:.85rem;color:#c9d1d9">'
                            '<tr style="color:#9fb3d1">'
                            '<th style="text-align:left;padding:.35rem .5rem;font-weight:600">Evaluation window</th>'
                            '<th style="text-align:center;padding:.35rem .5rem;font-weight:600">P1 window return</th>'
                            '<th style="text-align:center;padding:.35rem .5rem;font-weight:600">P2 window return</th>'
                            '</tr>' + _pw_tr + '</table>'
                            '<div style="color:#6b7f99;font-size:.7rem;margin-top:.4rem">'
                            '✓ finished at/above H · ⚠ finished below H.</div></div>', unsafe_allow_html=True)
                except Exception as _wfe:
                    st.caption(f"(walk-forward unavailable: {_wfe})")

        except Exception as _e:
            st.error(str(_e))
            import traceback as _tb
            with st.expander("Traceback"):
                st.code(_tb.format_exc())



elif _view == "about":
    import os as _os
    with st.container():
        _ab_l, _ab_mid, _ab_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _ab_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _ab_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em">About</h2></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if _os.path.exists("profile.jpeg"):
            import base64 as _b64mod
            with open("profile.jpeg","rb") as _f2:
                _b64_2 = _b64mod.b64encode(_f2.read()).decode()
            st.markdown(
                f'''<a href="https://www.linkedin.com/in/sami-jeddou-25787a404" target="_blank"
                   style="text-decoration:none" title="Connect on LinkedIn">
  <div style="position:relative;display:inline-block;width:160px">
    <img src="data:image/jpeg;base64,{_b64_2}"
         style="width:160px;border-radius:8px;display:block;cursor:pointer"/>
    <div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,91,181,0.75);
                color:#ffffff;font-size:.72rem;text-align:center;padding:4px 0;
                border-radius:0 0 8px 8px"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:3px"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>LinkedIn</div>
  </div>
</a>''',
                unsafe_allow_html=True)
    with col_b:
        st.markdown("## Sami Jeddou")
        st.markdown("**Senior Financial Services Executive — Transformation, Risk & Capital Markets**")
        st.markdown("Risk · Capital Markets · Post-Trade & Clearing · High-Value Payments · Quantitative Finance · Front-to-Back Delivery · Regulatory Programs")
        st.markdown('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>Paris, France', unsafe_allow_html=True)
        st.markdown('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>[LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) &nbsp;|&nbsp; <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>[GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer) &nbsp;|&nbsp; <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 5L2 7"/></svg>sami.jeddou@protonmail.com', unsafe_allow_html=True)

    # ── Reusable helpers for the About page ──────────────────────────────────
    def _about_tbl(headers, rows):
        _hc = "background:#141a23;color:#E3C77E;font-weight:700;text-align:center;padding:.5rem .6rem;border:1px solid #C9A24B"
        _dc = "background:#0d1117;color:#c9d1d9;padding:.45rem .6rem;border:1px solid #30363d;vertical-align:top;text-align:center"
        _h = "".join(f'<td style="{_hc}">{c}</td>' for c in headers)
        _b = "".join("<tr>" + "".join(f'<td style="{_dc}">{c}</td>' for c in r) + "</tr>" for r in rows)
        st.html(f'<table style="width:100%;border-collapse:collapse;font-size:.86rem;margin:.3rem 0 .2rem 0"><tbody><tr>{_h}</tr>{_b}</tbody></table>')

    def _about_fig(path, caption, cols=(1, 1, 1)):
        if _os.path.exists(path):
            _fcs = st.columns(list(cols))
            with _fcs[1]:
                st.image(path, use_container_width=True)
            st.caption(caption)

    _about_toc = [
        "About this app",
        "What it does — in plain terms",
        "Getting your data in — input & cleaning",
        "How it works — the grid optimisation algorithm",
        "Constraint methods & resolutions",
        "Scaling up — Monte-Carlo + CVaR",
        "Out-of-sample back-test",
        "Profiling, tracking & stress-testing your portfolio",
        "The theory — MVT / MAT equivalence",
        "Supported derivatives & structured products",
        "Academic references",
        "About the author",
    ]
    _about_n = [0]
    def _about_head(title):
        _about_n[0] += 1
        _i = _about_n[0]
        st.subheader(f"{_i} · {title}", anchor=f"sec-about-{_i}")

    st.markdown("---")

    # ── Contents (clickable) ─────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<h3 style="color:#E3C77E;margin:.1rem 0 .5rem">Contents</h3>', unsafe_allow_html=True)
        _toc_links = "".join(
            f'<a href="#sec-about-{i+1}" style="display:block;color:#c9d1d9;text-decoration:none;'
            f'padding:.2rem .25rem;border-bottom:1px solid #1b2330">'
            f'<span style="display:inline-block;width:1.6rem;color:#E3C77E;font-weight:700">{i+1}</span>{t}</a>'
            for i, t in enumerate(_about_toc))
        st.markdown(f'<div style="font-size:.92rem;line-height:1.45">{_toc_links}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ═══ 1 · OVERVIEW ═══════════════════════════════════════════════════════
    with st.container(border=True):
        _about_head("About this app")
        st.markdown(
            "**Beyond Mean-Variance Portfolio Optimiser** is an interactive research tool that builds "
            "goal-based portfolios which can include **derivatives and structured products** — something "
            "the classical Markowitz mean-variance framework does not handle.")
        st.markdown(
            "Rather than trading return against variance, it **maximises expected return subject to a "
            "downside rule you set** — either a maximum probability of loss (VaR) or a maximum expected "
            "loss in the tail (Expected Shortfall) — and finds the portfolio, including any hedges, that "
            "best meets that goal.")
        st.markdown(
            "It provides **two optimisers**: a **grid optimiser** — the thesis method, which searches "
            "exhaustively and is exact for small portfolios — and a **scalable Monte-Carlo + CVaR "
            "optimiser** that simulates thousands of scenarios to handle large, institutional-size "
            "portfolios. An **out-of-sample back-test** then holds the chosen portfolio on later, "
            "unseen data and reports its realised performance (alpha, beta and R² versus a benchmark).")
        st.markdown(
            "**Callable beyond the app.** The optimisation and back-testing engines are exposed through a "
            "REST API and an MCP server, so external portfolio, risk and trading systems — or an AI agent — "
            "can call them directly, not just through this interface.")
        st.markdown(
            "It grew out of my MSc Finance thesis (USI Lugano, 2012), which extended the mental-accounting "
            "framework of Das & Statman (2009) and Das, Markowitz, Scheid & Statman (2010). The original "
            "optimiser was written in R; this app is a full Python re-implementation — corrected, validated "
            "against the thesis, and extended with live market data and a wider library of structured "
            "products.")
        _guide_file = "Beyond_Mean_Variance_Portfolio_Optimiser_User_Guide.pdf"
        _guide_url = ("https://raw.githubusercontent.com/SamiJeddou/behavioral-portfolio-optimizer/"
                      "main/Beyond_Mean_Variance_Portfolio_Optimiser_User_Guide.pdf")
        _guide_link_md = ("📄 **[Download the User Guide (PDF)]"
                          f"({_guide_url})** — step-by-step guide to using the app")
        if _os.path.exists(_guide_file):
            try:
                with open(_guide_file, "rb") as _ugf:
                    _guide_bytes = _ugf.read()
                _gcol_l, _gcol_c, _gcol_r = st.columns([3, 2, 3])
                with _gcol_c:
                    st.download_button(
                        "Download the User Guide (PDF)", icon=":material/picture_as_pdf:",
                        data=_guide_bytes, file_name=_guide_file,
                        mime="application/pdf", type="primary",
                        key="guide_dl", use_container_width=True)
                st.caption("A step-by-step guide to using the app.")
                st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            except Exception:
                st.markdown(_guide_link_md, unsafe_allow_html=False)
        else:
            st.markdown(_guide_link_md, unsafe_allow_html=False)
        _examples_file = "Beyond_Mean_Variance_Worked_Examples.pdf"
        _examples_url = ("https://raw.githubusercontent.com/SamiJeddou/behavioral-portfolio-optimizer/"
                         "main/Beyond_Mean_Variance_Worked_Examples.pdf")
        _examples_link_md = ("📄 **[Download the worked examples (PDF)]"
                             f"({_examples_url})** — step-by-step examples with real inputs, outputs and charts")
        if _os.path.exists(_examples_file):
            try:
                with open(_examples_file, "rb") as _exf:
                    _examples_bytes = _exf.read()
                _ecol_l, _ecol_c, _ecol_r = st.columns([3, 2, 3])
                with _ecol_c:
                    st.download_button(
                        "Download the worked examples (PDF)", icon=":material/picture_as_pdf:",
                        data=_examples_bytes, file_name=_examples_file,
                        mime="application/pdf", type="primary",
                        key="examples_dl", use_container_width=True)
                st.caption("Step-by-step worked examples with real inputs, outputs and charts.")
            except Exception:
                st.markdown(_examples_link_md, unsafe_allow_html=False)
        else:
            st.markdown(_examples_link_md, unsafe_allow_html=False)
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        _paper_file = "Beyond_Mean_Variance_Portfolio_Optimiser_Paper.pdf"
        if _os.path.exists(_paper_file):
            try:
                with open(_paper_file, "rb") as _ppf:
                    _paper_bytes = _ppf.read()
                _pcol_l, _pcol_c, _pcol_r = st.columns([3, 2, 3])
                with _pcol_c:
                    st.download_button(
                        "Download the technical paper (PDF)", icon=":material/picture_as_pdf:",
                        data=_paper_bytes, file_name=_paper_file,
                        mime="application/pdf", type="primary",
                        key="paper_dl", use_container_width=True)
                st.caption("A short companion note — the work, the approaches, the mathematical "
                           "framework and the validation.")
            except Exception:
                st.markdown(
                    "📄 **[Download the technical paper (PDF)]"
                    "(https://raw.githubusercontent.com/SamiJeddou/behavioral-portfolio-optimizer/"
                    "main/Beyond_Mean_Variance_Portfolio_Optimiser_Paper.pdf)** — the work, the "
                    "approaches and the mathematical framework", unsafe_allow_html=False)
        else:
            st.markdown(
                "📄 **[Download the technical paper (PDF)]"
                "(https://raw.githubusercontent.com/SamiJeddou/behavioral-portfolio-optimizer/"
                "main/Beyond_Mean_Variance_Portfolio_Optimiser_Paper.pdf)** — the work, the "
                "approaches and the mathematical framework", unsafe_allow_html=False)

    # ═══ 2 · WHAT IT DOES, IN PLAIN TERMS ═══════════════════════════════════
    with st.container(border=True):
        _about_head("What it does — in plain terms")
        st.markdown("**This tool lets you:**")
        st.markdown("""
- **Set a goal, not a risk-aversion number** — tell it how much downside you will accept (e.g. "no more than a 5% chance of losing 10%") and it builds the best portfolio for that goal.
- **Put derivatives inside the optimisation** — options, collars, capital-guaranteed and barrier notes (16 instruments) are priced and optimised *jointly* with your assets, not bolted on afterwards.
- **Use real, live data** — pull 10,000+ tickers (equities, ETFs, crypto) from the market, or enter your own figures.
- **Choose how hard it works** — four precision modes including a Turbo solver ~60× faster, plus a scalable Monte-Carlo + CVaR engine for institutional-size portfolios.
- **Check it out-of-sample** — back-test the chosen portfolio on later data and read its realised alpha, beta and R² versus a benchmark.
- **Size your risk first** — a Grable–Lytton questionnaire profiles your tolerance and sets the constraint (H, α, L) for you.
- **Hold and track a live portfolio** — keep a real book of **securities *and* derivatives**, marked to market, with realised return, risk, drawdown, VaR/CVaR and CAPM alpha/beta; monitor it against your limit (VaR or ES), **stress-test** it (historical, custom-path and parametric), and **save it to your own Google Drive**.
- **Understand every result** — built-in AI explanations and an interactive glossary turn the maths into plain language, and a one-click PDF captures the run.
""")
        st.markdown("**The three building blocks at a glance:**")
        _about_tbl(
            ["Grid optimiser", "Monte-Carlo optimiser", "Back-test"],
            [["Exhaustive grid search over portfolios — exact for small portfolios. Reproduces the thesis method and prices all 16 instruments, under a VaR or Expected-Shortfall limit.",
              "Simulates thousands of joint scenarios and solves a CVaR linear program; cost grows with the number of scenarios, not the number of assets.",
              "Freezes the chosen weights and holds them on later, unseen data, marking the derivative to market along the way."],
             ["<strong>Best for:</strong> small portfolios needing thesis-grade precision; any of the 16 instruments.",
              "<strong>Best for:</strong> large, institutional-size portfolios with many assets and derivatives; α-CVaR tail control.",
              "<strong>Best for:</strong> out-of-sample validation — realised return, alpha, beta and R² versus a benchmark."]])
        st.markdown("*The sections below explain how it works, what you can configure, and the theory behind it.*")

    # ═══ 3 · GETTING YOUR DATA IN ═══════════════════════════════════════════
    with st.container(border=True):
        _about_head("Getting your data in — input & cleaning")
        st.markdown(
            "The grid optimiser supports four data input modes. For live market data and CSV uploads, "
            "returns are automatically cleaned before being passed to the optimizer:")
        st.markdown("""
- **Default**: Das & Statman (2009) base case — 3 securities, pre-calibrated parameters, reproduces thesis results exactly
- **Live market data**: any global ticker from Yahoo Finance, daily or monthly frequency, over a user-defined date range. Auto-adjusted for splits and dividends. Cleaned automatically: stale price rows (zero returns) are removed and outliers beyond ±5 standard deviations are winsorised
- **Manual entry**: enter means, standard deviations, and correlations directly for 2–10 securities
- **CSV upload**: upload historical prices — returns computed automatically with the same cleaning applied as for live data
""")
        st.markdown(
            "The scalable **Monte-Carlo + CVaR** engine takes only the **base case** or **live tickers** — "
            "manual entry and CSV upload apply to the grid optimiser only.")

    # ═══ 4 · THE ENGINES ════════════════════════════════════════════════════
    with st.container(border=True):
        _about_head("How it works — the grid optimisation algorithm")
        st.markdown(
            "The full algorithm is described in Das & Statman (2009) — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*. "
            "The original R implementation is provided in the appendix of the thesis (Jeddou, 2012). "
            "This app is a Python reimplementation of that algorithm, with enhancements and extensions including support for live market data, a custom structured product composer, and an extended optimiser for larger portfolios.")
        st.markdown("""
**Step 1 — State space construction**
A discrete grid of return scenarios is built for all primary securities.
For each scenario, derivative returns are computed analytically using Black-Scholes pricing.

**Step 2 — Probability assignment**
Each state is assigned a probability using a Gaussian copula, correctly capturing the dependence structure between assets.

**Step 3 — Optimization**
For each candidate weight vector, the portfolio return distribution is evaluated against the mental-account constraint. Two constraint types are supported:
- **VaR constraint**: P(return < H) ≤ α — probability of loss beyond H must not exceed α
- **ES constraint**: E[return | return < H] ≥ L — expected loss in the tail must not exceed L

The best eligible portfolio (highest expected return satisfying the constraint) is selected via:
- *≤ 4 securities*: exhaustive grid search over all weight combinations
- *≥ 5 securities*: differential evolution — a global stochastic optimiser that scales to larger portfolios without exhaustive enumeration
""")
        st.markdown(
            "The optimisation is single-period over a **one-year horizon**: because the inputs are "
            "annualised, the weights, the expected return and the VaR/ES limit are all one-year "
            "quantities — the portfolio is assumed to be set today and held for the year. This lines "
            "up cleanly with an annual mandate, annual risk budget, or roughly yearly rebalancing.")

    with st.container(border=True):
        _about_head("Constraint methods & resolutions")
        st.markdown("There are two independent choices — the **constraint method** "
                    "(what downside rule is enforced) and the **resolution / solver** "
                    "(how the optimiser searches). Two routing conditions can override "
                    "the resolution choice: the number of securities, and whether a "
                    "derivative is present.")
        st.markdown("**The three constraint / objective methods**")
        _about_tbl(
            ["Method", "What it optimises", "Best / recommended for"],
            [["<strong>VaR</strong> (Method I)", "max E[r] s.t. P(r &lt; H) ≤ α — a probability-of-shortfall threshold", "The thesis's primary method; most cases"],
             ["<strong>ES — thesis-faithful</strong> (default Method II)", "ES-eligible grid seed, but the COBYLA refinement still targets the <strong>VaR</strong> penalty — faithfully reproduces the original R thesis", "Reproducing the thesis tables exactly"],
             ["<strong>Rigorous ES</strong>", "max E[r] s.t. ES ≥ L, with a genuinely <strong>ES-aware</strong> COBYLA penalty", "Real decision-making — recovers up to ~2.4pp of E[r] the thesis method leaves unused (e.g. L = −15%: 15.5% vs 13.2%)"]])
        with st.expander("The four resolutions / solvers — and where each applies", expanded=False, icon=":material/tune:"):
            _about_tbl(
                ["Resolution", "VaR", "ES (thesis)", "Rigorous ES", "Grid (m / m')", "Speed / reliability", "Best for"],
                [["<strong>Fast</strong>", "✓", "✓", "—", "21 / 15", "fastest; coarse, visible discretisation error", "quick previews"],
                 ["<strong>Standard</strong>", "✓", "✓", "—", "35 / 50", "moderate; safe with derivatives", "daily work, derivative cases"],
                 ["<strong>High precision</strong>", "✓", "✓", "—", "51 / 99", "slow (~15–30 min full frontier); thesis-grade", "publication numbers, validation, derivative cases"],
                 ["<strong>Turbo</strong>", "✓ <em>(n ≤ 4, no-derivative)</em>", "✗", "—", "51, coarse-to-fine", "~seconds (~60× faster than High); <strong>unreliable with a derivative</strong> (up to 32% disagreement)", "fast no-derivative VaR frontier exploration"],
                 ["<strong>Rigorous-ES</strong> (own mode, resolution fixed)", "—", "—", "✓", "51 (fixed)", "~seconds; ES-aware", "ES decision-making"]])
            st.markdown("*Legend: ✓ available · ✗ deliberately disabled · — not applicable (separate fixed-resolution mode).*")
            st.markdown("**Routing rules that override the resolution choice.** "
                        "Fast / Standard / High serve both **VaR** and **thesis-ES**; "
                        "**Turbo** is **VaR-only** and live only for **≤ 4 total securities "
                        "with no derivative** (it is hidden for ES and for 5+ securities); "
                        "**Rigorous-ES** is a separate mode whose resolution is fixed at m = 51. "
                        "The derivative counts toward the security total: **n ≤ 4 → exhaustive "
                        "grid search**, **n ≥ 5 → differential evolution** (stochastic global "
                        "optimiser). Only Turbo's and Rigorous-ES's coarse-to-fine seeding is "
                        "exposed to derivative basin-miss errors; the exhaustive-grid resolutions "
                        "are immune to that and limited only by grid coarseness.")
        _about_fig("about_fig_grid.png", "Grid Optimiser — how the constraint, resolution and engine are chosen (blue = you choose · gold = the tool decides automatically).")

    with st.container(border=True):
        _about_head("Scaling up — Monte-Carlo + CVaR")
        st.markdown(
            "The exact grid above is precise, but its state space grows as *m^n′* and becomes "
            "impractical beyond a handful of assets. A second, **scalable engine** is included for "
            "institutional-size portfolios:")
        st.markdown("""
- **Scenario generation** — joint return and derivative-payoff scenarios are sampled through a copula (Gaussian or Student-t). The Student-t copula captures tail dependence — assets crashing together
- **CVaR linear program** — the goal is solved as a Rockafellar–Uryasev CVaR linear program, so cost grows *linearly* in the number of assets and several derivatives can be optimised at once, even on different underlyings
- **Smooth frontier** — the frontier is swept with common random numbers so points are directly comparable
""")
        st.markdown(
            "This engine uses an **α-CVaR** objective; it is a scalable complement to the exact grid "
            "rather than a bit-for-bit reproduction of it.")

    with st.container(border=True):
        _about_head("Out-of-sample back-test")
        st.markdown(
            "To test the *efficiency* of each optimisation method — not just its in-sample fit — the app "
            "can build portfolio weights on a construction window and then **buy-and-hold** those weights "
            "through a later, out-of-sample window, with any derivative marked to market, comparing "
            "expected against realised outcomes. It also reports the realised **alpha, beta and R²** "
            "of each security and of the portfolio against a benchmark you select (S&P 500, global ACWI, "
            "a 60/40 blend, or any ticker), with an optional expected-market-return input that adds a "
            "CAPM required return and an ex-ante alpha.")
        st.markdown("**Resolution & routing.** "
                    "The back-test reuses the same three constraint methods — VaR, thesis-faithful ES, "
                    "and Rigorous-ES (fixed m = 51). It offers three resolutions — Fast (m = 21, m′ = 15), "
                    "Standard (m = 35, m′ = 50) and High (m = 51, m′ = 99) — but **not Turbo**: "
                    "because the optimiser is re-run at every walk-forward window, Turbo's coarse-to-fine "
                    "seeding is not reliable across the rolling windows, so it is deliberately omitted. "
                    "The same security-count routing applies under the hood "
                    "(**n ≤ 4 → exhaustive grid**, **n ≥ 5 → differential evolution**). "
                    "Re-optimising at every window makes the back-test the heaviest path, so **Fast** is "
                    "recommended on the hosted demo, with Standard / High better run locally.")
        _about_fig("about_fig_backtest.png", "Back-test — same constraint choice, resolutions without Turbo, re-optimised at every walk-forward window (blue = you choose · gold = the tool decides automatically).")

    # ═══ 4b · PROFILING, TRACKING & STRESS ══════════════════════════════════
    with st.container(border=True):
        _about_head("Profiling, tracking & stress-testing your portfolio")
        st.markdown(
            "Beyond constructing and back-testing portfolios, the app helps you **size your risk, hold a "
            "real book, and pressure-test it** — the three tools below complete the picture from *how risk-"
            "tolerant am I?* to *how would my actual holdings behave in a crisis?*")
        st.markdown("**Risk Profile.** A 13-question Grable–Lytton risk-tolerance questionnaire scores you into "
                    "a band and maps that to the *same* simulation parameters the optimisers use — loss "
                    "threshold **H**, shortfall probability **α**, and CVaR floor **L** — with a plain-language "
                    "mental-accounting explanation and a dated history so a re-take never overwrites the old "
                    "result. One click seeds those values into the Grid or Scalable optimiser.")
        st.markdown("**Ticker Analytics.** Type any stock, ETF or index and get its key figures and CFA-style "
                    "ratios, each with a short plain-language explanation.")
        st.markdown("**Live Portfolio.** Build and track a *real* portfolio over time — and, unlike a "
                    "securities-only tracker, it holds **derivatives and structured products too** (puts, "
                    "calls, straddles, strangles, collars, spreads, capital-guaranteed notes). Each position "
                    "is held from its entry date and **marked to market** — derivatives priced with real "
                    "expiry on their underlying via the same engine as the back-test — and the app reports "
                    "realised **return, volatility, Sharpe, max drawdown, daily VaR/CVaR** and **CAPM alpha, "
                    "beta and R²** versus a benchmark. It then layers on:")
        st.markdown("""
- **Risk vs your tolerance** — monitors the book against your limit using **VaR** (breach frequency ≤ α) or **Expected Shortfall** (tail-average ≥ L), shows the implied risk-aversion λ, and offers a **time-varying** mode that judges each period against the limits that were in force then — driven by an editable, dated **tolerance timeline**.
- **Stress testing** — three complementary lenses, each judged against your H limit: **historical scenario replay** (2008, 2011, 2015, 2018, COVID-2020, 2022) applied to your *current* holdings and **projected forward from today**; a **custom shock** (an instantaneous market move via your β, a per-asset shock, or a multi-leg β-driven **path over time**); and a **parametric stress** that scales volatilities and pushes correlations toward 1 (the "diversification breaks in a crash" effect) and re-derives VaR. Drawdown- and horizon-return-vs-limit charts visualise each.
- **Save / load** — sign in with Google to store each portfolio in **your own Drive** (one-click, least-privilege `drive.file` scope — the app only ever touches files it created), or export a portable **JSON** file. Holdings, benchmark, risk settings, your Risk Profile history and tolerance timeline all travel together.
""")

    # ═══ 5 · THE THEORY ═════════════════════════════════════════════════════
    with st.container(border=True):
        _about_head("The theory — MVT / MAT equivalence")
        st.markdown(
            "When no derivatives are present, the mean-variance and behavioral frontiers converge exactly. "
            "For any choice of H and α, there exists a unique implied risk-aversion coefficient λ such that "
            "the mean-variance optimal portfolio and the behavioral optimal portfolio are identical. "
            "For example, at H = -10% and α = 5%, the implied λ = 3.795. "
            "This app computes and displays the implied λ dynamically — simply adjust the H and α sliders "
            "in the sidebar under the Mental-account constraint section to see the corresponding λ update in real time. "
            "Adding derivatives breaks this equivalence and reveals the superiority of the behavioral approach.")
        _about_fig("about_fig_mvt.jpg", "Figure 4.1 — Implied Risk-Aversion Coefficient Surface (Jeddou, 2012). The 3-D surface shows the implied risk-aversion coefficient λ for different combinations of mental-account threshold H and shortfall probability α; each feasible (H, α) pair where the MVT/MAT equivalence holds confirms that the behavioral and mean-variance approaches are mathematically equivalent when no derivatives are present.", cols=(3, 2, 3))

    # ═══ 6 · REFERENCE ══════════════════════════════════════════════════════
    with st.container(border=True):
        _about_head("Supported derivatives & structured products")
        st.markdown("Sixteen instruments are priced by Black-Scholes and optimised jointly with your assets. "
                    "Open the list below for the full set.")
        with st.expander("See all 16 instruments", expanded=False, icon=":material/list:"):
            _about_tbl(
                ["Type", "Description"],
                [["<strong>Put / Call</strong>", "Standard European options"],
                 ["<strong>Safety collar</strong>", "Long put + short call"],
                 ["<strong>Aggressive collar</strong>", "Long call + short put"],
                 ["<strong>Straddle / Strangle</strong>", "Long call + long put (same or different strikes)"],
                 ["<strong>Capital-guaranteed note</strong>", "Uncapped or capped, with floor and participation rate"],
                 ["<strong>Barrier-M note</strong>", "Corridor note with digital components"],
                 ["<strong>Bull call spread</strong>", "Long call + short higher call — bullish, capped, lower cost than a call"],
                 ["<strong>Bear put spread</strong>", "Long put + short lower put — cheaper bearish hedge, capped"],
                 ["<strong>Long butterfly (calls)</strong>", "Long–short²–long calls — low-volatility &#8220;pin&#8221; bet, very cheap"],
                 ["<strong>Call condor</strong>", "Four-strike range bet with a flat maximum payoff between the inner strikes"],
                 ["<strong>Reverse convertible</strong>", "Zero-coupon bond − short put — high coupon, capped upside, principal at risk"],
                 ["<strong>Discount certificate</strong>", "Synthetic underlying − short call — bought at a discount, upside capped"],
                 ["<strong>Outperformance certificate</strong>", "Synthetic underlying + extra call — full downside, geared (&gt;100%) upside"],
                 ["<strong>Custom composer</strong>", "Build any payoff from calls, puts, digitals, and zero-coupon bonds"]])

    with st.container(border=True):
        _about_head("Academic references")
        st.markdown("""
- **Das, Sanjiv and Meir Statman (2009)** — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*
- **Das, Sanjiv, Harry Markowitz, Jonathan Scheid and Meir Statman (2010)** — *Portfolio Optimization with Mental Accounts*, Journal of Financial and Quantitative Analysis, Vol. 45, No. 2, pp. 311–334
- **Jeddou, Sami (2012)** — *Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios*, MSc Finance Thesis, Università della Svizzera italiana (USI Lugano), supervised by Prof. Enrico De Giorgi. Available on [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) and [USI institutional repository](https://thesis.bul.sbu.usi.ch/theses/1012-1112BenJeddou/pdf?1390987439)
- **Rockafellar, R. Tyrrell and Stanislav Uryasev (2000)** — *Optimization of Conditional Value-at-Risk*, Journal of Risk, Vol. 2, No. 3, pp. 21–41 — the CVaR linear-programming result underlying the scalable Monte-Carlo + CVaR engine
""")

    # ═══ 7 · LEGAL ══════════════════════════════════════════════════════════
    st.markdown("""
<div style="background:#0d1a2e;border:1px solid #f59e0b;border-radius:8px;padding:1rem 1.4rem;color:#c0c8d8;font-size:.85rem">

<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg><b style="color:#f59e0b">Legal Disclaimer</b><br><br>
This application is based on the mental accounts portfolio optimisation framework of Das &amp; Statman (2009) and Das, Markowitz, Scheid &amp; Statman (2010), as extended in Jeddou (2012) through additional derivative simulations and parameter analysis. The app further develops this work with live market data connectivity, an expanded derivative library, and an interactive optimisation interface.<br><br>
It is provided for <b>educational and research purposes only</b> and does not constitute financial advice, investment recommendations, or a solicitation to buy or sell any financial instrument. Results are purely illustrative and should not be used as the basis for any investment decision. Past performance and modelled outputs are not indicative of future results.<br><br>
The framework is designed to be extensible — future versions may incorporate additional derivative structures, alternative risk measures, and API connectivity for institutional workflows.

</div>
""", unsafe_allow_html=True)

    # ═══ 8 · AUTHOR & CONTACT ═══════════════════════════════════════════════
    with st.container(border=True):
        _about_head("About the author")
        st.markdown(
            "With over 20 years of experience in financial services transformation, I have delivered "
            "large-scale risk, regulatory, and front-to-back programs at and for tier-1 institutions — "
            "including BNP Paribas CIB, Crédit Agricole, BIL Luxembourg, and TMX Group — across roles "
            "as senior consultant, program director, and independent transformation lead.")
        st.markdown(
            "**Education:** Engineering and finance background — MEng, MSc Project and Program Management, "
            "École des Mines de Saint-Étienne · Master in Finance, USI Lugano · CFA Level I")
        st.markdown("""
**Key achievements:**
- Delivered €2M+ annual cost savings and reduced operational risk across global operations
- Designed and delivered greenfield risk and clearing platforms for CDS and OTC derivatives at leading central counterparties (CCPs)
- Built and led a €25M+ portfolio of concurrent risk and finance transformation initiatives
- Delivered major regulatory programs across multiple jurisdictions (EMIR, Basel IV, FRTB, IRRBB, IFRS 9, MiFID II, ISO 20022)

Open to senior leadership engagements — freelance, contract, or permanent — in France, Europe, or remote/hybrid.
""")

    st.markdown("""
<div style="background:#0f1923;border:1px solid #1a6bbf;border-radius:8px;padding:1rem 1.4rem;color:#ffffff">

<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 5L2 7"/></svg>**Get in touch**

Interested in collaborating, discussing an opportunity, or learning more about this work?
Connect directly, or send me a message below:

<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>[LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) &nbsp;&nbsp;|&nbsp;&nbsp; <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 5L2 7"/></svg>sami.jeddou@protonmail.com

</div>
""", unsafe_allow_html=True)
    with st.form("contact_form"):
        sender_name  = st.text_input("Your name")
        sender_email = st.text_input("Your email")
        message      = st.text_area("Message", height=100,
                                     placeholder="Introduce yourself, share feedback, or tell me about an opportunity...")
        submitted = st.form_submit_button("Send message", type="primary")
        if submitted:
            if sender_name and sender_email and message:
                import requests as _req
                try:
                    resp = _req.post(
                        "https://formspree.io/f/xvzyepoe",
                        data={"name": sender_name, "email": sender_email, "message": message},
                        headers={"Accept": "application/json"},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        st.success("✓ Message sent successfully. I will get back to you shortly.")
                    else:
                        st.error(f"Could not send message (status {resp.status_code}). Please try again or email sami.jeddou@protonmail.com directly.")
                except Exception as ex:
                    st.error(f"Could not send message: {ex}. Please email sami.jeddou@protonmail.com directly.")
            else:
                st.warning("Please fill in all fields before sending.")

elif _view == "glossary":
    with st.container():
        _gl_l, _gl_mid, _gl_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _gl_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _gl_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.5rem"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>AI Glossary &amp; Reference</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "Click any term below for an AI-generated explanation, or type your own question. "
        "Answers are tailored to the context of behavioural portfolio optimisation.")
    st.info(":material/lightbulb: Click any term below for an AI-generated explanation, or type your own question — the answer opens and scrolls into view automatically.", icon=":material/auto_awesome:")

    GLOSSARY_TERMS = {
        "Derivatives & structured products": [
            "Put option", "Call option",
            "Safety collar (long put + short call)",
            "Aggressive collar (long call + short put)",
            "Straddle (long call + long put)",
            "Strangle (long call + long put, diff strikes)",
            "Capital-guaranteed note (CGN)", "Barrier-M note",
            "Bull call spread (long call + short higher call)",
            "Bear put spread (long put + short lower put)",
            "Long butterfly (calls)", "Call condor",
            "Reverse convertible (bond − short put)",
            "Discount certificate (capped underlying)",
            "Outperformance certificate (geared upside)",
            "Digital option", "Zero-coupon bond"
        ],
        "Risk measures": [
            "Value at Risk (VaR)", "Expected Shortfall (ES)",
            "Shortfall probability", "Drawdown", "Horizon-return",
            "Stress testing", "Skewness", "Excess kurtosis"
        ],
        "Performance & benchmarking": [
            "Alpha (Jensen's alpha)", "Beta", "R-squared (R²)", "Benchmark",
            "CAPM (Capital Asset Pricing Model)", "Excess return", "Risk-free rate"
        ],
        "Scalable engine (Monte-Carlo + CVaR)": [
            "α-CVaR (Conditional VaR)", "Monte-Carlo scenario generation",
            "Student-t copula", "Rockafellar–Uryasev CVaR linear program",
            "Common random numbers (CRN)"
        ],
        "Back-test & solvers": [
            "Out-of-sample back-test", "Construction vs evaluation window",
            "Mark-to-market", "Buy-and-hold",
            "Rigorous Expected Shortfall (beyond thesis)",
            "Turbo solver (coarse-to-fine)", "Differential evolution"
        ],
        "Portfolio theory": [
            "Mean-variance efficient frontier", "Markowitz optimization",
            "Mental accounting", "Optimum mental-accounting portfolio", "Behavioral portfolio theory",
            "MVT/MAT equivalence", "Implied risk aversion lambda",
            "Downside-aware utility (mean − κ·CVaR)", "Constrained optimum (downside-feasible region)",
            "Gaussian copula", "Black-Scholes pricing"
        ],
        "Academic references": [
            "Das & Statman (2009) — Beyond Mean-Variance",
            "Das, Markowitz, Scheid & Statman (2010) JFQA",
            "Jeddou (2012) MSc thesis USI Lugano",
            "Rockafellar & Uryasev (2000) — Optimization of CVaR"
        ]
    }

    # Map glossary derivative terms to a payoff config so we can draw an
    # illustrative payoff diagram next to the AI explanation.
    _GLOSSARY_DERIV = {
        "Put option": ("put", {"strike": 0.9}),
        "Call option": ("call", {"strike": 1.2}),
        "Safety collar (long put + short call)": ("safety_collar", {"strike_p": 1.2, "strike_c": 1.6}),
        "Aggressive collar (long call + short put)": ("aggressive_collar", {"strike_p": 1.2, "strike_c": 1.6}),
        "Straddle (long call + long put)": ("straddle", {"strike": 1.0}),
        "Strangle (long call + long put, diff strikes)": ("strangle", {"strike_kp": 0.85, "strike_kc": 1.15}),
        "Capital-guaranteed note (CGN)": ("cgn_capped", {"floor": 0.01, "participation": 1.0, "cap": 0.20, "premium": 0.0}),
        "Barrier-M note": ("barrier_m", {"M": 0.40, "premium_bm": 0.10}),
        "Bull call spread (long call + short higher call)": ("bull_call_spread", {"k1": 1.0, "k2": 1.2}),
        "Bear put spread (long put + short lower put)": ("bear_put_spread", {"k1": 1.1, "k2": 0.9}),
        "Long butterfly (calls)": ("butterfly_call", {"center": 1.0, "width": 0.2}),
        "Call condor": ("condor_call", {"center": 1.0, "w_in": 0.1, "w_out": 0.3}),
        "Reverse convertible (bond − short put)": ("reverse_convertible", {"kp": 0.9}),
        "Discount certificate (capped underlying)": ("discount_certificate", {"kc": 1.2, "premium": 0.0}),
        "Outperformance certificate (geared upside)": ("outperformance_certificate", {"k": 1.0, "premium": 0.0}),
    }

    # At-a-glance facts: (market view, max loss, max gain, payoff formula)
    _GLOSSARY_FACTS = {
        "Put option": ("Bearish / downside hedge", "Premium paid", "Large (strike → 0)", "max(K − S, 0)"),
        "Call option": ("Bullish", "Premium paid", "Unlimited", "max(S − K, 0)"),
        "Safety collar (long put + short call)": ("Protective / hedged", "Limited by put floor", "Capped at call strike", "long put + short call"),
        "Aggressive collar (long call + short put)": ("Bullish / geared", "Large (short put)", "Uncapped above call", "long call + short put"),
        "Straddle (long call + long put)": ("High volatility (direction-neutral)", "Both premiums", "Unlimited", "max(S−K,0) + max(K−S,0)"),
        "Strangle (long call + long put, diff strikes)": ("High volatility (cheaper)", "Both premiums", "Unlimited", "long OTM call + long OTM put"),
        "Capital-guaranteed note (CGN)": ("Cautious bullish", "Protected to floor", "Capped (if capped)", "floor + participation × positive return"),
        "Barrier-M note": ("Range-bound / corridor", "Varies with structure", "Digital coupons in corridor", "corridor digital payoff"),
        "Bull call spread (long call + short higher call)": ("Moderately bullish", "Net premium paid", "Capped (upper − lower strike)", "long lower-strike call − short higher-strike call"),
        "Bear put spread (long put + short lower put)": ("Moderately bearish", "Net premium paid", "Capped (upper − lower strike)", "long higher-strike put − short lower-strike put"),
        "Long butterfly (calls)": ("Low volatility / pin", "Net premium paid", "Max at centre strike", "long − 2·short − long calls"),
        "Call condor": ("Range-bound", "Net premium paid", "Flat between inner strikes", "4-strike call combination"),
        "Reverse convertible (bond − short put)": ("Neutral / mildly bullish", "Principal at risk", "Coupon (capped)", "zero-coupon bond − short put"),
        "Discount certificate (capped underlying)": ("Mildly bullish", "Full underlying downside", "Capped at cap strike", "underlying − short call (cap)"),
        "Outperformance certificate (geared upside)": ("Bullish", "Full underlying downside", "Geared above strike", "underlying return + geared upside above strike"),
    }

    def _glossary_payoff_fig(term):
        info = _GLOSSARY_DERIV.get(term)
        if not info:
            return None
        dtype, params = info
        try:
            p = dict(params); p["r"] = 0.03; p["T"] = 1.0
            cfg = build_der_config(dtype, p, [0.25], 0)
            if not cfg:
                return None
            cfg["r"] = 0.03; cfg["T"] = 1.0
            return plot_named_payoff(cfg, "underlying")
        except Exception:
            return None

    # ── Conceptual diagrams for the non-derivative terms ──────────────────────
    def _gfx_axes(title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors="#9fb3d1", labelsize=8)
        ax.grid(True, color="#1e2130", linewidth=0.6)
        ax.set_title(title, color="#E3C77E", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, color="#c0c8d8", fontsize=9)
        ax.set_ylabel(ylabel, color="#c0c8d8", fontsize=9)
        return fig, ax

    def _fig_var_cvar():
        rng = np.random.RandomState(7)
        r = rng.normal(6, 18, 150000)
        fig, ax = _gfx_axes("Value-at-Risk & CVaR — worst-5% tail", "Portfolio return (%)", "Frequency")
        counts, edges, patches = ax.hist(r, bins=70, color="#4a9eff", alpha=0.7, edgecolor="none")
        var = float(np.percentile(r, 5)); cvar = float(r[r <= var].mean())
        for p, e in zip(patches, edges[:-1]):
            if e <= var:
                p.set_facecolor("#d15866")
        ymax = float(counts.max())
        ax.axvline(var, color="#f5b342", ls="--", lw=1.6)
        ax.axvline(cvar, color="#fb6a78", ls="--", lw=1.6)
        ax.text(var, ymax * 1.02, "VaR (5% quantile)", color="#f5b342", fontsize=8, ha="left")
        ax.text(cvar, ymax * 0.55, "CVaR\n(avg loss in tail)", color="#fb6a78", fontsize=8, ha="right")
        ax.set_yticks([]); fig.tight_layout()
        return fig

    def _fig_alpha_beta():
        rng = np.random.RandomState(3)
        xb = rng.normal(0, 1.2, 110); alpha, beta = 0.5, 0.75
        yb = alpha + beta * xb + rng.normal(0, 0.5, 110)
        fig, ax = _gfx_axes("Alpha & Beta — portfolio vs benchmark",
                            "Benchmark excess return (%)", "Portfolio excess return (%)")
        ax.scatter(xb, yb, s=14, color="#4a9eff", alpha=0.55, edgecolor="none")
        xs = np.linspace(xb.min(), xb.max(), 50)
        ax.plot(xs, alpha + beta * xs, color="#f59e0b", lw=2.2)
        ax.axhline(0, color="#3a3a5a", lw=0.8); ax.axvline(0, color="#3a3a5a", lw=0.8)
        ax.text(xb.max() * 0.45, alpha + beta * xb.max() * 0.45 + 0.4,
                "slope = β = %.2f" % beta, color="#f59e0b", fontsize=8)
        ax.text(0.15, alpha + 1.0, "intercept = α = %.2f%%" % alpha, color="#86e0b0", fontsize=8)
        fig.tight_layout()
        return fig

    def _fig_frontier():
        rng = np.random.RandomState(11)
        vol = np.linspace(4, 42, 120); ret = 3 + 3.4 * np.sqrt(np.clip(vol - 3, 0, None))
        fig, ax = _gfx_axes("Mean-variance efficient frontier",
                            "Risk — volatility (%)", "Expected return (%)")
        sv = rng.uniform(8, 41, 150)
        sr = 3 + 3.4 * np.sqrt(np.clip(sv - 3, 0, None)) - rng.uniform(1.0, 9.0, 150)
        ax.scatter(sv, sr, s=10, color="#4a9eff", alpha=0.35, edgecolor="none", label="feasible portfolios")
        ax.plot(vol, ret, color="#f59e0b", lw=2.4, label="efficient frontier")
        ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c0c8d8", fontsize=8, loc="lower right")
        fig.tight_layout()
        return fig

    def _fig_skew():
        rng = np.random.RandomState(5)
        fig, ax = _gfx_axes("Skewness — asymmetry of returns", "Return", "Density")
        for data, color, lab in [
                (rng.normal(0, 1, 200000), "#9fb3d1", "Symmetric (skew 0)"),
                (rng.gamma(2.0, 1.0, 200000) - 2.0, "#86e0b0", "Positive skew"),
                (-(rng.gamma(2.0, 1.0, 200000) - 2.0), "#d15866", "Negative skew")]:
            h, edges = np.histogram(np.clip(data, -6, 6), bins=120, density=True)
            ax.plot(0.5 * (edges[:-1] + edges[1:]), h, color=color, lw=2, label=lab)
        ax.set_xlim(-6, 6); ax.set_yticks([])
        ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c0c8d8", fontsize=8)
        fig.tight_layout()
        return fig

    def _fig_kurtosis():
        rng = np.random.RandomState(9)
        norm = rng.normal(0, 1, 300000)
        df = 3; t = rng.normal(0, 1, 300000) / np.sqrt(rng.chisquare(df, 300000) / df)
        fig, ax = _gfx_axes("Excess kurtosis — fat tails vs normal", "Return", "Density")
        for data, color, lab in [(norm, "#9fb3d1", "Normal (excess kurtosis 0)"),
                                 (t, "#f59e0b", "Fat-tailed (excess kurtosis > 0)")]:
            h, edges = np.histogram(np.clip(data, -6, 6), bins=160, density=True)
            ax.plot(0.5 * (edges[:-1] + edges[1:]), h, color=color, lw=2, label=lab)
        ax.set_xlim(-6, 6); ax.set_yticks([])
        ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c0c8d8", fontsize=8)
        fig.tight_layout()
        return fig

    def _fig_copula():
        rng = np.random.RandomState(2); n = 4000
        z1 = rng.normal(0, 1, n); z2 = 0.5 * z1 + np.sqrt(1 - 0.25) * rng.normal(0, 1, n)
        fig, ax = _gfx_axes("Joint scenarios & tail dependence (copula)",
                            "Asset A return", "Asset B return")
        qa = float(np.percentile(z1, 8)); qb = float(np.percentile(z2, 8))
        crash = (z1 <= qa) & (z2 <= qb)
        ax.scatter(z1[~crash], z2[~crash], s=4, color="#4a9eff", alpha=0.25, edgecolor="none")
        ax.scatter(z1[crash], z2[crash], s=8, color="#d15866", alpha=0.8, edgecolor="none",
                   label="joint-crash tail")
        ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c0c8d8", fontsize=8, loc="upper left")
        fig.tight_layout()
        return fig

    def _fig_mental_accounts():
        import matplotlib.patches as _mp
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1.16)
        ax.set_title("Mental accounts — the behavioural portfolio pyramid",
                     color="#E3C77E", fontsize=12, fontweight="bold")

        def _edges(y):
            hw = 0.46 * (1 - y)
            return 0.5 - hw, 0.5 + hw
        bands = [
            (0.00, 0.34, "#86e0b0", "Safety layer", "cash, bonds, protective puts — do not breach H"),
            (0.34, 0.68, "#4a9eff", "Income / stability", "balanced, dividend assets — ordinary goals"),
            (0.68, 1.00, "#E3C77E", "Aspiration / upside", "growth, calls, structured notes — reach for gains"),
        ]
        for y0, y1, c, title, desc in bands:
            l0, r0 = _edges(y0); l1, r1 = _edges(y1)
            ax.add_patch(_mp.Polygon([(l0, y0), (r0, y0), (r1, y1), (l1, y1)], closed=True,
                                     facecolor=c, alpha=0.30, edgecolor=c, linewidth=1.6))
            ym = (y0 + y1) / 2.0
            ax.text(0.5, ym + 0.03, title, ha="center", va="center", color=c, fontsize=10, fontweight="bold")
            ax.text(0.5, ym - 0.055, desc, ha="center", va="center", color="#c0c8d8", fontsize=7.0)
        ax.annotate("", xy=(0.04, 1.02), xytext=(0.04, 0.05),
                    arrowprops=dict(arrowstyle="->", color="#9fb3d1", lw=1.2))
        ax.text(0.065, 0.55, "higher goals · more risk", rotation=90, ha="center", va="center",
                color="#9fb3d1", fontsize=7.5)
        fig.tight_layout()
        return fig

    def _fig_black_scholes():
        from math import erf
        K, T, r, sig = 100.0, 1.0, 0.03, 0.25
        S = np.linspace(50, 150, 200)
        Ncdf = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / np.sqrt(2.0))))
        d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        call = S * Ncdf(d1) - K * np.exp(-r * T) * Ncdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
        fig, ax = _gfx_axes("Black-Scholes — call value vs underlying", "Underlying price S", "Call value")
        ax.fill_between(S, intrinsic, call, color="#4a9eff", alpha=0.18, label="time value")
        ax.plot(S, intrinsic, color="#9fb3d1", lw=1.6, ls="--", label="intrinsic value (at expiry)")
        ax.plot(S, call, color="#f59e0b", lw=2.2, label="Black-Scholes value (T = 1y)")
        ax.axvline(K, color="#3a3a5a", lw=0.9, ls=":")
        ax.text(K, ax.get_ylim()[1] * 0.05, " strike K", color="#9fb3d1", fontsize=7.5)
        ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c0c8d8", fontsize=8, loc="upper left")
        fig.tight_layout()
        return fig

    def _gradient_under(_ax, _x, _y, _baseline=0.0):
        import matplotlib.colors as _mcl
        from matplotlib.patches import Polygon as _Poly
        _cmap = _mcl.LinearSegmentedColormap.from_list("rg", ["#f85149", "#e3c77e", "#2ea043"])
        _ylo = min(float(np.min(_y)), _baseline); _yhi = max(float(np.max(_y)), _baseline)
        _g = np.linspace(0, 1, 256).reshape(-1, 1)
        _im = _ax.imshow(_g, extent=[float(_x[0]), float(_x[-1]), _ylo, _yhi], origin="lower",
                         aspect="auto", cmap=_cmap, alpha=0.92, zorder=1)
        _verts = [(float(_x[0]), _baseline)] + list(zip(_x.astype(float), _y)) + [(float(_x[-1]), _baseline)]
        _poly = _Poly(_verts, closed=True, facecolor="none", edgecolor="none")
        _ax.add_patch(_poly); _im.set_clip_path(_poly)

    def _fig_drawdown():
        _r = np.random.RandomState(5)
        _eq = np.cumprod(1 + _r.normal(0.0004, 0.011, 252))
        _dd = (_eq / np.maximum.accumulate(_eq) - 1) * 100
        _x = np.arange(len(_dd))
        fig, ax = _gfx_axes("Drawdown vs limit — loss from the running peak", "Time", "Drawdown (%)")
        _gradient_under(ax, _x, _dd, 0.0)
        ax.plot(_x, _dd, color="#cfd8e3", lw=1.1, zorder=3)
        ax.axhline(-10, color="#f85149", ls="--", lw=1.8, zorder=4)
        ax.text(_x[-1], -10, " loss limit H", color="#f85149", fontsize=8, va="bottom", ha="right")
        ax.set_xticks([]); ax.set_ylim(min(float(_dd.min()) * 1.15, -12), 1.5)
        fig.tight_layout()
        return fig

    def _fig_horizon_return():
        _r = np.random.RandomState(9)
        _y = 7 + 11 * np.sin(np.linspace(0.2, 3.3, 252)) + _r.normal(0, 2.4, 252) - 5.0
        _x = np.arange(len(_y))
        fig, ax = _gfx_axes("Horizon-return vs limit — rolling-window return", "Time", "Horizon return (%)")
        _gradient_under(ax, _x, _y, 0.0)
        ax.plot(_x, _y, color="#cfd8e3", lw=1.1, zorder=3)
        ax.axhline(0, color="#3a4762", lw=0.8, zorder=2)
        ax.axhline(-10, color="#f85149", ls="--", lw=1.8, zorder=4)
        ax.text(_x[-1], -10, " loss limit H", color="#f85149", fontsize=8, va="bottom", ha="right")
        ax.set_xticks([])
        fig.tight_layout()
        return fig

    _GLOSSARY_CONCEPT_FIG = {
        "Value at Risk (VaR)": _fig_var_cvar,
        "Drawdown": _fig_drawdown,
        "Horizon-return": _fig_horizon_return,
        "Expected Shortfall (ES)": _fig_var_cvar,
        "Shortfall probability": _fig_var_cvar,
        "α-CVaR (Conditional VaR)": _fig_var_cvar,
        "Rigorous Expected Shortfall (beyond thesis)": _fig_var_cvar,
        "Rockafellar–Uryasev CVaR linear program": _fig_var_cvar,
        "Alpha (Jensen's alpha)": _fig_alpha_beta,
        "Beta": _fig_alpha_beta,
        "R-squared (R²)": _fig_alpha_beta,
        "CAPM (Capital Asset Pricing Model)": _fig_alpha_beta,
        "Excess return": _fig_alpha_beta,
        "Benchmark": _fig_alpha_beta,
        "Mean-variance efficient frontier": _fig_frontier,
        "Markowitz optimization": _fig_frontier,
        "Skewness": _fig_skew,
        "Excess kurtosis": _fig_kurtosis,
        "Monte-Carlo scenario generation": _fig_copula,
        "Student-t copula": _fig_copula,
        "Gaussian copula": _fig_copula,
        "Mental accounting": _fig_mental_accounts,
        "Behavioral portfolio theory": _fig_mental_accounts,
        "Black-Scholes pricing": _fig_black_scholes,
    }

    _GLOSSARY_CONCEPT_FACTS = {
        "Value at Risk (VaR)": [("Measures", "Loss not exceeded at conf. 1−α"), ("Captures", "A quantile / threshold"), ("Blind spot", "Severity beyond the quantile")],
        "Expected Shortfall (ES)": [("Measures", "Avg loss beyond threshold H"), ("vs VaR", "Tail-severity aware"), ("Coherent", "Yes")],
        "α-CVaR (Conditional VaR)": [("Measures", "Avg loss in worst α%"), ("vs VaR", "Tail depth, not just frequency"), ("Optimisation", "Convex — linear program"), ("Engine", "Scalable Monte-Carlo")],
        "Shortfall probability": [("Measures", "Probability return falls below H"), ("Used in", "VaR-style constraint"), ("Ignores", "How bad the breach is")],
        "Rigorous Expected Shortfall (beyond thesis)": [("Enforces", "ES floor in the optimiser itself"), ("vs thesis", "ES-aware penalty"), ("Benefit", "Recovers unused expected return")],
        "Rockafellar–Uryasev CVaR linear program": [("Result", "CVaR minimisation is convex"), ("Form", "Linear program over scenarios"), ("Why", "Powers the scalable engine")],
        "Alpha (Jensen's alpha)": [("Meaning", "Return above CAPM-expected"), ("From", "Regression intercept"), ("Sign", "plus = outperformance")],
        "Beta": [("Meaning", "Sensitivity to benchmark"), ("β = 1", "Moves with market"), ("β under 1", "Defensive"), ("β over 1", "Aggressive")],
        "R-squared (R²)": [("Meaning", "Variance explained by benchmark"), ("Range", "0 to 1"), ("High R²", "Return tracked by market")],
        "CAPM (Capital Asset Pricing Model)": [("Formula", "rf + β·(market − rf)"), ("Gives", "Required return for risk"), ("Alpha", "Excess over this line")],
        "Excess return": [("Meaning", "Return minus risk-free rate"), ("Used in", "Alpha / beta regression")],
        "Benchmark": [("Role", "Reference index for α / β"), ("Examples", "S&P 500, ACWI, 60/40")],
        "Mean-variance efficient frontier": [("Trade-off", "Max return per unit variance"), ("Author", "Markowitz (1952)"), ("Limit", "Assumes normal returns")],
        "Markowitz optimization": [("Objective", "Mean-variance trade-off"), ("Inputs", "Means, vols, correlations"), ("Blind spot", "Skew and fat tails")],
        "Skewness": [("Measures", "Asymmetry of returns"), ("Positive", "Big gains, small losses"), ("Negative", "Big losses"), ("Added by", "Calls, CGNs")],
        "Excess kurtosis": [("Measures", "Fat-tailedness vs normal"), ("Positive", "Extreme moves more likely"), ("Normal", "Excess = 0")],
        "Monte-Carlo scenario generation": [("Method", "Sample joint return scenarios"), ("Scales with", "Scenario count, not asset count"), ("Enables", "Large multi-derivative portfolios")],
        "Student-t copula": [("Captures", "Joint-crash tail dependence"), ("vs Gaussian", "Fatter joint tails"), ("Param", "Degrees of freedom")],
        "Gaussian copula": [("Captures", "Linear correlation"), ("Joint tails", "Thin (no tail dependence)"), ("Use", "Baseline dependence model")],
        "Mental accounting": [("Coined by", "Richard Thaler"), ("Idea", "Wealth split into goal-based accounts"), ("In this app", "Each account = threshold H + shortfall prob α"), ("Enables", "Targeted protection via derivatives")],
        "Optimum mental-accounting portfolio": [("Maximises", "Expected return"), ("Subject to", "Downside limit (H, α or CVaR floor L)"), ("On the chart", "Top edge of the feasible (coloured) region"), ("Grey zone", "Infeasible — breaches the limit"), ("With derivatives", "Higher return at the same limit")],
        "Behavioral portfolio theory": [("Authors", "Shefrin & Statman (2000)"), ("Structure", "Layered pyramid of goal portfolios"), ("vs Markowitz", "Goal-based, not one mean-variance mix")],
        "Black-Scholes pricing": [("Prices", "European options"), ("Inputs", "Spot, strike, vol, rate, maturity"), ("Output", "Fair value = intrinsic + time value"), ("Assumes", "Lognormal prices, constant vol")],
    }

    # Glossary terms illustrated by an image file (e.g. figures from the thesis)
    _GLOSSARY_IMAGE = {
        "MVT/MAT equivalence": "risk aversion curve.png",
        "Implied risk aversion lambda": "risk aversion curve.png",
        "Gaussian copula": "gaussian copula.png",
        "Optimum mental-accounting portfolio": "home_optimiser_grid3d.png",
    }

    def _glossary_fig(term):
        if term in _GLOSSARY_DERIV:
            return _glossary_payoff_fig(term)
        fn = _GLOSSARY_CONCEPT_FIG.get(term)
        if fn:
            try:
                return fn()
            except Exception:
                return None
        return None

    if "glossary_response" not in st.session_state:
        st.session_state["glossary_response"] = ""
    if "glossary_term" not in st.session_state:
        st.session_state["glossary_term"] = ""
    if "glossary_notice" not in st.session_state:
        st.session_state["glossary_notice"] = ""

    for category, terms in GLOSSARY_TERMS.items():
        st.markdown(
            f'<h4 style="color:#E3C77E;margin:0.4rem 0 0.25rem">{category.replace("&", "&amp;")}</h4>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for i, term in enumerate(terms):
            if cols[i % 3].button(term, key=f"gloss_{term}", use_container_width=True):
                st.session_state["glossary_term"] = term
                st.session_state["glossary_notice"] = f"Showing explanation for {term} — see below."
                with st.spinner(f"Looking up: {term}..."):
                    st.session_state["glossary_response"] = get_explanation(term)
        st.markdown("")

    st.markdown("---")
    st.markdown("### Ask your own question")
    custom_q = st.text_input(
        "Type a term or question",
        placeholder="e.g. What is the difference between VaR and ES?")
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    if st.button("ASK AI", type="primary", icon=":material/smart_toy:", key="gloss_ask"):
        if custom_q.strip():
            st.session_state["glossary_term"] = custom_q
            st.session_state["glossary_notice"] = ""
            with st.spinner("Thinking..."):
                _gresp = get_ai_chat_response(
                    custom_q,
                    portfolio_context=f"Portfolio has {len(means_in)} securities with means {[f'{m*100:.1f}%' for m in means_in]}")
            if _gresp:
                st.session_state["glossary_response"] = _gresp
            else:
                st.session_state["glossary_response"] = ""
                st.warning("AI answers aren't available right now (the API key isn't configured in this "
                           "environment). Click any term above for its built-in explanation.")
        else:
            st.warning("Please enter a question first.")

    if st.session_state["glossary_response"]:
        st.markdown("---")
        _gterm = st.session_state['glossary_term']
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 .7rem">'
            '<span style="font-family:Georgia,serif;font-size:1.4rem;font-weight:600;color:#E3C77E">'
            + _gterm +
            '</span><span style="font-size:.58rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
            'padding:3px 9px;border-radius:999px;color:#1a1205;background:linear-gradient(135deg,#E3C77E,#caa14a)">'
            '✨ AI explained</span></div>',
            unsafe_allow_html=True)
        import streamlit.components.v1 as _components
        st.session_state["_gloss_scroll_n"] = st.session_state.get("_gloss_scroll_n", 0) + 1
        _components.html(
            "<script>/* " + str(st.session_state["_gloss_scroll_n"]) + " */"
            "setTimeout(function(){var w=window.parent,d=w.document;"
            "var ns=d.querySelectorAll('span'),s=null,i;"
            "for(i=0;i<ns.length;i++){if(ns[i].children.length===0 && ns[i].textContent.indexOf('AI explained')>-1){s=ns[i];break;}}"
            "if(!s)return;"
            "var a=s.parentElement||s;"
            "var c=a;"
            "while(c){var o=w.getComputedStyle(c).overflowY;if((o==='auto'||o==='scroll')&&c.scrollHeight>c.clientHeight)break;c=c.parentElement;}"
            "if(!c)return;"
            "var er=a.getBoundingClientRect(),cr=c.getBoundingClientRect();"
            "var t=c.scrollTop+(er.top-cr.top)-225;"
            "try{c.scrollTo({top:t,behavior:'smooth'});}catch(e){c.scrollTop=t;}"
            "setTimeout(function(){if(Math.abs(c.scrollTop-t)>4)c.scrollTop=t;},450);"
            "},350);</script>",
            height=0)
        if _gterm in _GLOSSARY_FACTS:
            _pairs = list(zip(["Market view", "Max loss", "Max gain", "Payoff"], _GLOSSARY_FACTS[_gterm]))
        else:
            _pairs = _GLOSSARY_CONCEPT_FACTS.get(_gterm)
        if _pairs:
            _chip = ("background:#0e1521;border:1px solid #30363d;border-radius:999px;"
                     "padding:.28rem .85rem;font-size:.78rem;color:#9fb3d1")
            _chips = "".join(
                f'<span style="{_chip}">{lab}: <b style="color:#E3C77E">{val}</b></span>'
                for lab, val in _pairs)
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin:0 0 1rem">{_chips}</div>',
                unsafe_allow_html=True)
        _notation = (
            '<div style="margin-top:.75rem;padding-top:.55rem;border-top:1px solid #1b2230;'
            'color:#8b949e;font-size:.78rem"><b style="color:#9fb3d1">Notation:</b> '
            'S = price of the underlying at maturity · K = strike price '
            '(as a multiple of the entry spot)</div>'
        ) if _gterm in _GLOSSARY_DERIV else ''
        _expl_html = (
            '<div style="background:#0f1923;border:1px solid #30363d;border-radius:8px;'
            'padding:1rem 1.2rem;color:#c0c8d8;font-size:.92rem;line-height:1.65">'
            + st.session_state["glossary_response"] + _notation + '</div>')
        _img_path = None
        if _gterm in _GLOSSARY_IMAGE:
            _p = _os.path.join(_HOME_DIR, _GLOSSARY_IMAGE[_gterm])
            if _os.path.exists(_p):
                _img_path = _p
        _gfig = None if _img_path else _glossary_fig(_gterm)
        if _img_path is not None:
            _gcol1, _gcol2 = st.columns([1.05, 1], vertical_alignment="center")
            with _gcol1:
                st.image(_img_path, width=405)
                st.caption(
                    "The Grid optimiser's 3D objective landscape — grey = infeasible (breaches the "
                    "downside limit), coloured = feasible; the optimum sits at the top edge."
                    if _gterm == "Optimum mental-accounting portfolio" else
                    "Figure from the author's MSc thesis (Jeddou, 2012, USI Lugano).")
            with _gcol2:
                st.markdown(_expl_html, unsafe_allow_html=True)
        elif _gfig is not None:
            _is_deriv = _gterm in _GLOSSARY_DERIV
            _gcol1, _gcol2 = st.columns([1.05, 1], vertical_alignment="center")
            with _gcol1:
                st.pyplot(_gfig, use_container_width=True)
                plt.close(_gfig)
                st.caption("Illustrative payoff — derivative return vs underlying return "
                           "(priced at ~25% vol, 1-year maturity)." if _is_deriv else
                           "Illustrative concept diagram — schematic, not computed from your portfolio.")
            with _gcol2:
                st.markdown(_expl_html, unsafe_allow_html=True)
        else:
            st.markdown(_expl_html, unsafe_allow_html=True)
        if st.button("CLEAR RESPONSE", type="primary", key="gloss_clear"):
            st.session_state["glossary_response"] = ""
            st.session_state["glossary_term"] = ""
            st.session_state["glossary_notice"] = ""
            st.rerun()

elif _view == "portfolio":
    from core import portfolio as _pf
    from core import stress as _ss
    from core.pricing import live_derivative_series as _lds, LIVE_DER_TYPES as _LDT
    import plotly.graph_objects as _go
    import pandas as _pd
    import datetime as _dt
    import json as _json

    with st.container():
        _pf_l, _pf_mid, _pf_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _pf_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _pf_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E3C77E" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:.5rem"><path d="M3 3v18h18"/><path d="M7 14l3.5-3.5 2.5 2.5L18 8"/></svg>Live Portfolio</h2></div>', unsafe_allow_html=True)
    st.caption("Build a portfolio, then track its performance against a benchmark — return since inception, "
               "risk, realised alpha &amp; beta, and whether it stays within your risk tolerance. Use one date for "
               "the initial mix; add rows with a later date to record a rebalance. (One-click sign-in to save to "
               "your Google Drive — coming next.)")

    _BENCH_PRESETS = {"S&P 500 — SPY": "SPY", "MSCI ACWI (world) — ACWI": "ACWI",
                      "Nasdaq-100 — QQQ": "QQQ", "Russell 2000 — IWM": "IWM",
                      "Euro Stoxx 50 — EZU": "EZU", "US bonds — AGG": "AGG",
                      "Gold — GLD": "GLD", "Custom ticker…": "__custom__"}
    st.session_state.setdefault('pf_name_inp', 'My portfolio')
    st.session_state.setdefault('pf_bench_custom', 'SPY')
    st.session_state.setdefault('pf_capital_inp', 10000.0)
    _GTITLE = "<span style='font-size:20px;font-weight:600;color:#E3C77E'>%s</span>"
    _pc1, _pc2, _pc3 = st.columns([2, 1, 1])
    with _pc1:
        st.markdown(_GTITLE % "Portfolio name", unsafe_allow_html=True)
        pf_name = st.text_input("Portfolio name", key='pf_name_inp', label_visibility="collapsed")
    with _pc2:
        st.markdown(_GTITLE % "Benchmark", unsafe_allow_html=True)
        _bsel = st.selectbox("Benchmark", list(_BENCH_PRESETS.keys()), key='pf_bench_sel',
                             label_visibility="collapsed")
        if _BENCH_PRESETS[_bsel] == "__custom__":
            pf_bench = (st.text_input("Custom benchmark ticker", key='pf_bench_custom') or "").strip().upper()
        else:
            pf_bench = _BENCH_PRESETS[_bsel]
    with _pc3:
        st.markdown(_GTITLE % "Starting capital (€)", unsafe_allow_html=True)
        pf_capital = st.number_input("Starting capital", key='pf_capital_inp', min_value=0.0,
                                     step=1000.0, format="%.0f", label_visibility="collapsed",
                                     help="Amount invested at inception. Scales the growth-of-€1 "
                                          "curve into a euro portfolio value (right axis of the "
                                          "Total-return chart).")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(_GTITLE % "Holdings", unsafe_allow_html=True)
    st.caption("Each row is a position you hold, kept from its **Entry date** onward — positions "
               "**accumulate** (a later row adds to the book, it doesn't replace earlier ones). "
               "Securities and derivatives together must **total 100%**. The earliest entry date is "
               "the portfolio's inception; a position added later sits in cash until its entry date.")
    if 'pf_table_data' not in st.session_state:
        _inc = _dt.date.today().replace(year=_dt.date.today().year - 1)
        st.session_state['pf_table_data'] = _pd.DataFrame([
            {"Date": _inc, "Ticker": "AAPL", "Weight %": 40.0},
            {"Date": _inc, "Ticker": "MSFT", "Weight %": 35.0},
            {"Date": _inc, "Ticker": "GLD",  "Weight %": 25.0},
        ])
    pf_table = st.data_editor(
        st.session_state['pf_table_data'], num_rows="dynamic", hide_index=True,
        use_container_width=True, key="pf_table_ed",
        column_config={
            "Date": st.column_config.DateColumn("Entry date", format="YYYY-MM-DD",
                help="When you started holding this position. The earliest entry date is the "
                     "portfolio's inception; the position is held from this date onward."),
            "Ticker": st.column_config.TextColumn("Ticker", help="e.g. AAPL, MSFT, GLD, MC.PA, BTC-USD"),
            "Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0,
                                                      step=1.0, format="%.1f"),
        })

    # ── Derivatives & structured products (optional) ──
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(_GTITLE % ("Derivatives &amp; structured products "
                "<span style='font-size:.8rem;color:#8b97a8;font-weight:400'>(optional)</span>"),
                unsafe_allow_html=True)
    st.caption("Each position is marked to market on the underlying you name — same pricing engine as the "
               "Backtest, with real expiry. Strikes are fractions of the entry spot (0.90 = 90%); Tenor is the "
               "option life in years. These weights count toward the same 100% total as your securities.")
    def _empty_der_df():
        return _pd.DataFrame({
            "Date": _pd.Series([], dtype="datetime64[ns]"),
            "Underlying": _pd.Series([], dtype="object"),
            "Type": _pd.Series([], dtype="object"),
            "Strike 1": _pd.Series([], dtype="float64"),
            "Strike 2": _pd.Series([], dtype="float64"),
            "Tenor (y)": _pd.Series([], dtype="float64"),
            "Weight %": _pd.Series([], dtype="float64")})
    if 'pf_der_data' not in st.session_state:
        st.session_state['pf_der_data'] = _empty_der_df()
    pf_der = st.data_editor(
        st.session_state['pf_der_data'], num_rows="dynamic", hide_index=True,
        use_container_width=True, key="pf_der_ed",
        column_config={
            "Date": st.column_config.DateColumn("Entry date", format="YYYY-MM-DD",
                help="When you entered the position (the option's contract start)."),
            "Underlying": st.column_config.TextColumn("Underlying", help="Ticker the derivative is written on, e.g. AAPL, SPY."),
            "Type": st.column_config.SelectboxColumn("Type", options=list(_LDT.keys()), width="medium"),
            "Strike 1": st.column_config.NumberColumn("Strike 1", min_value=0.0, max_value=3.0, step=0.05,
                format="%.2f", help="Fraction of entry spot (0.90 = 90%). Main strike / collar put / spread long / CGN floor."),
            "Strike 2": st.column_config.NumberColumn("Strike 2", min_value=0.0, max_value=3.0, step=0.05,
                format="%.2f", help="Second strike where the type needs one (collar call, strangle call, spread short)."),
            "Tenor (y)": st.column_config.NumberColumn("Tenor (y)", min_value=0.1, max_value=10.0, step=0.25,
                format="%.2f", help="Option life in years."),
            "Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        })

    # ── Live weight check: all positions (securities + derivatives) must total 100% ──
    _wtot = 0.0; _der_inc = False; _any_pos = False
    try:
        _tblw = pf_table.dropna(how="all") if hasattr(pf_table, "dropna") else pf_table
        for _, _rw in _tblw.iterrows():
            _tw = _rw.get("Ticker"); _ww = _rw.get("Weight %")
            if not isinstance(_tw, str) or not _tw.strip():
                continue
            if not _pd.isna(_ww):
                _wtot += float(_ww); _any_pos = True
        _derw = pf_der.dropna(how="all") if hasattr(pf_der, "dropna") else pf_der
        for _, _rd in _derw.iterrows():
            _dd2 = _rd.get("Date"); _td2 = _rd.get("Type"); _ud2 = _rd.get("Underlying"); _wd2 = _rd.get("Weight %")
            _hd2 = not _pd.isna(_dd2)
            _ht2 = isinstance(_td2, str) and _td2.strip() != "" and _td2 in _LDT
            _hu2 = isinstance(_ud2, str) and _ud2.strip() != ""
            _hw2 = (not _pd.isna(_wd2)) and float(_wd2) != 0.0
            if not (_hd2 or _ht2 or _hu2 or _hw2):
                continue  # blank row
            if not (_hd2 and _ht2 and _hu2 and _hw2):
                _der_inc = True; continue  # incomplete — flag it
            _wtot += float(_wd2); _any_pos = True
    except Exception:
        _wtot = 0.0; _der_inc = False; _any_pos = False
    _weights_ok = _any_pos and abs(_wtot - 100.0) <= 0.1
    if _der_inc:
        st.markdown("<div style='color:#ffb4ae;font-size:.83rem;margin:.1rem 0 .2rem'>"
                    "⚠ A derivative row is incomplete — each needs a <b>date, type, underlying and "
                    "weight</b>. Fill or remove it before computing (it isn't counted until complete)."
                    "</div>", unsafe_allow_html=True)
    if _any_pos and not _weights_ok:
        st.markdown("<div style='color:#ffb4ae;font-size:.83rem;margin:.1rem 0 .2rem'>"
                    f"⚠ Your positions (securities + derivatives) total <b>{_wtot:.1f}%</b> — they must "
                    "total <b>100%</b> before computing.</div>", unsafe_allow_html=True)
    elif _weights_ok and not _der_inc:
        st.markdown("<div style='color:#9be9a8;font-size:.83rem;margin:.1rem 0 .2rem'>"
                    "✓ Positions total 100% (securities + derivatives).</div>", unsafe_allow_html=True)

    # ── View range (above Compute) — bounds derived from the holdings dates ──
    _today2 = _dt.date.today()
    _hdates = []
    try:
        _tbl0 = pf_table.dropna(how="all") if hasattr(pf_table, "dropna") else pf_table
        for _, _r0 in _tbl0.iterrows():
            _d0 = _r0.get("Date")
            if _d0 is not None and not (isinstance(_d0, float) and _d0 != _d0):
                _hdates.append(_pd.Timestamp(_d0).date())
    except Exception:
        pass
    _rng_lo = min(_hdates) if _hdates else _today2.replace(year=_today2.year - 1)
    _rng_hi = _today2
    if _rng_lo >= _rng_hi:
        _rng_lo = _rng_hi - _dt.timedelta(days=1)
    _cf = st.session_state.get('pf_view_from')
    if _cf is None or _cf < _rng_lo or _cf > _rng_hi:
        st.session_state['pf_view_from'] = _rng_lo
    _ct = st.session_state.get('pf_view_to')
    if _ct is None or _ct < _rng_lo or _ct > _rng_hi:
        st.session_state['pf_view_to'] = _rng_hi
    _vrc0, _vrc1, _vrc2 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with _vrc1:
        st.date_input("View from", min_value=_rng_lo, max_value=_rng_hi,
                      key="pf_view_from", format="YYYY-MM-DD")
    with _vrc2:
        st.date_input("View to", min_value=_rng_lo, max_value=_rng_hi,
                      key="pf_view_to", format="YYYY-MM-DD")
    with _vrc0:
        st.caption("Optional — narrow the analytics to a date range (default: full history). "
                   "Applies to the results below after you compute.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    _pcrun = st.columns([2, 1, 2])
    with _pcrun[1]:
        pf_run = st.button(":material/query_stats: Compute analytics", type="primary",
                           use_container_width=True, key="pf_run")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if pf_run:
        try:
            # ── positions (each row = a held position, kept from its own entry date) ──
            _short = {"put": "Put", "call": "Call", "straddle": "Straddle", "strangle": "Strangle",
                      "safety_collar": "Collar-S", "aggressive_collar": "Collar-A",
                      "bull_call_spread": "BullSpr", "bear_put_spread": "BearSpr", "cgn_uncapped": "CGN"}
            _positions = []; _secseen = {}; _idseen = {}; _der_rows = []; _der_bad = []
            # securities
            _rows = pf_table.dropna(how="all") if hasattr(pf_table, "dropna") else pf_table
            for _, _r in _rows.iterrows():
                _d = _r.get("Date"); _t = _r.get("Ticker"); _w = _r.get("Weight %")
                if not (isinstance(_t, str) and _t.strip()):
                    continue
                if _pd.isna(_w) or float(_w) == 0.0:
                    continue
                _tk = _t.strip().upper()
                _ent = _pd.Timestamp(_d).normalize() if not _pd.isna(_d) else None
                _secseen[_tk] = _secseen.get(_tk, 0) + 1
                _sid = _tk if _secseen[_tk] == 1 else f"{_tk} #{_secseen[_tk]}"
                _positions.append({"id": _sid, "kind": "sec", "ticker": _tk, "entry": _ent, "w": float(_w)})
            # derivatives
            _derrows = pf_der.dropna(how="all") if hasattr(pf_der, "dropna") else pf_der
            for _ri, (_, _rd) in enumerate(_derrows.iterrows(), start=1):
                _d = _rd.get("Date"); _lab = _rd.get("Type"); _u = _rd.get("Underlying"); _w = _rd.get("Weight %")
                _s1 = _rd.get("Strike 1"); _s2 = _rd.get("Strike 2"); _ten = _rd.get("Tenor (y)")
                _has_date = not _pd.isna(_d)
                _has_type = isinstance(_lab, str) and _lab.strip() != ""
                _has_und = isinstance(_u, str) and _u.strip() != ""
                _has_w = (not _pd.isna(_w)) and float(_w) != 0.0
                if not (_has_date or _has_type or _has_und or _has_w):
                    continue  # blank row
                _miss = []
                if not _has_date: _miss.append("date")
                if not _has_type or _lab not in _LDT: _miss.append("type")
                if not _has_und: _miss.append("underlying")
                if not _has_w: _miss.append("weight")
                if _miss:
                    _der_bad.append((_ri, _miss)); continue
                _dtype = _LDT[_lab]; _u = _u.strip().upper(); _key = _pd.Timestamp(_d).normalize()
                try: _s1f = float(_s1)
                except Exception: _s1f = 1.0
                try: _s2f = float(_s2)
                except Exception: _s2f = float("nan")
                try: _tenf = float(_ten)
                except Exception: _tenf = 1.0
                if not (_tenf > 0): _tenf = 1.0
                _base = f"{_short.get(_dtype, _dtype)} {_u} K{_s1f:.2f}" + (f"/{_s2f:.2f}" if _s2f == _s2f else "")
                _idseen[_base] = _idseen.get(_base, 0) + 1
                _did = _base if _idseen[_base] == 1 else f"{_base} #{_idseen[_base]}"
                _positions.append({"id": _did, "kind": "der", "und": _u, "entry": _key, "dtype": _dtype,
                                   "s1": _s1f, "s2": _s2f, "tenor": _tenf, "w": float(_w)})
                _der_rows.append({"Date": _key.strftime("%Y-%m-%d"), "Underlying": _u, "Type": _lab,
                                  "Strike 1": _s1f, "Strike 2": (None if _s2f != _s2f else _s2f),
                                  "Tenor (y)": _tenf, "Weight %": float(_w)})
            _wtotal = sum(_p["w"] for _p in _positions)
            if _der_bad:
                st.error("Some derivative rows are incomplete — fill every field before computing: "
                         + "; ".join(f"row {_ri} missing {', '.join(_m)}" for _ri, _m in _der_bad)
                         + ". (Each derivative needs a date, type, underlying and weight.)")
            elif not _positions:
                st.warning("Add at least one position (security or derivative) with a weight.")
            elif abs(_wtotal - 100.0) > 0.1:
                st.error(f"Your positions total **{_wtotal:.1f}%** — securities + derivatives must "
                         "total **100%** before computing.")
            else:
                _bench = (pf_bench or "").strip().upper()
                _sectk = sorted({_p["ticker"] for _p in _positions if _p["kind"] == "sec"})
                _undtk = sorted({_p["und"] for _p in _positions if _p["kind"] == "der"})
                _fetch = list(dict.fromkeys(_sectk + _undtk + ([_bench] if _bench else [])))
                _entd = [_p["entry"] for _p in _positions if _p["entry"] is not None]
                _incept = (min(_entd).date() if _entd
                           else _dt.date.today().replace(year=_dt.date.today().year - 1))
                for _p in _positions:
                    if _p["entry"] is None:
                        _p["entry"] = _pd.Timestamp(_incept)
                _end = _dt.date.today() + _dt.timedelta(days=1)
                with st.spinner("Fetching prices and marking positions to market…"):
                    _px, _err = fetch_close_prices(_fetch, _incept, _end)
                if _err or _px is None:
                    st.error(f"Price data: {_err or 'no data returned'}")
                else:
                    # one growth-of-1 column per position, starting at its own entry date (cash before)
                    _pxc = _pd.DataFrame(index=_px.index); _miss = []
                    for _p in _positions:
                        if _p["kind"] == "sec":
                            if _p["ticker"] not in _px.columns:
                                _miss.append(_p["id"]); continue
                            _s = _px[_p["ticker"]].loc[_p["entry"]:].dropna()
                            _g = (_s / float(_s.iloc[0])) if len(_s) >= 2 else None
                        else:
                            if _p["und"] not in _px.columns:
                                _miss.append(_p["id"]); continue
                            try:
                                _g = _lds(_px[_p["und"]].loc[_p["entry"]:], _p["dtype"],
                                          _p["s1"], _p["s2"], _p["tenor"])
                            except Exception:
                                _g = None
                        if _g is None or getattr(_g, "empty", True):
                            _miss.append(_p["id"]); continue
                        _pxc[_p["id"]] = _g
                    if _bench and _bench in _px.columns:
                        _pxc[_bench] = _px[_bench]
                    if _miss:
                        st.warning("Couldn't price these positions (no data): " + ", ".join(_miss) + ".")
                    _w_by_id = {}
                    for _p in _positions:
                        if _p["id"] in _pxc.columns:
                            _w_by_id[_p["id"]] = _w_by_id.get(_p["id"], 0.0) + _p["w"] / 100.0
                    if not _w_by_id:
                        st.error("None of your positions could be priced — check the tickers/underlyings.")
                    else:
                        _log = [{"date": _pd.Timestamp(_incept).strftime("%Y-%m-%d"), "weights": _w_by_id}]
                        # securities grouped by entry date (for save/load round-trips)
                        _sec_by_date = {}
                        for _p in _positions:
                            if _p["kind"] == "sec" and _p["id"] in _pxc.columns:
                                _dk = _pd.Timestamp(_p["entry"]).strftime("%Y-%m-%d")
                                _sec_by_date.setdefault(_dk, {})
                                _sec_by_date[_dk][_p["ticker"]] = _sec_by_date[_dk].get(_p["ticker"], 0.0) + _p["w"] / 100.0
                        _sec_log = [{"date": _dk, "weights": _wd} for _dk, _wd in sorted(_sec_by_date.items())]
                        _res = _pf.analyze(_pxc, _log, benchmark=(_bench if _bench in _pxc.columns else None))
                        _holdtk = list(_w_by_id.keys())
                        _moments = None
                        try:
                            _rh = _pxc[_holdtk].pct_change().dropna()
                            if len(_rh) > 5 and len(_holdtk) >= 1:
                                _moments = ((_rh.mean() * 252).values, (_rh.cov() * 252).values)
                        except Exception:
                            _moments = None
                        st.session_state['pf_result'] = {
                            'res': _res, 'log': _log, 'sec_log': _sec_log, 'der_rows': _der_rows,
                            'name': pf_name, 'bench': _bench,
                            'missing': [t for t in _fetch if t not in _px.columns],
                            'moments': _moments, 'moment_tickers': _holdtk,
                        }
        except Exception as _e:
            st.error(f"Couldn't compute: {_e}")

    _PR = st.session_state.get('pf_result')
    if _PR:
        _res = _PR['res']
        if _res['metrics']['n'] == 0:
            st.warning("Couldn't build a track from these inputs — check the tickers and dates.")
        else:
            if _PR['missing']:
                st.warning("No price data for: " + ", ".join(_PR['missing']) + " — skipped.")
            _port_full = _res['port_returns']; _bench_full = _res['bench_returns']
            _fs = _port_full.index[0].date(); _fe = _port_full.index[-1].date()
            # View range comes from the From/To boxes above the Compute button
            _vs = st.session_state.get('pf_view_from', _fs)
            _ve = st.session_state.get('pf_view_to', _fe)
            if _vs > _ve:
                _vs, _ve = _ve, _vs
            _pdates = _port_full.index.date
            _pv = _port_full[(_pdates >= _vs) & (_pdates <= _ve)]
            if len(_pv) < 3:
                _pv = _port_full; _vs, _ve = _fs, _fe
            _full_view = (_vs <= _fs and _ve >= _fe)
            _m = _pf.metrics(_pv)
            _eq = _pf.equity_curve(_pv)
            _cap = None; _bench_eq = None
            if _bench_full is not None and len(_bench_full):
                _bdates = _bench_full.index.date
                _bv = _bench_full[(_bdates >= _vs) & (_bdates <= _ve)]
                if len(_bv) > 2:
                    _bench_eq = _pf.equity_curve(_bv); _cap = _pf.capm(_pv, _bv)
            # Risk inputs (VaR or ES), seeded from the Risk Profile; drive headline + panel.
            _rp0 = st.session_state.get('rp_result')
            _dl0 = max(-60, min(-2, int(_rp0['H_pct']))) if (_rp0 and _rp0.get('H_pct') is not None) else -15
            _da0 = max(1, min(25, int(_rp0['alpha_pct']))) if (_rp0 and _rp0.get('alpha_pct') is not None) else 5
            _dL0 = max(-60, min(-2, int(_rp0['L_pct']))) if (_rp0 and _rp0.get('L_pct') is not None) else -20
            st.session_state.setdefault('pf_risk_lim', _dl0)
            st.session_state.setdefault('pf_risk_alpha', _da0)
            st.session_state.setdefault('pf_risk_L', _dL0)
            st.session_state.setdefault('pf_risk_method', 'VaR')
            _is_es = str(st.session_state.get('pf_risk_method', 'VaR')) == 'ES'
            _H = float(st.session_state.get('pf_risk_lim', _dl0))
            _A = float(st.session_state.get('pf_risk_alpha', _da0))
            _Lf = float(st.session_state.get('pf_risk_L', _dL0))
            # ── Time-varying tolerance: step-function (H, α, L) from the editable tolerance timeline ──
            _rph = st.session_state.get('rp_history', []) or []
            # Seed the operational timeline from the assessment history the first time we need it.
            if 'tol_timeline' not in st.session_state and _rph:
                st.session_state['tol_timeline'] = [
                    {'date': str(h.get('date')), 'H_pct': h.get('H_pct'),
                     'alpha_pct': h.get('alpha_pct'), 'L_pct': h.get('L_pct')}
                    for h in sorted(_rph, key=lambda x: str(x.get('date')))]
            def _tl_rows():
                """Valid, sorted tolerance-timeline rows from session (date + H/α/L all present)."""
                _out = []
                for _r in (st.session_state.get('tol_timeline') or []):
                    _d = _r.get('date'); _h = _r.get('H_pct'); _a = _r.get('alpha_pct'); _l = _r.get('L_pct')
                    if _d in (None, '') or _h is None or _a is None or _l is None:
                        continue
                    try:
                        _out.append((_pd.Timestamp(_d), float(_h), float(_a), float(_l)))
                    except Exception:
                        continue
                return sorted(_out, key=lambda s: s[0])
            _tlr = _tl_rows()
            _distinct = {(r[1], r[2], r[3]) for r in _tlr}
            _tv_avail = (len(_tlr) >= 2 and len(_distinct) >= 2)
            _tv = bool(st.session_state.get('pf_risk_tv', False)) and _tv_avail
            def _seg_list():
                """Sorted tolerance segments (start_ts, H, α, L). Single level when TV off."""
                if _tv and _tlr:
                    return _tlr
                return [(_pd.Timestamp.min, _H, _A, _Lf)]
            def _thr(index):
                """Per-date in-force (H, α, L) as aligned Series (most recent assessment on/before)."""
                _idx = _pd.DatetimeIndex(index)
                _Hs = _pd.Series(np.nan, index=_idx); _As = _Hs.copy(); _Ls = _Hs.copy()
                for _ts, _h, _a, _l in _seg_list():
                    _m = _idx >= _ts
                    _Hs[_m] = _h; _As[_m] = _a; _Ls[_m] = _l
                _s0 = _seg_list()[0]
                return _Hs.fillna(_s0[1]), _As.fillna(_s0[2]), _Ls.fillna(_s0[3])
            def _seg_stats(_series):
                """Per-period breach stats over a value series (drawdown / horizon-return)."""
                _s = _series.dropna(); _segs = _seg_list(); _out = []
                for _i, (_ts, _h, _a, _l) in enumerate(_segs):
                    _end = _segs[_i + 1][0] if _i + 1 < len(_segs) else _pd.Timestamp.max
                    _mask = (_s.index < _end) if _i == 0 else ((_s.index >= _ts) & (_s.index < _end))
                    _seg = _s[_mask]; _n = int(len(_seg))
                    if _n == 0:
                        continue
                    _nb = int((_seg < _h).sum()); _fr = _nb / _n * 100.0
                    _tl = float(_seg[_seg < _h].mean()) if _nb else 0.0
                    _ok = ((_nb == 0) or (_tl >= _l)) if _is_es else (_fr <= _a)
                    _out.append({'start': _seg.index[0], 'end': _seg.index[-1], 'H': _h, 'a': _a,
                                 'L': _l, 'n': _n, 'nb': _nb, 'fr': _fr, 'tail': _tl, 'ok': _ok})
                return _out
            _dd_all = (_eq / _eq.cummax() - 1.0) * 100.0
            _Hall, _Aall, _Lall = _thr(_eq.index)
            _brk_all = (_dd_all < _Hall)                       # per-day breach vs the in-force H
            _bn_all = int(_brk_all.sum())
            _bfreq_all = (_bn_all / len(_dd_all) * 100.0) if len(_dd_all) else 0.0
            _tail_dd = float(_dd_all[_brk_all].mean()) if _bn_all > 0 else 0.0
            _segst_all = _seg_stats(_dd_all)
            _all_ok = all(_s['ok'] for _s in _segst_all) if _segst_all else True
            if _is_es:
                _bcol = "#fafafa" if _bn_all == 0 else ("#f5b942" if _all_ok else "#f85149")
                _m2_label = "Tail-avg loss (vs L)" if not _tv else "Tail vs L (worst period)"
                _m2_value = f"{_tail_dd:.1f}%" if _bn_all > 0 else "—"
                _m2_tip = ("Average drawdown on the days beyond H — should stay at or above your CVaR floor L."
                           + (" Evaluated per tolerance period." if _tv else ""))
            else:
                _bcol = "#fafafa" if _bn_all == 0 else ("#f5b942" if _all_ok else "#f85149")
                _m2_label = "Breach frequency"
                _m2_value = f"{_bfreq_all:.1f}%"
                _m2_tip = (f"Share of days the drawdown was beyond H — your α caps this."
                           + (" Each period is checked against its own α." if _tv else f" α = {_A:.0f}%."))
            def _pf_breach_metric(_label, _value, _color, _tip):
                # matches st.metric styling: label 14px / value 36px, colour #e7ecf4, weight 400
                return ("<div title='" + _tip + "'>"
                        "<div style='color:#e7ecf4;font-size:14px;font-weight:400;line-height:1.6'>" + _label + "</div>"
                        "<div style='color:" + _color + ";font-size:36px;font-weight:400;"
                        "line-height:1.15'>" + _value + "</div></div>")
            _rng_note = "full history" if _full_view else f"{_vs} → {_ve}"
            with st.container(border=True):
                st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;margin:.1rem 0 .95rem'>"
                            f"<span style='color:#2dd4bf;font-size:1.3rem;font-weight:700'>{_PR['name']}</span>"
                            f"<span style='color:#8b97a8;font-size:.85rem'> &nbsp;·&nbsp; {_rng_note} "
                            f"&nbsp;·&nbsp; {_m['n']} trading days</span></div>", unsafe_allow_html=True)
                _c1, _c2, _c3, _c4, _c5 = st.columns(5)
                _c1.metric("Return since inception" if _full_view else "Return (range)",
                           f"{_m['cumulative']*100:.1f}%")
                _c2.metric("Annualised return", f"{_m['annualised']*100:.1f}%")
                _c3.metric("Volatility (ann.)", f"{_m['vol']*100:.1f}%")
                _c4.metric("Sharpe", f"{_m['sharpe']:.2f}" if _m['sharpe'] == _m['sharpe'] else "—")
                _c5.markdown(_pf_breach_metric(
                    f"Breaches (vs H = {_H:.0f}%)", f"{_bn_all}", _bcol,
                    "Number of days the portfolio's drawdown fell past your loss limit H."),
                    unsafe_allow_html=True)
                _c6, _c7, _c8, _c9, _c10 = st.columns(5)
                _c6.metric("Max drawdown", f"{_m['max_drawdown']*100:.1f}%")
                _aval = _cap['alpha'] if (_cap and _cap['alpha'] == _cap['alpha']) else None
                _acol = "#f85149" if (_aval is not None and _aval < 0) else "#e7ecf4"
                _c7.markdown(_pf_breach_metric(
                    f"Jensen's α vs {_PR['bench'] or 'benchmark'} (ann.)",
                    (f"{_aval*100:+.2f}%" if _aval is not None else "—"), _acol,
                    "Annualised Jensen's alpha — return beyond what benchmark exposure (β) explains."),
                    unsafe_allow_html=True)
                _c8.metric("Beta (β)", f"{_cap['beta']:.2f}" if _cap and _cap['beta'] == _cap['beta'] else "—")
                _c9.metric("R²", f"{_cap['r2']:.2f}" if _cap and _cap['r2'] == _cap['r2'] else "—")
                _c10.markdown(_pf_breach_metric(_m2_label, _m2_value, _bcol, _m2_tip),
                              unsafe_allow_html=True)
                st.caption(f"Daily VaR (5%): {_m['var']*100:.2f}%  ·  Daily CVaR (5%): {_m['cvar']*100:.2f}%  "
                           f"·  risk-free {int(_pf.RF_ANNUAL*100)}% p.a.")

            # synced left (return %) and right (portfolio value €) y-ranges so the value
            # area's top edge coincides exactly with the return line.
            _cap0 = float(pf_capital) if pf_capital else 0.0
            _allret = list((_eq.values - 1.0) * 100)
            if _bench_eq is not None and len(_bench_eq):
                _allret += list((_bench_eq.values - 1.0) * 100)
            _ymin = min(_allret) if _allret else -1.0
            _ymax = max(_allret) if _allret else 1.0
            _ypad = max((_ymax - _ymin) * 0.06, 1.0)
            _ylo, _yhi = _ymin - _ypad, _ymax + _ypad
            _vlo, _vhi = _cap0 * (1.0 + _ylo / 100.0), _cap0 * (1.0 + _yhi / 100.0)

            _fig = _go.Figure()
            # portfolio value (€) — faint area on the secondary axis, drawn behind the lines
            if _cap0 > 0:
                _fig.add_trace(_go.Scatter(x=_eq.index, y=_cap0 * _eq.values, mode='lines',
                                           name='Portfolio value (€)', yaxis='y2',
                                           line=dict(width=0), fill='tozeroy',
                                           fillcolor='rgba(45,212,191,0.10)',
                                           hovertemplate='%{x|%Y-%m-%d}<br>value €%{y:,.0f}<extra></extra>'))
            _fig.add_trace(_go.Scatter(x=_eq.index, y=(_eq.values - 1.0) * 100, mode='lines',
                                       name=_PR['name'], line=dict(color='#2dd4bf', width=2.2),
                                       hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>'))
            if _bench_eq is not None and len(_bench_eq):
                _be = _bench_eq
                _fig.add_trace(_go.Scatter(x=_be.index, y=(_be.values - 1.0) * 100, mode='lines',
                                           name=_PR['bench'], line=dict(color='#8b949e', width=1.6, dash='dot'),
                                           hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>'))
            # breach-day markers (days the drawdown fell past the in-force loss limit H)
            _brm = _brk_all
            if bool(_brm.any()):
                _bx = _eq.index[_brm.values]; _byv = (_eq.values[_brm.values] - 1.0) * 100
                _brname = ('Breach day (drawdown past in-force limit)' if _tv
                           else f'Breach day (drawdown < {_H:.0f}%)')
                _fig.add_trace(_go.Scatter(x=_bx, y=_byv, mode='markers', name=_brname,
                                           marker=dict(color='#f85149', size=5, symbol='circle', line=dict(width=0)),
                                           hovertemplate='%{x|%Y-%m-%d}<br><b>breach</b> — drawdown past H<extra></extra>'))
            # rebalance markers (skip inception)
            for _e in _PR['log'][1:]:
                _rd = _pd.Timestamp(_e['date'])
                _near = _eq.index[_eq.index >= _rd]
                if len(_near):
                    _yy = (_eq.loc[_near[0]] - 1.0) * 100
                    _fig.add_trace(_go.Scatter(x=[_near[0]], y=[_yy], mode='markers',
                                               marker=dict(color='#f5b942', size=9, symbol='diamond',
                                                           line=dict(color='#0d1117', width=1)),
                                               name='Rebalance', showlegend=False,
                                               hovertemplate=f'Rebalanced {_e["date"]}<extra></extra>'))
            _fig.update_layout(template='plotly_dark', paper_bgcolor='#1b2330', plot_bgcolor='#0e1521',
                               height=420, margin=dict(t=58, b=40, l=64, r=(72 if _cap0 > 0 else 24)),
                               title=dict(text='Total return vs benchmark<br>'
                                               '<span style="font-size:11px;color:#8b949e">Growth of €1</span>',
                                          x=0.5, xanchor='center',
                                          font=dict(color='#E3C77E', size=14)),
                               legend=dict(bgcolor='rgba(26,26,46,0.6)', x=0.01, y=0.99,
                                           font=dict(color='#c9d1d9', size=11)),
                               yaxis=dict(title='Total return (%)', gridcolor='#27344e', griddash='dot',
                                          color='#c0c8d8', zerolinecolor='#3a4762', automargin=False,
                                          range=[_ylo, _yhi], autorange=False),
                               yaxis2=dict(title='Portfolio value (€)', overlaying='y', side='right',
                                           range=[_vlo, _vhi], autorange=False, showgrid=False,
                                           color='#2dd4bf', tickformat=',.0f', automargin=False,
                                           visible=(_cap0 > 0)),
                               xaxis=dict(gridcolor='#27344e', griddash='dot', color='#c0c8d8',
                                          range=([_eq.index[0], _eq.index[-1]] if len(_eq) else None),
                                          autorange=(False if len(_eq) else True)))
            st.plotly_chart(_fig, use_container_width=True, config={'displayModeBar': True})

            with st.container(border=True):
                st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
                # ── Risk vs your tolerance (VaR or ES) ──
                _from_profile = bool(_rp0 and _rp0.get('H_pct') is not None)
                st.markdown("<div style='text-align:center;color:#E3C77E;font-size:20px;font-weight:700;"
                            "margin:1.1rem 0 .2rem'>Risk vs your tolerance</div>", unsafe_allow_html=True)
                _msel1, _msel2 = st.columns([1.1, 1.9], vertical_alignment="center")
                with _msel1:
                    st.radio("Risk measure", ["VaR", "ES"],
                             format_func=lambda x: "VaR · P(r<H) ≤ α" if x == "VaR"
                                                   else "Expected Shortfall · E[r|r<H] ≥ L",
                             horizontal=True, key="pf_risk_method")
                with _msel2:
                    if _from_profile:
                        _sc0 = _rp0.get('score'); _asd = _rp0.get('date')
                        _sctxt = ((f" &nbsp;·&nbsp; score <b style='color:#E3C77E'>{_sc0}/47</b>"
                                   + (f" <span style='color:#6b7689'>(assessed {_asd})</span>" if _asd else ""))
                                  if _sc0 is not None else "")
                        _scope = (f"From your <b>Risk Profile</b>: <b style='color:#c9d1d9'>{_rp0.get('band','—')}</b>"
                                  f"{_sctxt} &nbsp;·&nbsp; H = {_rp0.get('H_pct')}%, α = {_rp0.get('alpha_pct')}%, "
                                  f"L = {_rp0.get('L_pct')}%.")
                    else:
                        _scope = "Complete the <b>Risk Profile</b> to seed H, α and L automatically."
                    # Which profile band do the *current* limits most resemble? (correspondence,
                    # not a fresh score — the sliders don't produce a psychometric score.)
                    _band_order = ["Low", "Below-average", "Average / moderate", "Above-average", "High"]
                    _score_disp = {"Low": "13–18", "Below-average": "19–22", "Average / moderate": "23–28",
                                   "Above-average": "29–32", "High": "33–47"}
                    _bb, _bbd = _band_order[0], 1e18
                    for _bn in _band_order:
                        _hb, _ab = _rp.BAND_TO_HALPHA[_bn]; _lb = _rp.BAND_TO_L[_bn]
                        _dd = (((_H - _hb) / 25.0) ** 2 + ((_Lf - _lb) / 30.0) ** 2) if _is_es \
                              else (((_H - _hb) / 25.0) ** 2 + ((_A - _ab) / 10.0) ** 2)
                        if _dd < _bbd:
                            _bbd, _bb = _dd, _bn
                    _corr = (f"<br>These limits align with the <b style='color:#c9d1d9'>{_bb}</b> band "
                             f"(scores {_score_disp[_bb]}).")
                    if _from_profile and _rp0.get('band') in _band_order:
                        _ci = _band_order.index(_bb); _ai = _band_order.index(_rp0.get('band'))
                        if _ci < _ai:
                            _corr += " <span style='color:#e8cd84'>Stricter than your assessed profile.</span>"
                        elif _ci > _ai:
                            _corr += " <span style='color:#e8cd84'>Looser than your assessed profile.</span>"
                        else:
                            _corr += " <span style='color:#9be9a8'>Matches your assessed profile.</span>"
                    _lamtxt = ""
                    if (not _is_es) and (_PR.get('moments') is not None):
                        _lam = None
                        try:
                            _lam = implied_lambda(_H / 100.0, _A / 100.0, _PR['moments'][0], _PR['moments'][1])
                        except Exception:
                            _lam = None
                        if _lam is not None:
                            _lamtxt = (f"<br>Implied risk-aversion <b style='color:#7fb3e8'>λ ≈ {_lam:.2f}</b> — the "
                                       "mean-variance risk-aversion your VaR tolerance implies for this universe.")
                        else:
                            _lamtxt = ("<br>Implied risk-aversion <b style='color:#7fb3e8'>λ</b>: your VaR tolerance is "
                                       "<i>non-binding</i> for this universe at this H, α — the portfolio sits comfortably "
                                       "inside it, so no λ binds.")
                    st.markdown(f"<div style='color:#8b97a8;font-size:.82rem;line-height:1.5'>{_scope}{_corr}{_lamtxt}</div>",
                                unsafe_allow_html=True)
                st.toggle("Evaluate against my tolerance history (time-varying)", key="pf_risk_tv",
                          disabled=not _tv_avail,
                          help=("Judge each date against the risk limits that were in force then "
                                "(from your dated assessment history), instead of one fixed level."
                                if _tv_avail else
                                "Take the risk questionnaire more than once (with different results) to "
                                "unlock this — then each period is checked against the limits in force at the time."))
                _rl1, _rl2 = st.columns(2)
                with _rl1:
                    st.slider("Loss limit / threshold H", min_value=-60, max_value=-2, step=1,
                              key="pf_risk_lim", format="%d%%", disabled=_tv)
                with _rl2:
                    if _is_es:
                        st.slider("CVaR floor L  (E[r|r<H] ≥ L)", min_value=-60, max_value=-2, step=1,
                                  key="pf_risk_L", format="%d%%", disabled=_tv)
                    else:
                        st.slider("Max breach frequency α", min_value=1, max_value=25, step=1,
                                  key="pf_risk_alpha", format="%d%%", disabled=_tv)
                if _tv:
                    st.caption("⏱ Time-varying mode: limits come from your tolerance timeline, so the "
                               "sliders are inactive (they set the single-level view). The H (and L) line steps "
                               "on each effective date, and each period is judged against its own limits.")
                # ── Editable tolerance timeline (effective dates & levels) ──
                with st.expander("🗓  Tolerance timeline — effective dates & levels"):
                    st.caption("Each row is a loss-tolerance level and the date it took effect. Rows are seeded "
                               "from your risk assessments; you can backdate, edit levels, add or delete rows. "
                               "Your assessment score stays on the Risk Profile page — edited or added rows are "
                               "your own policy, marked “manual”. These rows drive the time-varying evaluation above.")
                    _src_keys = {(str(h.get('date')), h.get('H_pct'), h.get('alpha_pct'), h.get('L_pct')):
                                 h.get('score') for h in _rph}
                    _tl_disp = []
                    for _r in (st.session_state.get('tol_timeline') or []):
                        _k = (str(_r.get('date')), _r.get('H_pct'), _r.get('alpha_pct'), _r.get('L_pct'))
                        _sc = _src_keys.get(_k)
                        try:
                            _dval = _pd.Timestamp(_r.get('date')).date()
                        except Exception:
                            _dval = None
                        _tl_disp.append({"Effective date": _dval, "H %": _r.get('H_pct'),
                                         "α %": _r.get('alpha_pct'), "L %": _r.get('L_pct'),
                                         "Source": (f"assessment · {_sc}/47" if _sc is not None else "manual")})
                    _tl_df = _pd.DataFrame(_tl_disp, columns=["Effective date", "H %", "α %", "L %", "Source"])
                    _tl_edited = st.data_editor(
                        _tl_df, num_rows="dynamic", hide_index=True, use_container_width=True, key="tol_editor",
                        column_config={
                            "Effective date": st.column_config.DateColumn("Effective date", format="YYYY-MM-DD"),
                            "H %": st.column_config.NumberColumn("H %", min_value=-60, max_value=-2, step=1,
                                                                 help="Loss threshold (negative)."),
                            "α %": st.column_config.NumberColumn("α %", min_value=1, max_value=25, step=1,
                                                                 help="Max breach frequency (VaR)."),
                            "L %": st.column_config.NumberColumn("L %", min_value=-60, max_value=-2, step=1,
                                                                 help="CVaR floor (ES, negative)."),
                            "Source": st.column_config.TextColumn("Source", disabled=True,
                                                                  help="‘assessment · score/47’ when the row matches "
                                                                       "a questionnaire result; otherwise ‘manual’."),
                        })
                    _new_tl = []
                    for _, _er in _tl_edited.iterrows():
                        _d = _er.get("Effective date")
                        if _d is None or (isinstance(_d, float) and _d != _d):
                            continue
                        _h = _er.get("H %"); _a = _er.get("α %"); _l = _er.get("L %")
                        if _pd.isna(_h) or _pd.isna(_a) or _pd.isna(_l):
                            continue
                        _new_tl.append({"date": _pd.Timestamp(_d).strftime("%Y-%m-%d"),
                                        "H_pct": int(round(float(_h))), "alpha_pct": int(round(float(_a))),
                                        "L_pct": int(round(float(_l)))})
                    st.session_state['tol_timeline'] = _new_tl
                    if not _tv_avail:
                        st.caption("➕ Add at least two rows with different levels to unlock the "
                                   "time-varying toggle above.")
                def _grad_area(_x, _y, _hover):
                    try:
                        return _go.Scatter(x=_x, y=_y, mode='lines', line=dict(color='#cfd8e3', width=1.3),
                                           fill='tozeroy', fillgradient=dict(type='vertical',
                                           colorscale=[[0.0, '#f85149'], [0.5, '#e3c77e'], [1.0, '#2ea043']]),
                                           hovertemplate=_hover)
                    except Exception:
                        return _go.Scatter(x=_x, y=_y, mode='lines', line=dict(color='#2dd4bf', width=1.4),
                                           fill='tozeroy', fillcolor='rgba(45,212,191,.16)', hovertemplate=_hover)

                # Shared x-axis window so all three charts line up date-for-date, even when a
                # rolling series (e.g. horizon-return) has no data until its window has elapsed.
                _xwin = [_eq.index[0], _eq.index[-1]] if len(_eq) else None

                def _limit_overlay(_f, _x, _vals, _color, _dash, _name, _label):
                    """Draw a loss-limit reference: a flat hline (+annotation) when constant,
                    or a stepped dashed line when it changes across tolerance periods."""
                    _v = np.asarray(_vals, float)
                    if _v.size == 0:
                        return
                    if float(np.nanmax(_v)) == float(np.nanmin(_v)):
                        _f.add_hline(y=float(_v[0]), line=dict(color=_color, width=2.0, dash=_dash),
                                     annotation_text=f"{_label} = {_v[0]:.0f}%",
                                     annotation_position=("bottom right" if _dash == 'dash' else "top right"),
                                     annotation_font_color=_color)
                    else:
                        _f.add_trace(_go.Scatter(x=_x, y=_v, mode='lines', name=_name, showlegend=False,
                                     line=dict(color=_color, width=2.0, dash=_dash, shape='hv'),
                                     hovertemplate=f'%{{x|%Y-%m-%d}}<br>{_label} = %{{y:.0f}}%<extra></extra>'))

                def _risk_fig(_x, _y, _ytitle, _xrange=None, _Hv=None, _Lv=None):
                    _f = _go.Figure()
                    _f.add_trace(_grad_area(_x, _y, '%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>'))
                    if _Hv is None:
                        _f.add_hline(y=_H, line=dict(color='#f85149', width=2.2, dash='dash'),
                                     annotation_text=f"H = {_H:.0f}%", annotation_position="bottom right",
                                     annotation_font_color="#f85149")
                    else:
                        _limit_overlay(_f, _x, _Hv, '#f85149', 'dash', 'H limit', 'H')
                    if _is_es:
                        if _Lv is None:
                            _f.add_hline(y=_Lf, line=dict(color='#f5b942', width=1.8, dash='dot'),
                                         annotation_text=f"L floor = {_Lf:.0f}%", annotation_position="top right",
                                         annotation_font_color="#f5b942")
                        else:
                            _limit_overlay(_f, _x, _Lv, '#f5b942', 'dot', 'L floor', 'L')
                    _xaxis = dict(gridcolor='#27344e', griddash='dot', color='#c0c8d8')
                    if _xrange is not None:
                        _xaxis.update(range=_xrange, autorange=False)
                    _f.update_layout(template='plotly_dark', paper_bgcolor='#1b2330', plot_bgcolor='#0e1521',
                                     height=280, margin=dict(t=16, b=30, l=64, r=24), showlegend=False,
                                     yaxis=dict(title=_ytitle, gridcolor='#27344e', griddash='dot',
                                                color='#c0c8d8', zerolinecolor='#3a4762', automargin=False),
                                     xaxis=_xaxis)
                    return _f

                def _risk_status(_series, _what):
                    _s = _series.dropna()
                    if len(_s) == 0:
                        return
                    if _tv:
                        _rows = []; _any_bad = False
                        for _sg in _seg_stats(_series):
                            _lab = f"{_sg['start']:%d %b %Y} → {_sg['end']:%d %b %Y}"
                            if _sg['nb'] == 0:
                                _cl = "#9be9a8"; _ic = "&#10003;"
                                _txt = f"never beyond H = {_sg['H']:.0f}%"
                            elif _is_es:
                                if _sg['ok']:
                                    _cl = "#e8cd84"; _ic = "&#9888;"
                                    _txt = (f"{_sg['nb']}/{_sg['n']} days beyond H = {_sg['H']:.0f}%; tail avg "
                                            f"{_sg['tail']:.1f}% ≥ L = {_sg['L']:.0f}% ✓")
                                else:
                                    _cl = "#ffb4ae"; _ic = "&#9888;"; _any_bad = True
                                    _txt = (f"tail avg {_sg['tail']:.1f}% &lt; L = {_sg['L']:.0f}% "
                                            f"({_sg['nb']}/{_sg['n']} beyond H) — breached")
                            else:
                                if _sg['ok']:
                                    _cl = "#e8cd84"; _ic = "&#9888;"
                                    _txt = (f"{_sg['nb']}/{_sg['n']} days beyond H = {_sg['H']:.0f}% "
                                            f"({_sg['fr']:.1f}%) ≤ α = {_sg['a']:.0f}% ✓")
                                else:
                                    _cl = "#ffb4ae"; _ic = "&#9888;"; _any_bad = True
                                    _txt = (f"{_sg['fr']:.1f}% of days beyond H = {_sg['H']:.0f}% "
                                            f"&gt; α = {_sg['a']:.0f}% — breached")
                            _rows.append(f"<div style='color:{_cl};font-size:.85rem;margin:.18rem 0'>"
                                         f"{_ic} <b>{_lab}</b> — {_txt}</div>")
                        _bg = "rgba(248,81,73,.10)" if _any_bad else "rgba(38,166,65,.08)"
                        _bd = "#f85149" if _any_bad else "#2ea043"
                        st.markdown(f"<div style='background:{_bg};border:1px solid {_bd};border-radius:8px;"
                                    f"padding:.5rem .9rem'><div style='color:#cfd8e3;font-size:.8rem;"
                                    f"margin-bottom:.15rem'>{_what} — checked against each period's own limits:"
                                    f"</div>" + "".join(_rows) + "</div>", unsafe_allow_html=True)
                        return
                    _nt = int(len(_s)); _nb = int((_s < _H).sum())
                    _wv = float(_s.min()) if _nt else 0.0
                    _wd = (_s.idxmin().date() if (_nt and hasattr(_s.idxmin(), 'date')) else "—")
                    if _nb == 0:
                        st.markdown("<div style='background:rgba(38,166,65,.10);border:1px solid #2ea043;"
                                    "border-radius:8px;padding:.55rem 1rem;color:#9be9a8;font-size:.88rem'>"
                                    f"&#10003; <b>Never beyond H.</b> {_what} never passed your {_H:.0f}% limit — "
                                    f"worst {_wv:.1f}%.</div>", unsafe_allow_html=True)
                    elif _is_es:
                        _tail = float(_s[_s < _H].mean())
                        if _tail >= _Lf:
                            st.markdown("<div style='background:rgba(245,185,66,.10);border:1px solid #d6a32e;"
                                        "border-radius:8px;padding:.55rem 1rem;color:#e8cd84;font-size:.88rem'>"
                                        f"&#9888; <b>Within your ES floor.</b> {_what} went beyond H on <b>{_nb}</b> of "
                                        f"{_nt} days; the average loss in that tail is <b>{_tail:.1f}%</b>, at or above "
                                        f"your floor L = {_Lf:.0f}%. Worst {_wv:.1f}% on {_wd}.</div>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='background:rgba(248,81,73,.13);border:1px solid #f85149;"
                                        "border-radius:8px;padding:.55rem 1rem;color:#ffb4ae;font-size:.9rem'>"
                                        f"&#9888; <b>ES floor breached.</b> The average loss in the tail beyond H is "
                                        f"<b>{_tail:.1f}%</b> — <b>below your floor L = {_Lf:.0f}%</b> ({_nb} of {_nt} "
                                        f"days beyond H). Worst {_wv:.1f}% on {_wd}. Consider de-risking.</div>",
                                        unsafe_allow_html=True)
                    else:
                        _fr = _nb / _nt
                        if _fr <= (_A / 100.0):
                            st.markdown("<div style='background:rgba(245,185,66,.10);border:1px solid #d6a32e;"
                                        "border-radius:8px;padding:.55rem 1rem;color:#e8cd84;font-size:.88rem'>"
                                        f"&#9888; <b>Within your α tolerance.</b> {_what} passed the {_H:.0f}% limit on "
                                        f"<b>{_nb}</b> of {_nt} days (<b>{_fr*100:.1f}%</b>), at or under your "
                                        f"α = {_A:.0f}%. Worst {_wv:.1f}% on {_wd}.</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='background:rgba(248,81,73,.13);border:1px solid #f85149;"
                                        "border-radius:8px;padding:.55rem 1rem;color:#ffb4ae;font-size:.9rem'>"
                                        f"&#9888; <b>Tolerance breached.</b> {_what} passed your {_H:.0f}% limit on "
                                        f"<b>{_nb}</b> of {_nt} days (<b>{_fr*100:.1f}%</b>) — <b>above your "
                                        f"α = {_A:.0f}%</b>. Worst {_wv:.1f}% on {_wd}. Consider de-risking or revisiting "
                                        "your tolerance.</div>", unsafe_allow_html=True)

                # Chart 1 — drawdown breach
                st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.92rem;margin:.6rem 0 0'>"
                            "1 · Drawdown vs limit <span style='color:#8b97a8;font-weight:400'>— cumulative loss "
                            "from the running peak</span></div>", unsafe_allow_html=True)
                st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
                _dd = (_eq / _eq.cummax() - 1.0) * 100.0
                _ddH, _ddA, _ddL = _thr(_dd.index)
                st.plotly_chart(_risk_fig(_dd.index, _dd.values, 'Drawdown (%)', _xrange=_xwin,
                                          _Hv=_ddH.values, _Lv=_ddL.values),
                                use_container_width=True, config={'displayModeBar': False})
                _risk_status(_dd, "Drawdown")
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<details style='background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;"
                    "padding:.4rem .8rem;margin:.3rem 0 .2rem;font-size:.82rem'>"
                    "<summary style='cursor:pointer;color:#79b6ff;font-weight:600;list-style:none'>"
                    "✨ AI-powered: What is a drawdown breach?</summary>"
                    "<div style='color:#aebccd;margin-top:.4rem;line-height:1.55'>"
                    "<b>Drawdown</b> is how far the portfolio sits below its own running peak (its high-water mark). "
                    "A <i>drawdown breach</i> is when that decline passes your loss limit <b>H</b>. It is cumulative "
                    "and path-dependent — it captures a slow grind lower (many small losses adding up), not just one "
                    "bad day — so it's the most intuitive &ldquo;am I down more than I can stomach?&rdquo; reading, "
                    "measured from your best level so far. Keeping the share of breaching days at or under <b>&alpha;</b> "
                    "is the running-risk check.</div></details>", unsafe_allow_html=True)
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # Chart 2 — horizon-return breach
                _HZ = {"1 week (5d)": 5, "2 weeks (10d)": 10, "1 month (21d)": 21,
                       "3 months (63d)": 63, "6 months (126d)": 126, "1 year (252d)": 252}
                st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.92rem;margin:.8rem 0 0'>"
                            "2 · Horizon-return vs limit <span style='color:#8b97a8;font-weight:400'>— the "
                            "rolling-window return the optimiser constrains</span></div>", unsafe_allow_html=True)
                _hzc1, _hzc2 = st.columns([2, 3])
                with _hzc1:
                    _hsel = st.selectbox("Return horizon", list(_HZ.keys()), index=3, key="pf_hz")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                _hd = _HZ[_hsel]
                _hret = (_eq / _eq.shift(_hd) - 1.0) * 100.0
                _hret = _hret.dropna()
                if len(_hret) < 5:
                    st.info(f"Not enough history for a {_hsel} horizon yet — pick a shorter horizon, or come back "
                            "once the portfolio has more track record.")
                else:
                    _hH, _hA, _hL = _thr(_hret.index)
                    st.plotly_chart(_risk_fig(_hret.index, _hret.values, f'{_hsel} return (%)', _xrange=_xwin,
                                              _Hv=_hH.values, _Lv=_hL.values),
                                    use_container_width=True, config={'displayModeBar': False})
                    _gap0 = _hret.index[0].date() if len(_hret) else None
                    st.markdown(
                        f"<div style='color:#8b97a8;font-size:.78rem;line-height:1.5;margin:-.2rem 0 .2rem'>"
                        f"<b>Why the curve starts later than the others:</b> a {_hsel} horizon-return compares each "
                        f"day's value with the value <b>{_hd} trading days earlier</b>, so it can't be measured until "
                        f"one full window has elapsed. The first point is on "
                        f"<b>{_gap0:%d %b %Y}</b> — about {_hsel} after inception. The x-axis still spans the whole "
                        f"period (aligned with the charts above), so this opening gap is expected, not missing data."
                        f"</div>", unsafe_allow_html=True)
                    _risk_status(_hret, f"The {_hsel} return")
                st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<details style='background:rgba(74,158,255,.10);border:1px solid #34527a;border-radius:6px;"
                    "padding:.4rem .8rem;margin:.3rem 0 .2rem;font-size:.82rem'>"
                    "<summary style='cursor:pointer;color:#79b6ff;font-weight:600;list-style:none'>"
                    "✨ AI-powered: What is a horizon-return breach?</summary>"
                    "<div style='color:#aebccd;margin-top:.4rem;line-height:1.55'>"
                    "A <b>horizon-return breach</b> looks at the portfolio's return over a rolling window (the "
                    "<i>horizon</i> you choose) and flags each window whose return fell below your limit <b>H</b>. "
                    "Keeping the breach frequency at or under <b>&alpha;</b> is exactly the framework's constraint "
                    "<b>P(r &lt; H) &le; &alpha;</b> — the same rule the Grid/Scalable optimisers enforce. Unlike "
                    "drawdown it is measured from a fixed start (each window), not the running peak, so a slow grind "
                    "can read as &lsquo;within limit&rsquo; here while still showing up as a drawdown above. Short "
                    "horizons trip on single shocks; longer horizons need more history.</div></details>",
                    unsafe_allow_html=True)
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            if _cap:
                _av = _cap['alpha'] * 100
                _acol = '#26a641' if _av >= 0 else '#f85149'
                st.markdown(
                    "<div style='background:#0d1a2e;border:1px solid #1a3a5c;border-radius:10px;"
                    "padding:.7rem 1.1rem;margin:.2rem 0 .4rem;color:#c0c8d8;font-size:.9rem'>"
                    f"<b style='color:#7fb3e8'>CAPM vs {_PR['bench']}</b> &nbsp;—&nbsp; "
                    f"realised <b>Jensen's α</b> = <b style='color:{_acol}'>{_av:+.2f}%/yr</b>, "
                    f"<b>β</b> = {_cap['beta']:.2f}, <b>R²</b> = {_cap['r2']:.2f} "
                    f"<span style='color:#8b97a8'>(over {_cap['n']} overlapping days)</span>.<br>"
                    "<span style='color:#8b97a8;font-size:.84rem'>α is the return the portfolio earned "
                    "beyond what its benchmark exposure (β) explains; β is its sensitivity to the benchmark; "
                    "R² is how much of its moves the benchmark accounts for.</span></div>",
                    unsafe_allow_html=True)
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Stress test (scenarios & shocks) ──
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<span class="pf-frame-marker"></span>', unsafe_allow_html=True)
                _ICON_STRESS = ("<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#E3C77E' "
                            "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' "
                            "style='vertical-align:-3px;margin-right:.5rem'><path d='M9 3h6'/>"
                            "<path d='M10 3v6l-4.6 8.4A2 2 0 0 0 7.2 21h9.6a2 2 0 0 0 1.8-3.6L14 9V3'/>"
                            "<path d='M7.3 15h9.4'/></svg>")
                st.markdown("<div style='text-align:center'>" + (_GTITLE % (_ICON_STRESS + "Stress test"))
                            + "</div>", unsafe_allow_html=True)
                _curw = (max(_PR['log'], key=lambda e: _pd.Timestamp(e['date']))['weights']
                         if _PR.get('log') else {})
                _Hlim = float(st.session_state.get('pf_risk_lim', -15))
                if not _curw:
                    st.caption("Compute a portfolio to run stress tests.")
                else:
                    _stab = st.tabs(["Historical scenarios", "Custom shock", "Parametric stress"])
                    with _stab[0]:
                        st.caption("Takes each past crisis's daily shock sequence and applies it to your "
                                   "**current** portfolio — i.e. *“what if the same shock hit starting "
                                   f"today?”* — versus your loss limit H = {_Hlim:.0f}%. The charts below "
                                   "project the path forward from today (the calendar is illustrative).")
                        if st.button("Run historical scenarios", type="primary", key="ss_run"):
                            _stk = list(_curw.keys()); _out = []
                            with st.spinner("Fetching crisis-period prices…"):
                                for _sc in _ss.SCENARIOS:
                                    try:
                                        _end1 = (_pd.Timestamp(_sc["end"]) + _pd.Timedelta(days=1)).date()
                                        _spx, _serr = fetch_close_prices(_stk, _sc["start"], _end1)
                                        _r = _ss.replay(_spx, _curw) if (_spx is not None and not _serr) else None
                                    except Exception:
                                        _r = None
                                    _out.append((_sc, _r))
                            st.session_state['ss_results'] = _out
                        for _sc, _r in (st.session_state.get('ss_results') or []):
                            if _r is None:
                                st.markdown(f"<div style='color:#8b97a8;font-size:.85rem;margin:.25rem 0'>"
                                            f"<b style='color:#c9d1d9'>{_sc['name']}</b> — no price history for "
                                            f"your holdings in this window.</div>", unsafe_allow_html=True)
                                continue
                            _breach = (_r['mdd'] * 100.0) < _Hlim
                            _col = "#f85149" if _breach else "#9be9a8"
                            _verdict = ("breaches" if _breach else "within") + f" your {_Hlim:.0f}% limit"
                            _cov = ("" if _r['coverage'] >= 0.999 else
                                    f" · covers {_r['coverage']*100:.0f}% of weight" +
                                    (f" (no data: {', '.join(_r['missing'])})" if _r['missing'] else ""))
                            st.markdown(
                                f"<div style='border:1px solid #27344e;border-radius:8px;padding:.5rem .85rem;"
                                f"margin:.3rem 0;background:#0e1521'><b style='color:#c9d1d9'>{_sc['name']}</b> "
                                f"<span style='color:#8b97a8;font-size:.8rem'>· {_sc['blurb']}</span><br>"
                                f"Hypothetical return <b style='color:{_col}'>{_r['cum']*100:+.1f}%</b> · "
                                f"worst drawdown <b style='color:{_col}'>{_r['mdd']*100:.1f}%</b> "
                                f"<span style='color:{_col}'>({_verdict})</span>"
                                f"<span style='color:#8b97a8;font-size:.78rem'>{_cov}</span></div>",
                                unsafe_allow_html=True)
                        # ── detailed charts for a chosen scenario ──
                        _done = [(_s, _rr) for _s, _rr in (st.session_state.get('ss_results') or [])
                                 if _rr is not None and _rr.get('equity') is not None and len(_rr['equity']) > 3]
                        if _done:
                            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
                            _scmap = {_s['name']: (_s, _rr) for _s, _rr in _done}
                            _selname = st.selectbox("Chart a scenario", list(_scmap.keys()), key="ss_chartsel")
                            _ssc, _sr = _scmap[_selname]
                            # re-base the shock path FORWARD from today (apply the past shock to today's portfolio)
                            _eq0 = _sr['equity']
                            _fwd = _pd.bdate_range(_pd.Timestamp.today().normalize(), periods=len(_eq0))
                            _seq = _pd.Series(_eq0.values, index=_fwd)
                            _swin = [_seq.index[0], _seq.index[-1]] if len(_seq) else None
                            # ── portfolio value (start = 100) over the stressed window ──
                            _vbase = _seq.index[0] - _pd.offsets.BDay(1)
                            _val = _pd.concat([_pd.Series([100.0], index=[_vbase]), 100.0 * _seq])
                            _endv = float(_val.iloc[-1])
                            _vcol = "#f85149" if (_endv < 100.0) else "#9be9a8"
                            # euro value = starting capital × (value / 100), shown on a synced right axis
                            _scap = float(st.session_state.get('pf_capital_inp', 10000.0))
                            _eend = _scap * _endv / 100.0
                            st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.9rem;margin:.2rem 0 .2rem'>"
                                        "Projected portfolio value <span style='color:#8b97a8;font-weight:400'>— "
                                        f"start = 100, ending near <b style='color:{_vcol}'>{_endv:.1f}</b>"
                                        + (f" <b style='color:{_vcol}'>(€{_eend:,.0f})</b>" if _scap > 0 else "")
                                        + " if this shock hit today</span></div>", unsafe_allow_html=True)
                            _vmin = float(_val.min()); _vmax = float(_val.max())
                            _vpad = max((_vmax - _vmin) * 0.06, 1.0)
                            _vlo, _vhi = _vmin - _vpad, _vmax + _vpad
                            _vfig = _go.Figure()
                            _vfig.add_trace(_go.Scatter(x=_val.index, y=_val.values, mode='lines',
                                            line=dict(color='#2dd4bf', width=2.2),
                                            customdata=(_scap * _val.values / 100.0),
                                            hovertemplate='%{x|%Y-%m-%d}<br>value %{y:.1f}'
                                                          + ('<br>€%{customdata:,.0f}' if _scap > 0 else '')
                                                          + '<extra></extra>'))
                            if _scap > 0:   # invisible trace so the euro axis renders, synced to the value line
                                _vfig.add_trace(_go.Scatter(x=_val.index, y=_scap * _val.values / 100.0,
                                                mode='lines', line=dict(width=0), yaxis='y2',
                                                hoverinfo='skip', showlegend=False))
                            _vfig.add_hline(y=100.0, line=dict(color='#5b6675', width=1, dash='dot'))
                            _vfig.update_layout(template='plotly_dark', paper_bgcolor='#1b2330',
                                                plot_bgcolor='#0e1521', height=240, showlegend=False,
                                                margin=dict(t=16, b=30, l=64, r=(70 if _scap > 0 else 24)),
                                                yaxis=dict(title='Value (start = 100)', gridcolor='#27344e',
                                                           griddash='dot', color='#c0c8d8', automargin=False,
                                                           range=[_vlo, _vhi], autorange=False),
                                                yaxis2=dict(title='Portfolio value (€)', overlaying='y',
                                                            side='right', range=[_scap * _vlo / 100.0,
                                                            _scap * _vhi / 100.0], autorange=False, showgrid=False,
                                                            color='#2dd4bf', tickformat=',.0f', automargin=False,
                                                            visible=(_scap > 0)),
                                                xaxis=dict(gridcolor='#27344e', griddash='dot', color='#c0c8d8',
                                                           range=[_vbase, _seq.index[-1]], autorange=False))
                            st.plotly_chart(_vfig, use_container_width=True, config={'displayModeBar': False})
                            st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.9rem;margin:.5rem 0 .2rem'>"
                                        "Projected drawdown vs limit <span style='color:#8b97a8;font-weight:400'>"
                                        "— if this shock hit your portfolio starting today</span></div>",
                                        unsafe_allow_html=True)
                            _sdd = (_seq / _seq.cummax() - 1.0) * 100.0
                            st.plotly_chart(_risk_fig(_sdd.index, _sdd.values, 'Drawdown (%)', _xrange=_swin),
                                            use_container_width=True, config={'displayModeBar': False})
                            _nwin = len(_seq)
                            _hopt = [_hh for _hh in (5, 10, 21, 63) if _hh < _nwin - 1]
                            if _hopt:
                                _hh = _hopt[-1]
                                _hr = (_seq / _seq.shift(_hh) - 1.0) * 100.0
                                _hr = _hr.dropna()
                                if len(_hr) >= 3:
                                    st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.9rem;"
                                                "margin:.5rem 0 .2rem'>Projected horizon-return vs limit "
                                                f"<span style='color:#8b97a8;font-weight:400'>— rolling {_hh}-day "
                                                "window, from today</span></div>", unsafe_allow_html=True)
                                    st.plotly_chart(_risk_fig(_hr.index, _hr.values, f'{_hh}d return (%)', _xrange=_swin),
                                                    use_container_width=True, config={'displayModeBar': False})
                            else:
                                st.caption("This window is too short for a horizon-return chart.")
                            st.caption("Dates are projected forward from today for illustration — the shape is "
                                       "the crisis's actual daily moves applied to your current holdings.")
                    with _stab[1]:
                        _beta = (_cap.get('beta') if _cap else None)
                        st.markdown("**Market shock** — an instantaneous market move, propagated via your β.")
                        _msh = st.number_input("Market move (%)", min_value=-60, max_value=60, value=-20,
                                               step=1, key="ss_mkt")
                        _mi = _ss.market_shock(_beta, _msh)
                        if _mi is None:
                            st.caption("Set a benchmark and compute analytics to get β for the market shock.")
                        else:
                            _mc = "#f85149" if (_mi * 100) < _Hlim else ("#e8cd84" if _mi < 0 else "#9be9a8")
                            st.markdown(f"β = {_beta:.2f} → estimated portfolio impact "
                                        f"<b style='color:{_mc}'>{_mi*100:+.1f}%</b> "
                                        f"<span style='color:#8b97a8;font-size:.78rem'>(first-order; ignores α "
                                        f"and stock-specific risk)</span>", unsafe_allow_html=True)
                        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                        st.markdown("**Per-asset shock** — set a % move for each holding.")
                        _shdf = _pd.DataFrame([{"Ticker": _t, "Shock %": 0.0} for _t in _curw])
                        _shed = st.data_editor(_shdf, hide_index=True, use_container_width=True, key="ss_assets",
                                               column_config={
                                                   "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                                                   "Shock %": st.column_config.NumberColumn("Shock %", min_value=-90,
                                                              max_value=90, step=1, format="%.0f")})
                        _shocks = {_rr["Ticker"]: _rr["Shock %"] for _, _rr in _shed.iterrows()}
                        _ai = _ss.asset_shock(_curw, _shocks)
                        _ac = "#f85149" if (_ai * 100) < _Hlim else ("#e8cd84" if _ai < 0 else "#9be9a8")
                        st.markdown(f"Weighted portfolio impact <b style='color:{_ac}'>{_ai*100:+.1f}%</b>",
                                    unsafe_allow_html=True)
                        # ── Shock path over time (β-projected trajectory) ──
                        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                        st.markdown("**Shock path over time** — a custom market trajectory, projected via β.")
                        if _beta is None:
                            st.caption("Set a benchmark and compute analytics to get β for the path projection.")
                        else:
                            st.caption("Each leg is a market move that compounds evenly over its days, applied "
                                       "through your β. The shape is your assumption (unlike the historical "
                                       "scenarios' real day-by-day sequences).")
                            if 'ss_path_data' not in st.session_state:
                                st.session_state['ss_path_data'] = _pd.DataFrame(
                                    [{"Days": 20, "Market move %": -20.0}, {"Days": 10, "Market move %": 5.0}])
                            _ped = st.data_editor(
                                st.session_state['ss_path_data'], num_rows="dynamic", hide_index=True,
                                use_container_width=True, key="ss_path_ed",
                                column_config={
                                    "Days": st.column_config.NumberColumn("Days", min_value=1, max_value=500,
                                                                          step=1, format="%d", help="Trading days this leg lasts."),
                                    "Market move %": st.column_config.NumberColumn("Market move %", min_value=-90,
                                                     max_value=90, step=1, format="%.1f",
                                                     help="Total market move over the leg (compounds evenly across its days).")})
                            _mdaily = []
                            for _, _lr in _ped.iterrows():
                                _dn = _lr.get("Days"); _mv = _lr.get("Market move %")
                                if _pd.isna(_dn) or _pd.isna(_mv):
                                    continue
                                _dn = int(_dn)
                                if _dn <= 0:
                                    continue
                                _rd = (1.0 + float(_mv) / 100.0) ** (1.0 / _dn) - 1.0
                                _mdaily += [_rd] * _dn
                            if len(_mdaily) >= 2:
                                _peq = np.cumprod(1.0 + np.array(_mdaily) * float(_beta))
                                _pidx = _pd.bdate_range(_pd.Timestamp.today().normalize(), periods=len(_peq))
                                _peqs = _pd.Series(_peq, index=_pidx)
                                _pwin = [_peqs.index[0], _peqs.index[-1]]
                                _ptot = float(_peq[-1] - 1.0)
                                _pdd = (_peqs / _peqs.cummax() - 1.0) * 100.0
                                _pmdd = float(_pdd.min())
                                _pc = "#f85149" if _pmdd < _Hlim else ("#e8cd84" if _ptot < 0 else "#9be9a8")
                                st.markdown(f"Projected total impact <b style='color:{_pc}'>{_ptot*100:+.1f}%</b> · "
                                            f"worst drawdown <b style='color:{_pc}'>{_pmdd:.1f}%</b> "
                                            f"<span style='color:#8b97a8;font-size:.8rem'>(over {len(_peq)} trading "
                                            f"days · β = {_beta:.2f})</span>", unsafe_allow_html=True)
                                st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.9rem;margin:.3rem 0 .2rem'>"
                                            "Projected drawdown vs limit</div>", unsafe_allow_html=True)
                                st.plotly_chart(_risk_fig(_pdd.index, _pdd.values, 'Drawdown (%)', _xrange=_pwin),
                                                use_container_width=True, config={'displayModeBar': False})
                                _pn = len(_peqs); _phopt = [_hh for _hh in (5, 10, 21, 63) if _hh < _pn - 1]
                                if _phopt:
                                    _hh2 = _phopt[-1]
                                    _phr = (_peqs / _peqs.shift(_hh2) - 1.0) * 100.0
                                    _phr = _phr.dropna()
                                    if len(_phr) >= 3:
                                        st.markdown("<div style='color:#cfd8e3;font-weight:600;font-size:.9rem;"
                                                    "margin:.5rem 0 .2rem'>Projected horizon-return vs limit "
                                                    f"<span style='color:#8b97a8;font-weight:400'>— rolling {_hh2}-day "
                                                    "window</span></div>", unsafe_allow_html=True)
                                        st.plotly_chart(_risk_fig(_phr.index, _phr.values, f'{_hh2}d return (%)', _xrange=_pwin),
                                                        use_container_width=True, config={'displayModeBar': False})
                            else:
                                st.caption("Add at least one leg (days + move) to project a path.")
                    with _stab[2]:
                        _mom = _PR.get('moments'); _mtk = _PR.get('moment_tickers')
                        if not _mom or not _mtk:
                            st.caption("Compute analytics first — parametric stress uses the holdings' covariance.")
                        else:
                            _wv = np.array([float(_curw.get(t, 0.0)) for t in _mtk], float)
                            if _wv.sum() > 0:
                                _wv = _wv / _wv.sum()
                            _pc1, _pc2 = st.columns(2)
                            with _pc1:
                                _vm = st.slider("Volatility ×", 1.0, 3.0, 1.5, 0.1, key="ss_vol")
                            with _pc2:
                                _cl = st.slider("Correlation → 1 (%)", 0, 100, 50, 5, key="ss_corr") / 100.0
                            _ps = _ss.parametric(_wv, _mom[1], vol_mult=_vm, corr_lambda=_cl, conf_pct=95.0)
                            _e1, _e2 = st.columns(2)
                            _e1.markdown("<div style='color:#9be9a8;font-size:1.05rem;line-height:1.5'>Base &nbsp;— vol "
                                         f"<b>{_ps['base_sigma']*100:.1f}%</b> · 95% VaR "
                                         f"<b>{_ps['base_var']*100:.1f}%</b></div>", unsafe_allow_html=True)
                            _e2.markdown("<div style='color:#ffb4ae;font-size:1.05rem;line-height:1.5'>Stressed — vol "
                                         f"<b>{_ps['str_sigma']*100:.1f}%</b> · 95% VaR "
                                         f"<b>{_ps['str_var']*100:.1f}%</b></div>", unsafe_allow_html=True)
                            st.caption("Annualised. Volatilities scaled and correlations blended toward 1, then "
                                       "VaR = z·σ re-derived — shows how diversification erodes and tail loss grows "
                                       "in a stressed regime.")
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # rebalance-log summary
            _seg = []
            for _e in _PR['log']:
                _parts = " · ".join(f"{_tk} {_w*100:.0f}%" for _tk, _w in _e['weights'].items())
                _seg.append(f"<b style='color:#c9d1d9'>{_e['date']}</b> — {_parts}")
            st.markdown("<div style='color:#8b97a8;font-size:.84rem;line-height:1.7;margin:.1rem 0 .2rem'>"
                        "Rebalance log:<br>" + "<br>".join(_seg) + "</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    # Gold + dark Browse-files button (matches the other gold action buttons), portfolio view only.
    st.markdown(
        "<style>section[data-testid='stMain'] [data-testid='stFileUploaderDropzone'] button,"
        "section[data-testid='stMain'] [data-testid='stFileUploaderDropzone'] button *"
        "{background:#E3C77E !important;color:#0d1117 !important;border:none !important}"
        "section[data-testid='stMain'] [data-testid='stFileUploaderDropzone'] button"
        "{width:20% !important;min-width:170px !important;flex:none !important}"
        "section[data-testid='stMain'] [data-testid='stFileUploaderDropzone'] button:hover,"
        "section[data-testid='stMain'] [data-testid='stFileUploaderDropzone'] button:hover *"
        "{background:#edd596 !important;color:#0d1117 !important}</style>", unsafe_allow_html=True)
    st.markdown(
        "<style>section[data-testid='stMain'] [data-testid='stExpander']:has(.bmv-saveload-mk) summary,"
        "section[data-testid='stMain'] [data-testid='stExpander']:has(.bmv-saveload-mk) summary *"
        "{font-size:20px !important;font-weight:600 !important;color:#E3C77E !important}"
        "section[data-testid='stMain'] [data-testid='stExpander']:has(.bmv-saveload-mk) summary"
        "{justify-content:center !important}</style>",
        unsafe_allow_html=True)
    with st.expander("Save / load this portfolio", expanded=True, icon=":material/save:"):
        st.markdown("<span class='bmv-saveload-mk'></span>", unsafe_allow_html=True)
        # Portfolio object used by both the Drive and JSON save paths.
        _obj_now = None
        if _PR:
            _obj_now = {"name": _PR['name'], "benchmark": _PR['bench'],
                        "starting_capital": float(st.session_state.get('pf_capital_inp', 10000.0)),
                        "rebalances": _PR.get('sec_log') or _PR['log'],
                        "derivatives": _PR.get('der_rows', []),
                        "risk": {"loss_limit_pct": int(st.session_state.get('pf_risk_lim', -15)),
                                 "alpha_pct": int(st.session_state.get('pf_risk_alpha', 5)),
                                 "cvar_floor_pct": int(st.session_state.get('pf_risk_L', -20)),
                                 "method": str(st.session_state.get('pf_risk_method', 'VaR'))},
                        "risk_profile": st.session_state.get('rp_result'),
                        "risk_profile_history": st.session_state.get('rp_history', []),
                        "tolerance_timeline": st.session_state.get('tol_timeline', [])}

        def _apply_portfolio_obj(_obj, _src):
            _rows2 = []
            for _e in _obj.get("rebalances", []):
                for _tk, _w in (_e.get("weights") or {}).items():
                    _rows2.append({"Date": _pd.Timestamp(_e["date"]).date(),
                                   "Ticker": _tk, "Weight %": round(float(_w) * 100, 2)})
            _der2 = _obj.get("derivatives") or []
            if not _rows2 and not _der2:
                st.warning("That portfolio has no positions to load.")
                return
            st.session_state['pf_table_data'] = _pd.DataFrame(
                _rows2 if _rows2 else {"Date": [], "Ticker": [], "Weight %": []})
            # restore the derivatives table (or clear it)
            if _der2:
                st.session_state['pf_der_data'] = _pd.DataFrame([
                    {"Date": _pd.Timestamp(_r.get("Date")).date(), "Underlying": _r.get("Underlying"),
                     "Type": _r.get("Type"), "Strike 1": _r.get("Strike 1"), "Strike 2": _r.get("Strike 2"),
                     "Tenor (y)": _r.get("Tenor (y)"), "Weight %": _r.get("Weight %")} for _r in _der2])
            else:
                st.session_state['pf_der_data'] = _pd.DataFrame({
                    "Date": _pd.Series([], dtype="datetime64[ns]"),
                    "Underlying": _pd.Series([], dtype="object"), "Type": _pd.Series([], dtype="object"),
                    "Strike 1": _pd.Series([], dtype="float64"), "Strike 2": _pd.Series([], dtype="float64"),
                    "Tenor (y)": _pd.Series([], dtype="float64"), "Weight %": _pd.Series([], dtype="float64")})
            st.session_state.pop('pf_der_ed', None)
            st.session_state['pf_name_inp'] = _obj.get("name", "My portfolio")
            _capj = _obj.get("starting_capital")
            if _capj is not None:
                try:
                    st.session_state['pf_capital_inp'] = float(_capj)
                except (TypeError, ValueError):
                    pass
            st.session_state['pf_bench_sel'] = "Custom ticker…"
            st.session_state['pf_bench_custom'] = (_obj.get("benchmark", "SPY") or "SPY")
            _rk = _obj.get("risk") or {}
            if _rk.get("loss_limit_pct") is not None:
                st.session_state['pf_risk_lim'] = int(_rk["loss_limit_pct"])
            if _rk.get("alpha_pct") is not None:
                st.session_state['pf_risk_alpha'] = int(_rk["alpha_pct"])
            if _rk.get("cvar_floor_pct") is not None:
                st.session_state['pf_risk_L'] = int(_rk["cvar_floor_pct"])
            if _rk.get("method") in ("VaR", "ES"):
                st.session_state['pf_risk_method'] = _rk["method"]
            _rpf = _obj.get("risk_profile")
            if isinstance(_rpf, dict) and _rpf.get("score") is not None:
                st.session_state['rp_result'] = _rpf
            _rph2 = _obj.get("risk_profile_history")
            if isinstance(_rph2, list) and _rph2:
                st.session_state['rp_history'] = _rph2
            _tlj = _obj.get("tolerance_timeline")
            if isinstance(_tlj, list):
                st.session_state['tol_timeline'] = _tlj
            else:
                st.session_state.pop('tol_timeline', None)
            for _k in ('tol_editor', 'pf_table_ed', 'pf_result', 'pf_view_range'):
                st.session_state.pop(_k, None)
            st.success("Loaded from %s — review the holdings above and click Compute analytics." % _src)
            st.rerun()

        # ── Google Drive (one-click, saves to the user's own Drive) ──
        if _gcfg:
            if st.session_state.get("gd_error"):
                st.warning(st.session_state.pop("gd_error"))
            _gtok = _gd_token()
            if not _gtok:
                if "gd_authurl" not in st.session_state:
                    _stt = _gd.new_state(); _ver, _chal = _gd.make_pkce()
                    _gd.pending_put(_stt, _ver, "portfolio")
                    st.session_state["gd_authurl"] = _gd.build_auth_url(
                        _gcfg["client_id"], _gcfg["redirect_uri"], _stt, _chal)
                st.markdown("**Save to your Google Drive** — sign in once; portfolios are stored in "
                            "*your own* Drive. The app uses the least-privilege `drive.file` scope, so it "
                            "can only ever see or change files it created — never the rest of your Drive.")
                _sgc = st.columns([2, 1, 2])
                with _sgc[1]:
                    st.link_button("🔐  Sign in with Google", st.session_state["gd_authurl"],
                                   type="primary", use_container_width=True)
                st.caption("Tip: sign in first, then load or build — signing in reloads the page.")
            else:
                _guser = st.session_state.get("gd_user") or {}
                _who = _guser.get("email") or _guser.get("name") or "your Google account"
                st.markdown("Signed in as **%s** &nbsp;·&nbsp; portfolios save to your Drive." % _who,
                            unsafe_allow_html=True)
                if _obj_now is not None:
                    _gsv = st.columns([2, 1, 2])
                    with _gsv[1]:
                        if st.button("⬆  Save to my Drive", use_container_width=True, type="primary",
                                     key="gd_save"):
                            try:
                                _ex = {f["name"]: f["id"] for f in _gd.list_portfolios(_gtok)}
                                _fid = _ex.get((_PR['name'] or 'portfolio').strip() + ".bmv.json")
                                _gd.save_portfolio(_gtok, _PR['name'], _obj_now, file_id=_fid)
                                st.success("Saved to your Drive%s." % (" (updated)" if _fid else ""))
                            except Exception as _e:
                                st.error("⚠ Couldn't save to your Drive — %s" % _gd_friendly(_e))
                else:
                    st.caption("Compute a portfolio above to enable saving it to your Drive.")
                _gso = st.columns([2, 1, 2])
                with _gso[1]:
                    if st.button("↪  Sign out", use_container_width=True, type="primary", key="gd_signout"):
                        for _k in ("gd_tokens", "gd_user", "gd_authurl"):
                            st.session_state.pop(_k, None)
                        st.rerun()
                try:
                    _files = _gd.list_portfolios(_gtok)
                except Exception as _e:
                    _files = []
                    st.caption("⚠ Couldn't list your Drive portfolios — %s" % _gd_friendly(_e))
                if _files:
                    _lbl = {("%s  ·  %s" % (f["name"].replace(".bmv.json", ""),
                                            f.get("modifiedTime", "")[:10])): f["id"] for f in _files}
                    _sel = st.selectbox("Load from your Drive", list(_lbl.keys()), key="gd_loadsel")
                    _gld = st.columns([2, 1, 2])
                    with _gld[1]:
                        if st.button("⬇  Load selected from Drive", use_container_width=True,
                                     type="primary", key="gd_load"):
                            try:
                                _apply_portfolio_obj(_gd.load_portfolio(_gtok, _lbl[_sel]), "Google Drive")
                            except Exception as _e:
                                st.error("⚠ Couldn't load from your Drive — %s" % _gd_friendly(_e))
                else:
                    st.caption("No saved portfolios in your Drive yet — compute one, then “Save to my Drive”.")
            st.markdown("<hr style='border-color:#26303f;margin:.7rem 0 .5rem'>", unsafe_allow_html=True)
        else:
            st.caption("☁ Google Drive sync isn't configured on this deployment — use the file "
                       "download/upload below. (Admin: add a `[google_oauth]` block to Streamlit secrets — see GOOGLE_DRIVE_SETUP.md in the repo for the full procedure.)")

        # ── Manual JSON (offline fallback) ──
        if _obj_now is not None:
            _dlc = st.columns([2, 1, 2])
            with _dlc[1]:
                st.download_button("⬇  Download portfolio (JSON)", _json.dumps(_obj_now, indent=2),
                                   file_name=(_PR['name'] or 'portfolio').replace(' ', '_') + ".json",
                                   mime="application/json", type="primary", use_container_width=True)
        _up = st.file_uploader("Load a portfolio file (JSON)", type=["json"], key="pf_up")
        if _up is not None:
            try:
                _apply_portfolio_obj(_json.load(_up), "file")
            except Exception as _e:
                st.error(f"Couldn't read that file: {_e}")
        st.caption("Each saved portfolio (Drive or file) bundles your holdings, benchmark, risk settings "
                   "(method, H, α, L), your Risk Profile with its dated assessment history, and your "
                   "editable tolerance timeline.")

elif _view == "riskprofile":
    import datetime as _dt
    with st.container():
        _bb_l, _bb_mid, _bb_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _bb_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _bb_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em">Risk Profile — Grable-Lytton Questionnaire</h2></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:flex-start;gap:.6rem;border:1px solid rgba(231,236,244,0.2);'
        'border-radius:.5rem;padding:.7rem .95rem;margin:.2rem 0 .8rem;color:#c0c8d8;font-size:.9rem;line-height:1.55">'
        '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" style="flex:none;margin-top:3px">'
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
        '<span>Answer 13 short questions from the validated <strong>Grable &amp; Lytton (1999)</strong> financial '
        'risk-tolerance scale. Your score maps to a tolerance band that <strong>sets your simulation’s risk '
        'parameters</strong> — the threshold H and shortfall probability &alpha; for the <strong>Grid</strong> '
        'engine, or the CVaR floor L and &alpha; for the <strong>Scalable</strong> engine (built for many '
        'assets). Apply the result to either, and you can still adjust the values there.</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:flex-start;gap:.5rem;background:rgba(245,185,66,.08);'
        'border:1px solid #8a6d2b;border-radius:6px;padding:.5rem .9rem;margin:0 0 1.2rem;'
        'color:#d9c79a;font-size:.82rem">'
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e0b84a" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" style="flex:none;margin-top:1px">'
        '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        '<span><b>Ceci ne constitue pas un conseil en investissement.</b> This questionnaire sets your '
        'simulation’s risk parameters only. It is an educational tool, not a regulated suitability '
        'assessment and not investment advice.</span></div>', unsafe_allow_html=True)

    st.markdown('<h3 style="color:#E3C77E;scroll-margin-top:90px;margin:.2rem 0 .5rem">'
                'Risk-Tolerance Assessment</h3>', unsafe_allow_html=True)
    with st.form("rp_form"):
        _rp_choices = []
        for _i, _item in enumerate(_rp.GL_ITEMS):
            _opts = _item["options"]
            # Escape '$' so Streamlit's markdown doesn't read paired $...$ as LaTeX math;
            # score by option index so the displayed (escaped) text never breaks the lookup.
            _q = _item["q"].replace("$", "\\$")
            _sel = st.radio(
                f"**{_i + 1}. {_q}**",
                options=list(range(len(_opts))),
                format_func=lambda _k, _o=_opts: _o[_k][0].replace("$", "\\$"),
                index=None, key=f"rp_q{_i}")
            _rp_choices.append(_opts[_sel][0] if _sel is not None else None)
            st.markdown("<div style='height:.2rem'></div>", unsafe_allow_html=True)
        _rp_submit = st.form_submit_button("Calculate my risk profile", type="primary")

    if _rp_submit:
        _n_missing = sum(c is None for c in _rp_choices)
        if _n_missing:
            st.error(f"Please answer all 13 questions — {_n_missing} still unanswered.")
            st.session_state.pop("rp_result", None)
        else:
            _rpres = _rp.profile(_rp_choices)
            _rpres["date"] = _dt.date.today().strftime("%Y-%m-%d")
            st.session_state["rp_result"] = _rpres
            # Dated risk-tolerance history (newest = active). A retake makes the new
            # result active but keeps prior assessments timestamped (nothing overwritten).
            _entry = {k: _rpres[k] for k in ("date", "score", "band", "H_pct", "alpha_pct", "L_pct")}
            _hist = list(st.session_state.get("rp_history", []))
            _last = _hist[-1] if _hist else None
            if (_last is None) or any(_last.get(_k) != _entry[_k]
                                      for _k in ("score", "H_pct", "alpha_pct", "L_pct")):
                _hist.append(_entry)
            else:
                _hist[-1] = _entry  # same result retaken — just refresh the date
            st.session_state["rp_history"] = _hist
            # Sync the operational tolerance timeline: add this assessment's level (de-duped).
            _tl = list(st.session_state.get("tol_timeline", []))
            _trow = {"date": _entry["date"], "H_pct": _entry["H_pct"],
                     "alpha_pct": _entry["alpha_pct"], "L_pct": _entry["L_pct"]}
            if not any((str(r.get("date")) == _trow["date"] and r.get("H_pct") == _trow["H_pct"]
                        and r.get("alpha_pct") == _trow["alpha_pct"] and r.get("L_pct") == _trow["L_pct"])
                       for r in _tl):
                _tl.append(_trow)
                st.session_state["tol_timeline"] = _tl
                st.session_state.pop("tol_editor", None)  # let the editor re-seed with the new row

    _rp_res = st.session_state.get("rp_result")
    if _rp_res:
        _chip = ("background:#0e1521;border:1px solid #C9A24B;border-radius:999px;"
                 "padding:.3rem .9rem;font-size:.85rem;color:#c0c8d8")
        _chipb = ("background:#0e1521;border:1px solid #34527a;border-radius:999px;"
                  "padding:.3rem .9rem;font-size:.82rem;color:#c0c8d8")
        _gauge = _rp_gauge_svg(_rp_res["score"], _rp_res["band"])
        # Put the frame AND the buttons in one shared middle column so the frame's
        # left/right edges line up exactly with the two buttons below it.
        _rp_csl, _rp_card, _rp_csr = st.columns([5, 8, 5])
        with _rp_card:
            st.markdown(
                '<div style="background:#1b2330;border:1px solid #C9A24B;border-radius:10px;'
                'padding:1rem 1.3rem;margin:.5rem 0 .3rem">'
                '<div style="color:#E3C77E;font-weight:700;font-size:1.05rem;margin-bottom:.4rem;text-align:center">'
                'Your risk profile</div>'
                + _gauge +
                '<div style="display:flex;flex-wrap:wrap;gap:.55rem;justify-content:center;margin-top:.5rem">'
                f'<span style="{_chip}">Score <b style="color:#E3C77E">{_rp_res["score"]}</b> / 47</span>'
                f'<span style="{_chip}">Tolerance: <b style="color:#E3C77E">{_rp_res["band"]}</b></span>'
                f'<span style="{_chip}">Shortfall &alpha; <b style="color:#E3C77E">{_rp_res["alpha_pct"]}%</b></span>'
                '</div>'
                '<div style="display:flex;flex-wrap:wrap;gap:.55rem;justify-content:center;margin-top:.55rem">'
                f'<span style="{_chipb}">Grid optimiser — Threshold H <b style="color:#4a9eff">{_rp_res["H_pct"]}%</b></span>'
                f'<span style="{_chipb}">Scalable optimiser — CVaR floor L <b style="color:#4a9eff">{_rp_res["L_pct"]}%</b></span>'
                '</div>'
                f'<div style="color:#c0c8d8;font-size:.88rem;margin-top:.75rem;text-align:center;line-height:1.55">'
                f'{_rp_res["blurb"]}</div></div>', unsafe_allow_html=True)

            # ── Plain-language meaning (mental accounting) — collapsible, under the gauge ──
            _Hn = abs(float(_rp_res["H_pct"])); _an = float(_rp_res["alpha_pct"]); _Ln = abs(float(_rp_res["L_pct"]))
            _band_l = str(_rp_res["band"]).lower()
            st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
            with st.expander("💬 What your profile means — in plain language", expanded=False):
                st.markdown(
                    '<div style="color:#c0c8d8;font-size:.9rem;line-height:1.62">'
                    '<p style="margin:.1rem 0 .7rem"><b style="color:#E3C77E">Mental accounting</b> is the idea that '
                    'you don\'t treat your wealth as one undifferentiated pot — you set aside a layer you are '
                    'determined to protect, and your <b>risk tolerance</b> decides how strict that protection is. '
                    f'Your answers placed you in the <b style="color:#E3C77E">{_band_l}</b> band, which the app turns '
                    'into two concrete rules the optimisers must obey:</p>'
                    '<p style="margin:.1rem 0 .6rem"><b style="color:#4a9eff">In the Grid optimiser</b> — threshold '
                    f'<b style="color:#4a9eff">H = {_rp_res["H_pct"]}%</b>, shortfall <b style="color:#4a9eff">&alpha; = {_an:.0f}%</b>.<br>'
                    'In plain words: <i>&ldquo;Build me the highest-returning portfolio you can, but keep the chance of '
                    f'losing more than {_Hn:.0f}% over the horizon at or below {_an:.0f}%.&rdquo;</i> A more cautious '
                    'profile uses a shallower H or a smaller &alpha; (you protect more of the layer); a bolder profile '
                    'uses a deeper H or a larger &alpha; (you accept more downside in exchange for reaching for return).</p>'
                    '<p style="margin:.1rem 0 .6rem"><b style="color:#f5b942">In the Scalable optimiser</b> — CVaR floor '
                    f'<b style="color:#f5b942">L = {_rp_res["L_pct"]}%</b>.<br>'
                    'In plain words: <i>&ldquo;In the worst outcomes, hold the <u>average</u> loss no deeper than '
                    f'{_Ln:.0f}%.&rdquo;</i> Where &alpha; limits how <i>often</i> you cross the line, L limits how '
                    '<i>bad</i> the tail gets when you do — it controls the severity of the rare, painful scenarios.</p>'
                    '<p style="margin:.1rem 0 0;color:#8b97a8;font-size:.84rem">Both rules describe the same protected '
                    'layer from two angles — the <i>probability</i> of a shortfall (Grid) and the <i>depth</i> of a '
                    'shortfall (Scalable). The optimiser then earns the most it can <i>without</i> breaking the rule '
                    'your profile set.</p>'
                    '</div>', unsafe_allow_html=True)

            # ── Score → parameters mapping table (golden About-style), under the gauge ──
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            st.markdown('<div style="text-align:center;color:#E3C77E;font-size:.9rem;font-weight:700;'
                        'margin:.1rem 0 .3rem">How your score maps to the simulation parameters</div>',
                        unsafe_allow_html=True)
            _rp_tbl = [("13–18", "Low", "Low"), ("19–22", "Below-avg", "Below-average"),
                       ("23–28", "Average", "Average / moderate"), ("29–32", "Above-avg", "Above-average"),
                       ("33–47", "High", "High")]
            _hc = "background:#141a23;color:#E3C77E;font-weight:700;text-align:center;padding:.4rem .5rem;border:1px solid #C9A24B"
            _dc = "background:#0d1117;color:#c9d1d9;padding:.36rem .5rem;border:1px solid #30363d;text-align:center"
            _ac = "background:#2a2410;color:#E3C77E;font-weight:700;padding:.36rem .5rem;border:1px solid #C9A24B;text-align:center"
            _hdr = "".join(f'<td style="{_hc}">{h}</td>' for h in
                           ["Score", "Tolerance", "Threshold H", "Shortfall &alpha;", "CVaR floor L"])
            _bodyr = ""
            for _sc, _shrt, _full in _rp_tbl:
                _Hh, _aa = _rp.BAND_TO_HALPHA[_full]
                _Ll = _rp.BAND_TO_L[_full]
                _cc = _ac if _full == _rp_res["band"] else _dc
                _bodyr += "<tr>" + "".join(f'<td style="{_cc}">{v}</td>'
                                           for v in [_sc, _shrt, f"{_Hh}%", f"{_aa}%", f"{_Ll}%"]) + "</tr>"
            st.html(f'<table style="width:100%;border-collapse:collapse;font-size:.8rem;margin:.1rem 0">'
                    f'<tbody><tr>{_hdr}</tr>{_bodyr}</tbody></table>')
            st.markdown('<div style="text-align:center;color:#8b97a8;font-size:.78rem;margin:.4rem 0 0;'
                        'line-height:1.5">Each band also implies a risk-aversion &lambda; (the mean-variance '
                        'equivalent). Because it depends on your data, it\'s shown live in the Grid optimiser; '
                        'the Scalable engine optimises CVaR directly via a linear program, so no &lambda; '
                        'applies there.</div>', unsafe_allow_html=True)

            # ── Dated assessment history (newest = active; retakes are kept, not overwritten) ──
            _rp_hist = st.session_state.get("rp_history", [])
            if len(_rp_hist) > 1:
                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
                with st.expander(f"🕓 Your assessment history ({len(_rp_hist)} taken)", expanded=False):
                    _hh = "".join(f'<td style="{_hc}">{h}</td>'
                                  for h in ["Date", "Score", "Tolerance", "H", "&alpha;", "L"])
                    _rows = ""
                    for _i, _h in enumerate(reversed(_rp_hist)):
                        _lab = (" &nbsp;<span style='color:#9be9a8;font-weight:700'>active</span>"
                                if _i == 0 else "")
                        _cc2 = _ac if _i == 0 else _dc
                        _rows += ("<tr>" + f'<td style="{_cc2}">{_h.get("date","—")}{_lab}</td>'
                                  + "".join(f'<td style="{_cc2}">{v}</td>' for v in
                                            [f'{_h.get("score","—")}/47', _h.get("band", "—"),
                                             f'{_h.get("H_pct","—")}%', f'{_h.get("alpha_pct","—")}%',
                                             f'{_h.get("L_pct","—")}%']) + "</tr>")
                    st.html(f'<table style="width:100%;border-collapse:collapse;font-size:.8rem;margin:.1rem 0">'
                            f'<tbody><tr>{_hh}</tr>{_rows}</tbody></table>')
                    st.caption("Your newest assessment is active and seeds the optimisers and Live Portfolio. "
                               "Retaking the questionnaire adds a new dated row — earlier ones are kept, "
                               "not overwritten.")

            st.markdown("<div style='height:2.4rem'></div>", unsafe_allow_html=True)  # space above the buttons
            # Buttons nested [3,2,3] inside the same column → outer edges match the frame.
            _rp_b1, _rp_gap, _rp_b2 = st.columns([3, 2, 3])
            with _rp_b1:
                st.button("Open in Grid optimiser  →", type="primary", use_container_width=True,
                          key="rp_apply_grid", on_click=_apply_risk_profile,
                          args=("grid", _rp_res["H_pct"], _rp_res["alpha_pct"], _rp_res["L_pct"]))
            with _rp_b2:
                st.button("Open in Scalable optimiser  →", type="primary", use_container_width=True,
                          key="rp_apply_scalable", on_click=_apply_risk_profile,
                          args=("scalable", _rp_res["H_pct"], _rp_res["alpha_pct"], _rp_res["L_pct"]))
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)  # space below the buttons
            st.caption("Each button pre-fills that engine’s sliders — you can still change them there. "
                       "The Grid engine is exact but suits small portfolios; the Scalable (Monte-Carlo + CVaR) "
                       "engine scales to larger portfolios with many assets.")

    st.markdown(
        '<div style="color:#6b7686;font-size:.74rem;margin-top:1.4rem;line-height:1.5">'
        'Source: Grable, J. E., &amp; Lytton, R. H. (1999). Financial risk tolerance revisited: the development '
        'of a risk assessment instrument. <i>Financial Services Review</i>, 8, 163–181.</div>',
        unsafe_allow_html=True)

elif _view == "ticker":
    with st.container():
        _tk_l, _tk_mid, _tk_x = st.columns([1, 4.2, 1], vertical_alignment="center")
        with _tk_l:
            st.button(":material/home: Back to Main Screen", key="_nav_back", use_container_width=True, on_click=_go_home)
        with _tk_mid:
            st.markdown('<style>section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2){position:sticky;top:60px;z-index:1000;background:#0d1117;border-bottom:1px solid #2a3340;box-shadow:0 8px 16px -10px rgba(0,0,0,.75);padding:.3rem 0 .85rem;margin-bottom:.7rem}section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.bmv-banner):has(h2) div[data-testid="stVerticalBlock"]{gap:.5rem!important}section[data-testid="stMain"] [data-testid="stMainBlockContainer"]{padding-top:3.75rem!important}section[data-testid="stMain"] div[data-testid="stVerticalBlock"]>div[data-testid="stElementContainer"]:has(~ div[data-testid="stVerticalBlockBorderWrapper"] .bmv-banner){display:none}</style><div class="bmv-banner" style="display:flex;align-items:center;justify-content:center;gap:14px;margin:0"><div style="width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#E3C77E,#C9A24B);color:#1a1205;font-weight:700;font-family:Georgia,serif;font-size:1.35rem">&beta;</div><div style="text-align:left"><div style="font-size:.8rem;font-weight:600;letter-spacing:.01em;color:#c9d1d9">Portfolio Optimisation <span style="color:#E3C77E;font-style:italic">with</span> Derivatives &amp; Structured Products</div><div style="font-family:Georgia,serif;font-weight:600;font-size:1.45rem;line-height:1.05;color:#fafafa">Beyond <span style="color:#E3C77E">Mean-Variance</span></div><div style="font-family:Georgia,serif;font-weight:500;font-size:1rem;color:#aeb9c9">Mental Accounting Framework</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#141a23;border:1px solid #C9A24B;border-radius:8px;padding:.12rem 1.2rem;margin:.85rem auto .4rem;max-width:calc(100% - 570px);text-align:center"><h2 style="color:#E3C77E;margin:0;font-family:Georgia,serif;font-size:1.55rem;letter-spacing:.05em">Ticker Analytics</h2></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:flex-start;gap:.6rem;border:1px solid rgba(231,236,244,0.2);'
        'border-radius:.5rem;padding:.7rem .95rem;margin:.2rem 0 .8rem;color:#c0c8d8;font-size:.9rem;line-height:1.55">'
        '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" style="flex:none;margin-top:3px"><circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
        '<span>Enter any stock, ETF or index ticker to see its key figures and CFA-style ratios — '
        '<strong>valuation, profitability, leverage and risk</strong> — each with a plain-language explanation. '
        'These are educational characteristics only: the tool never scores, ranks or recommends an instrument.</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:flex-start;gap:.5rem;background:rgba(245,185,66,.08);'
        'border:1px solid #8a6d2b;border-radius:6px;padding:.5rem .9rem;margin:0 0 1rem;color:#d9c79a;font-size:.82rem">'
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e0b84a" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" style="flex:none;margin-top:1px">'
        '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        '<span><b>Ceci ne constitue pas un conseil en investissement.</b> Figures come from public data and are '
        'shown for education and analysis only — not a recommendation, valuation opinion or solicitation to buy '
        'or sell any instrument.</span></div>', unsafe_allow_html=True)

    _tc1, _tc2 = st.columns([4, 1], vertical_alignment="bottom")
    with _tc1:
        _sym = st.text_input("Ticker symbol", value="AAPL", key="tk_symbol",
                             placeholder="e.g. AAPL, MSFT, MC.PA, BTC-USD").strip().upper()
    with _tc2:
        st.button("Fetch", type="primary", use_container_width=True, key="tk_fetch")

    if _sym:
        with st.spinner(f"Fetching {_sym}…"):
            _info, _err = _cached_ticker_info(_sym)
        if _err:
            st.error(_err)
        elif _info:
            _hdr = _rt.company_header(_sym, _info)
            _meta = " · ".join([x for x in [_hdr["sector"], _hdr["industry"], _hdr["exchange"]] if x])
            _asof_lbl = (f"as of {_hdr['price_asof']}" if _hdr["asof_is_market"]
                         else f"retrieved {_hdr['price_asof']}")
            st.markdown(
                '<div style="background:#1b2330;border:1px solid #30363d;border-radius:10px;'
                'padding:.8rem 1.1rem;margin:.2rem 0 1rem">'
                '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem;align-items:baseline">'
                f'<div><span style="color:#E3C77E;font-weight:700;font-size:1.15rem">{_hdr["name"]}</span>'
                f'<span style="color:#8b97a8;font-weight:600;margin-left:.45rem">{_hdr["ticker"]}</span>'
                f'<div style="color:#9aa7bd;font-size:.82rem;margin-top:.15rem">{_meta}</div>'
                f'<div style="color:#9aa7bd;font-size:.8rem;margin-top:.3rem">52-week range: '
                f'{_hdr["low52"]} – {_hdr["high52"]}</div></div>'
                f'<div style="text-align:right"><div style="color:#fafafa;font-size:1.1rem;font-weight:700">{_hdr["price"]}</div>'
                f'<div style="color:#9aa7bd;font-size:.8rem">Market cap {_hdr["market_cap"]}</div>'
                f'<div style="color:#6b7686;font-size:.74rem;margin-top:.12rem">{_asof_lbl}</div></div>'
                '</div></div>', unsafe_allow_html=True)

            # ── 52-week range bar ──
            _rpos = _hdr.get("range_pos")
            if _rpos is not None:
                _p = max(0.0, min(1.0, _rpos)) * 100
                st.markdown(
                    '<div style="margin:.1rem 0 1.1rem">'
                    f'<div style="position:relative;height:18px"><div style="position:absolute;left:{_p:.1f}%;'
                    'transform:translateX(-50%);font-size:.74rem;color:#E3C77E;font-weight:600;white-space:nowrap">'
                    f'{_hdr["price"]}</div></div>'
                    '<div style="position:relative;height:9px;background:linear-gradient(90deg,#26415f,#3a5a82);'
                    'border-radius:5px">'
                    f'<div style="position:absolute;left:{_p:.1f}%;top:-3.5px;transform:translateX(-50%);width:4px;'
                    'height:16px;background:#E3C77E;border-radius:2px;box-shadow:0 0 0 2px #0d1117"></div></div>'
                    '<div style="display:flex;justify-content:space-between;font-size:.74rem;color:#9aa7bd;margin-top:4px">'
                    f'<span>52-wk low {_hdr["low52"]}</span><span>52-week range</span>'
                    f'<span>52-wk high {_hdr["high52"]}</span></div></div>', unsafe_allow_html=True)

            # ── Price-history chart: period + chart-type selectors ──
            _pmap = {"6M": "6mo", "1Y": "1y", "5Y": "5y", "Max": "max"}
            _selc1, _selc2, _selc3 = st.columns([2, 2, 4])
            with _selc1:
                _per = st.radio("Period", list(_pmap.keys()), index=1, horizontal=True,
                                key="tk_period", label_visibility="collapsed")
            with _selc2:
                _ctype = st.radio("Chart type", ["Line", "Candlestick"], index=0, horizontal=True,
                                  key="tk_chart", label_visibility="collapsed")
            with st.spinner("Loading price history…"):
                _hist, _herr = _cached_ticker_history(_sym, _pmap[_per])
            if _hist is not None and len(_hist):
                _can_candle = all(_c in _hist.columns for _c in ("Open", "High", "Low", "Close"))
                if _ctype == "Candlestick" and _can_candle:
                    _pfig = _tk_candle_fig(_hist, _hdr["name"])
                else:
                    _pfig = _tk_price_fig(_hist, _hdr["name"])
                st.plotly_chart(_pfig, use_container_width=True,
                                config={"responsive": True, "displayModeBar": False}, key="tk_price_chart")
                # ── Performance & risk summary (over the selected period) ──
                try:
                    _close = _hist["Close"].dropna()
                    _pret = float(_close.iloc[-1] / _close.iloc[0] - 1.0) if len(_close) > 1 else float("nan")
                    _dret = _close.pct_change().dropna()
                    _vol = float(_dret.std() * (252 ** 0.5)) if len(_dret) > 2 else float("nan")
                    _mdd = float((_close / _close.cummax() - 1.0).min()) if len(_close) > 1 else float("nan")
                    _beta = _info.get("beta")
                    _beta = float(_beta) if isinstance(_beta, (int, float)) and _beta == _beta else None
                    if _vol == _vol:
                        if _vol < 0.15:   _rb, _rbc = "Low", "#2dd4bf"
                        elif _vol < 0.25: _rb, _rbc = "Moderate", "#E3C77E"
                        elif _vol < 0.40: _rb, _rbc = "High", "#f0a23f"
                        else:             _rb, _rbc = "Very high", "#f85149"
                    else:
                        _rb, _rbc = "—", "#6b7686"

                    def _tk_card(_lab, _val, _col="#fafafa"):
                        return ('<div style="flex:1;min-width:118px;background:#1b2330;border:1px solid #30363d;'
                                'border-radius:9px;padding:.5rem .7rem;text-align:center">'
                                f'<div style="color:#94a3ba;font-size:.7rem;text-transform:uppercase;'
                                f'letter-spacing:.05em">{_lab}</div>'
                                f'<div style="color:{_col};font-size:1.15rem;font-weight:700;'
                                f'margin-top:.15rem">{_val}</div></div>')

                    _pc = "#9be9a8" if (_pret == _pret and _pret >= 0) else "#f85149"
                    _strip = '<div style="display:flex;gap:.55rem;flex-wrap:wrap;margin:.5rem 0 .6rem">'
                    _strip += _tk_card(f"{_per} return", f"{_pret*100:+.1f}%" if _pret == _pret else "—", _pc)
                    _strip += _tk_card("Volatility (ann.)", f"{_vol*100:.1f}%" if _vol == _vol else "—")
                    _strip += _tk_card("Max drawdown",
                                       f"{_mdd*100:.1f}%" if _mdd == _mdd else "—",
                                       "#f85149" if (_mdd == _mdd and _mdd < 0) else "#fafafa")
                    if _beta is not None:
                        _strip += _tk_card("Beta (β)", f"{_beta:.2f}")
                    _strip += ('<div style="flex:1;min-width:118px;background:#1b2330;border:1px solid #30363d;'
                               'border-radius:9px;padding:.5rem .7rem;text-align:center">'
                               '<div style="color:#94a3ba;font-size:.7rem;text-transform:uppercase;'
                               'letter-spacing:.05em">Risk level</div>'
                               f'<div style="margin-top:.25rem"><span style="background:{_rbc};color:#10151f;'
                               f'font-weight:700;font-size:.95rem;border-radius:6px;padding:.12rem .6rem">{_rb}</span>'
                               '</div></div>')
                    _strip += '</div>'
                    st.markdown(_strip, unsafe_allow_html=True)
                    # Volatility-band table — explains the risk level and highlights the current band.
                    _vbands = [("Low", "&lt; 15%", "#2dd4bf"), ("Moderate", "15–25%", "#E3C77E"),
                               ("High", "25–40%", "#f0a23f"), ("Very high", "&ge; 40%", "#f85149")]
                    _vrows = ""
                    for _bn, _brange, _bc in _vbands:
                        _hl = (_bn == _rb)
                        _bg = "background:#16233a;" if _hl else ""
                        _mk = '<span style="color:#E3C77E;font-weight:700">&#9664; this ticker</span>' if _hl else ""
                        _vrows += (f'<tr style="{_bg}">'
                                   f'<td style="padding:2px 12px;border:1px solid #2a3340;color:{_bc};font-weight:600">{_bn}</td>'
                                   f'<td style="padding:2px 12px;border:1px solid #2a3340;color:#c0c8d8">{_brange}</td>'
                                   f'<td style="padding:2px 12px;border:1px solid #2a3340">{_mk}</td></tr>')
                    _voltxt = (f"{_vol*100:.1f}%" if _vol == _vol else "—")
                    st.markdown(
                        '<div style="color:#9aa7bd;font-size:.82rem;margin:.55rem 0 .35rem">'
                        'The <b style="color:#c0c8d8">risk level is volatility-based</b> — '
                        f'{_sym}&rsquo;s annualised volatility of <b style="color:#fff">{_voltxt}</b> places it in the '
                        f'<b style="color:{_rbc}">{_rb}</b> band:</div>'
                        '<table style="border-collapse:collapse;font-size:.8rem;margin-bottom:.5rem">'
                        '<tr style="color:#E3C77E;font-weight:600">'
                        '<td style="padding:2px 12px;border:1px solid #2a3340">Risk level</td>'
                        '<td style="padding:2px 12px;border:1px solid #2a3340">Annualised volatility</td>'
                        '<td style="padding:2px 12px;border:1px solid #2a3340"></td></tr>'
                        + _vrows + '</table>', unsafe_allow_html=True)
                    st.caption("Return and max drawdown are over the selected period; volatility is annualised "
                               "from daily moves. An educational characteristic, not a rating or recommendation.")
                except Exception:
                    pass
            else:
                st.caption("Price history isn't available for this instrument.")

            _data = _rt.build_ratios(_sym, _info)
            _hc = ("background:#141a23;color:#E3C77E;font-weight:700;text-align:center;"
                   "padding:.35rem .5rem;border:1px solid #C9A24B;font-size:.8rem")
            _dl = "background:#0d1117;color:#c9d1d9;padding:.3rem .55rem;border:1px solid #30363d;text-align:left"
            _dg = ("background:#0d1117;color:#9aa7bd;padding:.3rem .55rem;border:1px solid #30363d;"
                   "text-align:left;font-size:.8rem")

            def _vcell(_v):
                _col = "#E3C77E" if _v != "—" else "#6b7686"
                return (f"background:#0d1117;color:{_col};font-weight:600;padding:.3rem .55rem;"
                        f"border:1px solid #30363d;text-align:center")

            # Render two categories per row, each row in its OWN st.columns(2) so the
            # second-row titles align even when the first-row tables differ in height.
            for _ri in range(0, len(_rt.CATEGORIES), 2):
                _rowcols = st.columns(2)
                for _k, _cat in enumerate(_rt.CATEGORIES[_ri:_ri + 2]):
                    with _rowcols[_k]:
                        _rows_html = "".join(
                            f'<tr><td style="{_dl}">{_r["label"]}</td>'
                            f'<td style="{_vcell(_r["value"])}">{_r["value"]}</td>'
                            f'<td style="{_dg}">{_r["gauge"]}</td></tr>' for _r in _data[_cat])
                        st.markdown(f'<div style="color:#E3C77E;font-weight:700;font-size:.95rem;'
                                    f'text-align:center;margin:.3rem 0 .7rem">{_cat}</div>', unsafe_allow_html=True)
                        st.html(f'<table style="width:100%;border-collapse:collapse;font-size:.84rem;margin-bottom:.4rem">'
                                f'<tbody><tr><td style="{_hc}">Ratio</td><td style="{_hc}">Value</td>'
                                f'<td style="{_hc}">What it gauges</td></tr>{_rows_html}</tbody></table>')

            # ── Financials section: trends + balance-sheet snapshot + company profile ──
            _fin, _ = _cached_ticker_financials(_sym)
            _cf, _ = _cached_ticker_cashflow(_sym)
            _bal, _ = _cached_ticker_balance(_sym)
            _has_rev = bool(_fin) and any((_r.get("revenue") is not None or _r.get("net_income") is not None) for _r in _fin)
            _has_margin = bool(_fin) and any((_r.get("revenue") and (_r.get("gross_profit") is not None or _r.get("operating_income") is not None)) for _r in _fin)
            _has_cf = bool(_cf) and any((_r.get("ocf") is not None or _r.get("fcf") is not None) for _r in _cf)
            _has_bal = bool(_bal) and any(_bal.get(_k) is not None for _k in ("total_assets", "total_debt", "cash", "equity"))

            def _ctitle(_t):
                st.markdown(f'<div style="color:#E3C77E;font-weight:700;font-size:.95rem;'
                            f'text-align:center;margin:.3rem 0 .3rem">{_t}</div>', unsafe_allow_html=True)

            _cfg = {"responsive": True, "displayModeBar": False}
            if _has_rev or _has_margin or _has_cf or _has_bal:
                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
            # Row 1 — Revenue & net income | Margins
            if _has_rev or _has_margin:
                _fr1, _fr2 = st.columns(2)
                with _fr1:
                    if _has_rev:
                        _ctitle("Revenue &amp; net income (annual)")
                        st.plotly_chart(_tk_revenue_fig(_fin, _hdr["currency"]), use_container_width=True,
                                        config=_cfg, key="tk_rev_chart")
                with _fr2:
                    if _has_margin:
                        _ctitle("Margins (gross / operating / net)")
                        st.plotly_chart(_tk_margins_fig(_fin), use_container_width=True,
                                        config=_cfg, key="tk_margin_chart")
            # Row 2 — Cash flow | Balance-sheet snapshot
            if _has_cf or _has_bal:
                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)  # space above row 2
                _fr3, _fr4 = st.columns(2)
                with _fr3:
                    if _has_cf:
                        _ctitle("Cash flow (operating &amp; free)")
                        st.plotly_chart(_tk_cashflow_fig(_cf, _hdr["currency"]), use_container_width=True,
                                        config=_cfg, key="tk_cf_chart")
                with _fr4:
                    if _has_bal:
                        _ctitle(f"Balance sheet ({_bal.get('year', 'latest')})")
                        st.plotly_chart(_tk_balance_fig(_bal, _hdr["currency"]), use_container_width=True,
                                        config=_cfg, key="tk_bal_chart")

            # Company profile (full width)
            _summary = (_info.get("longBusinessSummary") or "").strip()
            _facts = []
            if _hdr["sector"]:
                _facts.append(("Sector", _hdr["sector"]))
            if _hdr["industry"]:
                _facts.append(("Industry", _hdr["industry"]))
            if _info.get("fullTimeEmployees"):
                _facts.append(("Employees", f"{int(_info['fullTimeEmployees']):,}"))
            _loc = ", ".join([x for x in [_info.get("city"), _info.get("country")] if x])
            if _loc:
                _facts.append(("Headquarters", _loc))
            if _hdr["exchange"]:
                _facts.append(("Exchange", _hdr["exchange"]))
            _web = _info.get("website")
            if _summary or _facts:
                st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)  # space above title
                _ctitle("Company profile")
                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)  # space below title
                _facts_html = "".join(
                    '<span style="background:#0e1521;border:1px solid #30363d;border-radius:999px;'
                    'padding:.25rem .7rem;font-size:.8rem;color:#c0c8d8;margin:.15rem .3rem .15rem 0;'
                    f'display:inline-block"><span style="color:#9aa7bd">{_k}:</span> <b>{_v}</b></span>'
                    for _k, _v in _facts)
                _parts = ['<div style="background:#1b2330;border:1px solid #30363d;border-radius:10px;'
                          'padding:.9rem 1.1rem">']
                if _facts:
                    _parts.append('<div style="margin-bottom:.55rem">' + _facts_html + '</div>')
                if _summary:
                    _parts.append('<div style="color:#c0c8d8;font-size:.9rem;line-height:1.6">' + _summary + '</div>')
                if _web:
                    _parts.append('<div style="margin-top:.55rem;font-size:.82rem">'
                                  f'<a href="{_web}" target="_blank" style="color:#79b6ff;text-decoration:none">{_web}</a></div>')
                _parts.append('</div>')
                st.markdown("".join(_parts), unsafe_allow_html=True)
                st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)  # space below profile

            st.markdown('<h3 style="color:#E3C77E;margin:1rem 0 .3rem">Explain a ratio or figure</h3>', unsafe_allow_html=True)
            _opts = {}  # label -> (explanation_html, value_for_ai)
            for _cat in _rt.CATEGORIES:
                for _r in _data[_cat]:
                    if _r["available"]:
                        _opts[_r["label"]] = (_r["explain"], _r["value"])
            _terms = getattr(_rt, "CHART_TERMS", {})
            _extra = []
            if _has_rev:
                _extra += ["Revenue", "Net income"]
            if _has_cf:
                _extra += ["Operating cash flow", "Free cash flow"]
            if _has_bal:
                _extra += ["Total assets", "Total debt", "Cash", "Shareholders' equity"]
            for _t in _extra:
                if _t in _terms and _t not in _opts:
                    _opts[_t] = (_terms[_t], "")
            if _opts:
                _elabels = list(_opts.keys())
                _esel = st.selectbox("Pick a ratio or figure", _elabels,
                                     key="tk_explain_sel", label_visibility="collapsed")
                if st.button("Explain", type="primary", icon=":material/smart_toy:", key="tk_explain_btn"):
                    _ehtml, _eval = _opts.get(_esel, ("", ""))
                    with st.spinner(f"Explaining {_esel}…"):
                        _eresp = get_ai_chat_response(
                            f"Explain '{_esel}' for {_sym} in plain language for a non-specialist investor.",
                            portfolio_context=(f"{_sym}: {_esel} = {_eval}" if _eval else f"{_sym}"))
                    _ebody = _eresp or _ehtml or "No explanation is available for this item right now."
                    st.markdown(
                        '<div style="background:#1b2330;border:1px solid #30363d;border-radius:10px;'
                        'padding:.9rem 1.1rem;margin-top:.6rem;color:#c0c8d8;font-size:.9rem;line-height:1.6">'
                        f'<b style="color:#E3C77E">{_esel}</b>'
                        + (f' <span style="color:#9aa7bd">— {_eval}</span>' if _eval else '')
                        + f'<div style="margin-top:.4rem">{_ebody}</div></div>',
                        unsafe_allow_html=True)

            st.markdown(
                '<div style="color:#6b7686;font-size:.74rem;margin-top:1.4rem;line-height:1.5">'
                'Figures come from public data and are shown for education and analysis only — not a '
                'recommendation, valuation opinion or solicitation to buy or sell any instrument.</div>',
                unsafe_allow_html=True)
