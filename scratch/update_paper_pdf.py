from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


SOURCE = Path(r"C:\Users\BAPS\Downloads\Research_paper_2.pdf")
OUTPUT = Path(r"C:\Red_queen\output\pdf\Research_paper_2_IEEE_Updated.pdf")
TEMP = Path(r"C:\Red_queen\scratch\paper_overlay.pdf")

ABSTRACT = (
    "<i><b>Abstract</b></i> - In competitive multi-agent reinforcement learning (MARL), "
    "unanchored self-play often develops cycles as each agent adapts to its opponent, a "
    "pattern known as the Red Queen effect. This shifting competition can also cause "
    "catastrophic forgetting, where a policy learns to counter recent behavior but loses "
    "previously learned routes. We investigate these problems in the Three Bridges Arena, "
    "an asymmetric zero-sum gridworld with three bridge crossings and one Defender, which "
    "cannot guard every crossing at once. We propose a symmetric Dual-Archive Co-Evolution "
    "approach that trains current policies against one another and against archived earlier "
    "policies. It also uses discounted potential-based reward shaping with a discount factor "
    "of 0.99, designed to preserve optimal policies under the policy-invariance result of "
    "Ng et al. (1999). Across 150 paired evaluation seeds, naive self-play showed route "
    "collapse: 46.0 percent North bridge traversal compared with 16.1 percent South, 17.3 "
    "percent unforced navigation timeouts, and an exploitability score of 0.160. "
    "Dual-Archive co-evolution produced more balanced use of the three corridors, with "
    "Shannon route entropy of 1.567 bits, 98.9 percent of the theoretical maximum of 1.585 "
    "bits, and no timeouts. It also reduced minimax exploitability to 0.140 and improved the "
    "worst-case security level from -0.207 to -0.187. Pooled win rates did not differ "
    "significantly between the methods: the difference was -0.6 percent, with a p-value of "
    "0.771. These results suggest that pooled win rates alone can hide strategic robustness "
    "differences, while route entropy and empirical minimax exploitability provide additional "
    "measures."
)

# Placeholder citations are assigned in their appearance order while preserving the
# three references already numbered in the submitted paper.
CITATIONS = {
    0: ["[4]", "[5]", "[6]", "[7]", "[8]"],
    1: ["[9]", "[9]", "[10]", "[5]", "[6]", "[7]", "[8]", "[11]", "[12]", "[13]", "[14]", "[6]"],
    2: ["[15]", "[9]"],
    4: ["[13]"],
    # These are suggested future-work approaches, so the unresolved template markers
    # are removed without changing the proposal itself.
    7: [",", ""],
}

REFERENCES = [
    ("[4]", "M. McCloskey and N. J. Cohen, \"Catastrophic interference in connectionist networks: The sequential learning problem,\" in <i>Psychology of Learning and Motivation</i>, vol. 24, pp. 109-165, 1989."),
    ("[5]", "D. Silver <i>et al.</i>, \"Mastering the game of Go with deep neural networks and tree search,\" <i>Nature</i>, vol. 529, pp. 484-489, 2016."),
    ("[6]", "O. Vinyals <i>et al.</i>, \"Grandmaster level in StarCraft II using multi-agent reinforcement learning,\" <i>Nature</i>, vol. 575, pp. 350-354, 2019."),
    ("[7]", "OpenAI <i>et al.</i>, \"Dota 2 with large scale deep reinforcement learning,\" Dec. 2019."),
    ("[8]", "D. Balduzzi <i>et al.</i>, \"Open-ended learning in symmetric zero-sum games,\" in <i>Proc. 36th Int. Conf. Machine Learning</i>, PMLR, vol. 97, pp. 434-443, 2019."),
    ("[9]", "S. Devlin and D. Kudenko, \"Theoretical considerations of potential-based reward shaping for multi-agent systems,\" in <i>Proc. 10th Int. Conf. Autonomous Agents and Multiagent Systems</i>, 2011, pp. 225-232."),
    ("[10]", "G. Tesauro, \"Temporal difference learning and TD-Gammon,\" <i>Communications of the ACM</i>, vol. 38, no. 3, pp. 58-68, 1995."),
    ("[11]", "G. W. Brown, \"Iterative solution of games by fictitious play,\" in <i>Activity Analysis of Production and Allocation</i>, T. C. Koopmans, Ed. New York, NY, USA: Wiley, 1951."),
    ("[12]", "H. B. McMahan, G. J. Gordon, and A. Blum, \"Planning in the presence of cost functions controlled by an adversary,\" in <i>Proc. 20th Int. Conf. Machine Learning</i>, 2003, pp. 536-543."),
    ("[13]", "M. Lanctot <i>et al.</i>, \"A unified game-theoretic approach to multiagent reinforcement learning,\" in <i>Advances in Neural Information Processing Systems</i>, vol. 30, 2017."),
    ("[14]", "J. Heinrich and D. Silver, \"Deep reinforcement learning from self-play in imperfect-information games,\" arXiv:1603.01121, 2016."),
    ("[15]", "J. von Neumann and O. Morgenstern, <i>Theory of Games and Economic Behavior</i>. Princeton, NJ, USA: Princeton University Press, 1944."),
]


def make_overlay(page_width: float, page_height: float) -> None:
    c = canvas.Canvas(str(TEMP), pagesize=(page_width, page_height))
    with pdfplumber.open(SOURCE) as pdf:
        for page_index in range(len(pdf.pages)):
            page = pdf.pages[page_index]
            if page_index == 0:
                # Reflow the abstract in its original first-column area. The IEEE
                # template forbids mathematical notation and special symbols here.
                c.setFillColorRGB(1, 1, 1)
                c.rect(47, page_height - 556.0, 258, 294, fill=1, stroke=0)
                style = ParagraphStyle(
                    "abstract",
                    fontName="Times-Bold",
                    fontSize=8.4,
                    leading=9.3,
                    alignment=TA_JUSTIFY,
                    allowWidows=0,
                    allowOrphans=0,
                    spaceAfter=0,
                    splitLongWords=1,
                )
                para = Paragraph(ABSTRACT, style)
                width = 232.5
                _, abstract_height = para.wrap(width, 286)
                if abstract_height > 286:
                    raise ValueError(f"Abstract exceeds original layout area: {abstract_height:.1f} pt")
                para.drawOn(c, 58.93, page_height - 267.01 - abstract_height)

            # Replace this page's unresolved citation markers in place. Each
            # replacement retains the punctuation following its placeholder.
            replacements = CITATIONS.get(page_index, [])
            hits = sorted(
                [w for w in page.extract_words(x_tolerance=1, y_tolerance=2) if w["text"].startswith("[?]")],
                key=lambda w: (round(w["top"], 1), w["x0"]),
            )
            if len(hits) != len(replacements):
                raise ValueError(
                    f"Page {page_index + 1}: expected {len(replacements)} citation markers, found {len(hits)}"
                )
            for word, citation in zip(hits, replacements):
                tail = word["text"][3:]
                c.setFillColorRGB(1, 1, 1)
                c.rect(
                    word["x0"] - 0.4,
                    page_height - word["bottom"] - 0.4,
                    word["x1"] - word["x0"] + 0.8,
                    word["bottom"] - word["top"] + 0.8,
                    fill=1,
                    stroke=0,
                )
                if citation or tail:
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Times-Roman", 7.2)
                    x_adjust = -5.0 if citation == "," or (not citation and tail == ".") else 0.0
                    c.drawString(word["x0"] + x_adjust, page_height - word["bottom"] + 1.7, citation + tail)
            c.showPage()

    # Add a continuation page for the complete IEEE-style reference entries.
    c.setPageSize((page_width, page_height))
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Times-Roman", 10)
    c.drawCentredString(page_width / 2, page_height - 43, "REFERENCES (CONTINUED)")
    ref_style = ParagraphStyle(
        "reference",
        fontName="Times-Roman",
        fontSize=8.0,
        leading=9.2,
        alignment=TA_LEFT,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=5,
        allowWidows=0,
        allowOrphans=0,
    )
    margin = 48
    gap = 22
    col_width = (page_width - 2 * margin - gap) / 2
    columns = [REFERENCES[:6], REFERENCES[6:]]
    for col, entries in enumerate(columns):
        x = margin + col * (col_width + gap)
        y_top = page_height - 62
        for label, text in entries:
            para = Paragraph(f"{label} {text}", ref_style)
            _, ph = para.wrap(col_width, y_top - 48)
            if ph > y_top - 48:
                raise ValueError(f"Reference {label} does not fit the continuation page")
            para.drawOn(c, x, y_top - ph)
            y_top -= ph + ref_style.spaceAfter
    c.showPage()
    c.save()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(SOURCE))
    if len(reader.pages) != 8:
        raise ValueError(f"Expected 8 source pages, found {len(reader.pages)}")
    width = float(reader.pages[0].mediabox.width)
    height = float(reader.pages[0].mediabox.height)
    make_overlay(width, height)
    overlay = PdfReader(str(TEMP))
    writer = PdfWriter()
    for index, page in enumerate(reader.pages):
        page.merge_page(overlay.pages[index])
        writer.add_page(page)
    writer.add_page(overlay.pages[8])
    writer.add_metadata({
        "/Title": "Mitigating Catastrophic Forgetting and Exploitability in Competitive Multi-Agent Reinforcement Learning via Dual-Archive Co-Evolution",
        "/Subject": "IEEE-style paper with completed references and abstract formatting",
    })
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    print(OUTPUT)


if __name__ == "__main__":
    main()
