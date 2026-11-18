"""Create one sample file per supported format (used by tests and for a quick demo)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

DRIP = ("Micro Irrigation Scheme: 100% subsidy is provided for small and marginal farmers up to 2 hectares "
        "and 75% subsidy for other farmers up to 5 hectares. The 12% GST is absorbed by the Government of Tamil Nadu.")
KUSUM = "PM-KUSUM Component B: standalone solar pumps get a total subsidy of 70% (State 40% + Centre 30%)."
SANDHAI = "The first Uzhavar Sandhai (farmers' market) was inaugurated at Madurai on 14 November 1999."
MECH = "Agricultural mechanisation: 50% subsidy on machinery for SC/ST, small, marginal and women farmers."
TA = ("நுண்ணீர் பாசனத் திட்டத்தில் சிறு மற்றும் குறு விவசாயிகளுக்கு 100% மானியம் வழங்கப்படுகிறது. "
      "இதர விவசாயிகளுக்கு 75% மானியம்.")


def make_all(out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    import pymupdf
    doc = pymupdf.open()
    for i, txt in enumerate([DRIP, KUSUM + "\n\n" + SANDHAI], 1):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), f"Page {i}\n\n{txt}\n\n" + ("Details. " * 40), fontsize=11)
    doc.set_metadata({"title": "Sample scheme booklet"})
    files["pdf"] = out / "scheme_booklet.pdf"
    doc.save(str(files["pdf"]))

    import docx
    d = docx.Document()
    d.add_heading("Agricultural Engineering Schemes", 0)
    d.add_heading("Mechanisation", 1)
    d.add_paragraph(MECH)
    t = d.add_table(rows=3, cols=3)
    for r, row in enumerate([["Machine", "Subsidy", "Eligible"], ["Power tiller", "50%", "Small farmers"],
                             ["Rotavator", "40%", "Others"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    files["docx"] = out / "aed_schemes.docx"
    d.save(str(files["docx"]))

    from pptx import Presentation
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "PM-KUSUM in Tamil Nadu"
    s.placeholders[1].text = KUSUM
    s.notes_slide.notes_text_frame.text = "Apply through the Agricultural Engineering Department."
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Uzhavar Sandhai"
    s2.placeholders[1].text = SANDHAI
    files["pptx"] = out / "kusum_deck.pptx"
    prs.save(str(files["pptx"]))

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Subsidy"
    ws.append(["Scheme", "Category", "Subsidy %", "Limit"])
    ws.append(["Micro irrigation", "Small/Marginal", "100%", "2 ha"])
    ws.append(["Micro irrigation", "Other farmers", "75%", "5 ha"])
    ws.append(["Farm pond", "All", "100%", "-"])
    files["xlsx"] = out / "subsidy_table.xlsx"
    wb.save(str(files["xlsx"]))

    files["csv"] = out / "district_targets.csv"
    files["csv"].write_text("district,scheme,target_ha\nCoimbatore,Micro irrigation,4200\nErode,Micro irrigation,3900\n"
                            "Madurai,Farm pond,120\n", encoding="utf-8")

    files["html"] = out / "kaviadp.html"
    files["html"].write_text(
        "<html><head><title>KAVIADP | AED</title><script>var x=1;</script></head><body>"
        "<nav>Home | About | Contact</nav><main><h1>Kalaignarin All Village Integrated Agriculture Development "
        "Programme</h1><p>The final phase in 2025-26 covers 2,338 gram panchayats with an outlay of Rs 269.50 crore."
        "</p><table><tr><th>Component</th><th>Subsidy</th></tr><tr><td>Farm machinery</td><td>100%</td></tr></table>"
        "</main><footer>© Govt</footer></body></html>", encoding="utf-8")

    files["txt"] = out / "sandhai.txt"
    files["txt"].write_text(SANDHAI + "\n\nFarmers sell vegetables directly to consumers.", encoding="utf-8")

    files["md"] = out / "tamil_guide.md"
    files["md"].write_text(f"# நுண்ணீர் பாசனம்\n\n{TA}\n\n## உழவர் சந்தை\n\nமுதல் உழவர் சந்தை 1999-ல் மதுரையில் தொடங்கப்பட்டது.\n",
                           encoding="utf-8")

    files["json"] = out / "scheme.json"
    files["json"].write_text(json.dumps({"scheme": "Crop loan waiver 2026", "limit": "Rs 75,000",
                                         "eligibility": {"lender": "cooperative societies",
                                                         "period": "1 May 2025 - 31 March 2026"}}), encoding="utf-8")

    files["xml"] = out / "notice.xml"
    files["xml"].write_text("<notice><go>G.O. Ms. No. 65</go><subject>Price Support Scheme green gram via NAFED"
                            "</subject></notice>", encoding="utf-8")

    files["doc"] = out / "legacy_note.doc"
    files["doc"].write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64 +
                             "Legacy Word note: farm pond scheme gives 100% subsidy to individual farmers."
                             .encode("utf-16le") + b"\x00" * 32)

    files["zip"] = out / "bundle.zip"
    with zipfile.ZipFile(files["zip"], "w") as z:
        z.writestr("inner/millet.txt", "Tamil Nadu Millet Mission 2023-24 to 2027-28 promotes minor millets.")
        z.writestr("inner/../../evil.txt", "zip slip attempt")  # must be ignored

    files["noext"] = out / "download_123"
    files["noext"].write_bytes(files["pdf"].read_bytes().replace(b"Sample scheme booklet", b"Sample scheme bookle2"))
    return files


if __name__ == "__main__":
    import sys
    for k, v in make_all(Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")).items():
        print(k, v)
