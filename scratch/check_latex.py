with open("docs/research_paper_ieee_updated.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    code = line.split("%")[0]
    c = code.count("$")
    if c % 2 != 0:
        print(f"Line {idx+1}: {c} dollars -> {line.strip()}")
