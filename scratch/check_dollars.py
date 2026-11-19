with open("docs/research_paper_ieee_updated.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()
for num in [189, 260, 342, 345, 427]:
    l = lines[num-1].strip()
    pos = [i for i, c in enumerate(l) if c == '$']
    print(f"Line {num} ({len(pos)}): {l}")
    for p in pos:
        start = max(0, p - 10)
        end = min(len(l), p + 15)
        print(f"  at {p}: ...{l[start:end]}...")
