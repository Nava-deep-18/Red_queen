with open("docs/research_paper_ieee_updated.tex", "r", encoding="utf-8") as f:
    text = f.read()

# remove escaped percent \%
text_no_esc = text.replace(r"\%", "")
# now split by lines and remove comments
for idx, line in enumerate(text_no_esc.split("\n")):
    code = line.split("%")[0]
    c = code.count("$")
    if c % 2 != 0:
        print(f"Line {idx+1}: {c} dollars -> {line[:60]}")
print("True math delimiter check complete!")
