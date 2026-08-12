"""Check that the LaTeX report is self-contained and references real assets."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


ROOT=Path(__file__).resolve().parent
REPORT=ROOT/"project_final_report"/"report.tex"
STYLE=ROOT/"project_final_report"/"neurips.sty"


def main():
    text=REPORT.read_text(encoding="utf-8"); errors=[]
    if not STYLE.exists() or STYLE.stat().st_size==0:
        errors.append(f"missing original-template style package: {STYLE}")
    for placeholder in ("TODO","TBD","TO_BE_GENERATED","INSERT RESULT"):
        if placeholder.lower() in text.lower(): errors.append(f"placeholder remains: {placeholder}")
    for match in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",text):
        path=ROOT/"project_final_report"/match.group(1)
        if not path.exists() or path.stat().st_size==0: errors.append(f"missing figure: {path}")
    for match in re.finditer(r"\\input\{([^}]+)\}",text):
        path=ROOT/"project_final_report"/(match.group(1)+("" if Path(match.group(1)).suffix else ".tex"))
        if not path.exists() or path.stat().st_size==0: errors.append(f"missing input: {path}")
    bibitem_list=re.findall(r"\\bibitem\{([^}]+)\}",text)
    bibitems=set(bibitem_list)
    if len(bibitems)!=len(bibitem_list): errors.append("duplicate bibliography key")
    bibliography=text[text.find(r"\begin{thebibliography}"):text.find(r"\end{thebibliography}")]
    entries=re.split(r"\\bibitem\{[^}]+\}",bibliography)[1:]
    if any(r"[Online]. Available: \url{" not in entry for entry in entries):
        errors.append("every IEEE bibliography entry must include an Online Available URL")
    citations=set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}",text): citations.update(part.strip() for part in group.split(","))
    missing_citations=citations-bibitems
    if missing_citations: errors.append(f"citations without bibitems: {sorted(missing_citations)}")
    label_list=re.findall(r"\\label\{([^}]+)\}",text); labels=set(label_list); references=set(re.findall(r"\\ref\{([^}]+)\}",text))
    if len(labels)!=len(label_list): errors.append("duplicate LaTeX label")
    if references-labels: errors.append(f"references without labels: {sorted(references-labels)}")
    begins=Counter(re.findall(r"\\begin\{([^}]+)\}",text)); ends=Counter(re.findall(r"\\end\{([^}]+)\}",text))
    if begins!=ends:
        errors.append(f"unbalanced LaTeX environments: begin={dict(begins-ends)}, end={dict(ends-begins)}")
    if errors:
        print("REPORT VALIDATION FAILED")
        for error in errors: print(f"- {error}")
        raise SystemExit(1)
    print(f"REPORT VALIDATION PASSED: {len(bibitems)} references, {len(labels)} labels, all assets present")


if __name__=="__main__": main()
