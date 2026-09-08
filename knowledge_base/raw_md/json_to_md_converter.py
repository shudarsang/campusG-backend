#!/usr/bin/env python3
"""Convert the college RAG knowledge-base JSON files to Markdown.

Rules:
  - Every key and every value in the source JSON appears in the output.
  - Nothing is summarised, merged, reworded or dropped.
  - Only structure/presentation changes (keys become headings/bold labels).
"""
import json
import os
import re

SRC = "/mnt/user-data/uploads"
DST = "/mnt/user-data/outputs/markdown"

ACRONYMS = {
    "ug": "UG", "pg": "PG", "phd": "PhD", "mba": "MBA", "mca": "MCA",
    "bca": "BCA", "bba": "BBA", "mphil": "M.Phil.", "naac": "NAAC",
    "nirf": "NIRF", "nss": "NSS", "ncc": "NCC", "yrc": "YRC",
    "iqac": "IQAC", "ecric": "ECRIC", "qs": "QS", "it": "IT",
    "url": "URL", "urls": "URLs", "faq": "FAQ", "faqs": "FAQs",
    "atm": "ATM", "hod": "HOD", "wifi": "WiFi", "ss": "SS", "id": "ID",
}


def human(key):
    parts = re.split(r"[_\s]+", str(key))
    out = []
    for p in parts:
        low = p.lower()
        out.append(ACRONYMS.get(low, p.capitalize() if p.islower() else p))
    return " ".join(out)


def scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def render(value, lines, indent=0):
    """Render a value as bullet lines at the given indent level."""
    pad = "  " * indent
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{human(k)}:**")
                render(v, lines, indent + 1)
            else:
                lines.append(f"{pad}- **{human(k)}:** {scalar(v)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                render(item, lines, indent)
                lines.append("")
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                render(item, lines, indent + 1)
            else:
                lines.append(f"{pad}- {scalar(item)}")
    else:
        lines.append(f"{pad}{scalar(value)}")


NAME_KEYS = ("name", "department_name", "course_name", "title")


def item_heading(item, idx):
    for k in NAME_KEYS:
        if k in item and isinstance(item[k], str):
            return item[k]
    return f"Item {idx}"


def convert(data, title):
    lines = [f"# {title}", ""]
    if not isinstance(data, dict):
        render(data, lines)
        return "\n".join(lines).rstrip() + "\n"

    for key, value in data.items():
        lines.append(f"## {human(key)}")
        lines.append("")
        if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
            # List of records -> one sub-heading per record, keeps chunks clean
            for idx, item in enumerate(value, 1):
                lines.append(f"### {item_heading(item, idx)}")
                lines.append("")
                render(item, lines)
                lines.append("")
        elif isinstance(value, (dict, list)):
            render(value, lines)
            lines.append("")
        else:
            lines.append(scalar(value))
            lines.append("")
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip() + "\n"


def convert_faq(data, title):
    lines = [f"# {title}", ""]
    lines.append(f"**FAQ Count:** {data['faq_count']}")
    lines.append("")
    current = None
    for i, f in enumerate(data["faqs"], 1):
        cat = f.get("category")
        if cat != current:
            lines.append(f"## {cat}")
            lines.append("")
            current = cat
        lines.append(f"### Q{i}. {f['question']}")
        lines.append("")
        lines.append(f["answer"])
        lines.append("")
        lines.append(f"**Category:** {cat}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


os.makedirs(DST, exist_ok=True)
report = []
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(SRC, fn), encoding="utf-8") as fh:
        data = json.load(fh)
    title = human(os.path.splitext(fn)[0])
    if fn == "faq.json":
        md = convert_faq(data, "FAQ")
    else:
        md = convert(data, title)
    out = os.path.join(DST, os.path.splitext(fn)[0] + ".md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    report.append((fn, len(md)))

for fn, n in report:
    print(f"{fn:28s} -> {os.path.splitext(fn)[0]+'.md':28s} {n:>7d} chars")
