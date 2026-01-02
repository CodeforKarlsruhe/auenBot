import re
import json
from pathlib import Path
import docx

# -----------------------------
# Page splitting (DOCX)
# -----------------------------
def iter_docx_pages(docx_path: str) -> list[str]:
    """
    Splits a DOCX into pages using explicit Word page breaks (<w:br w:type="page"/>).
    If your file uses only visual pagination (no page breaks), you must add page breaks
    in Word OR choose a different splitter heuristic.
    """
    doc = docx.Document(docx_path)

    pages: list[list[str]] = [[]]

    for p in doc.paragraphs:
        # Collect paragraph text
        if p.text.strip():
            pages[-1].append(p.text)

        # Detect explicit page breaks inside runs
        for run in p.runs:
            br_elems = run._element.findall(".//w:br", run._element.nsmap)
            for br in br_elems:
                if br.get(f"{{{run._element.nsmap['w']}}}type") == "page":
                    # start new page
                    pages.append([])

    # join and trim
    out = ["\n".join(x).strip() for x in pages if "\n".join(x).strip()]
    return out

# -----------------------------
# Tag extraction
# -----------------------------
def extract_tags(text: str, inline_tags: set[str] | None = None) -> dict[str, str]:
    """
    Extract tags and their content from text.

    - Opening tags: [Tag] or **[Tag]**
    - Closing tags: [/Tag] (NOT used as a "starter")
    - If closing tag exists, content is up to it.
      If not, content is up to the next opening tag.
    - inline_tags are extracted first and removed from markup so they don't
      truncate surrounding blocks (e.g. Größe inside Fortpflanzung).
    - Duplicates: combined with blank line separation.
    """
    inline_tags = set(inline_tags or set())

    # Normalize markdown bold wrappers around tags
    t = text.replace("**[", "[").replace("]**", "]")

    # 1) extract inline tags with their own closing tags and strip markup
    inline: dict[str, str] = {}
    for tag in inline_tags:
        pat = re.compile(
            rf"\[(?!/)\s*{re.escape(tag)}\s*\](.*?)\[/\s*{re.escape(tag)}\s*\]",
            re.DOTALL,
        )
        vals = [m.group(1).strip() for m in pat.finditer(t) if m.group(1).strip()]
        if vals:
            inline[tag] = "\n\n".join(vals)
        # replace markup with inner content so parent blocks keep the sentence intact
        t = pat.sub(lambda m: m.group(1), t)

    # 2) extract block tags
    open_pat = re.compile(r"\[(?!/)\s*([^\]\n]+?)\s*\]")
    matches = list(open_pat.finditer(t))

    block: dict[str, list[str]] = {}

    for i, m in enumerate(matches):
        tag = m.group(1).strip()
        if tag in inline_tags:
            continue

        start = m.end()

        # prefer matching closing tag if present
        close_pat = re.compile(rf"\[/\s*{re.escape(tag)}\s*\]")
        close_m = close_pat.search(t, start)

        if close_m:
            end = close_m.start()
        else:
            # fallback: next opening tag
            end = matches[i + 1].start() if i + 1 < len(matches) else len(t)

        content = t[start:end].strip()

        # remove any stray closing tags inside content (since closings can be messy)
        content = re.sub(r"\[/\s*[^]]+\s*\]", "", content).strip()

        if content:
            block.setdefault(tag, []).append(content)

    combined = {k: "\n\n".join(v).strip() for k, v in block.items()}
    combined.update(inline)
    return combined

# -----------------------------
# End-to-end: DOCX -> MD + JSON
# -----------------------------
def process_docx(docx_path: str, out_dir: str, inline_tags: set[str] | None = None):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    pages = iter_docx_pages(docx_path)

    for idx, page_text in enumerate(pages, start=1):
        md_path = outp / f"page_{idx}.md"
        json_path = outp / f"page_{idx}_tags.json"

        # "Markdown" here is just plain text; tags are preserved.
        md_path.write_text(page_text + "\n", encoding="utf-8")

        tags = extract_tags(page_text, inline_tags=inline_tags or set())

        json_path.write_text(json.dumps(tags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Example:
# process_docx("/mnt/data/Texte Krebstiere redigiert - Kopie.docx",
#              "/mnt/data/output",
#              inline_tags={"Größe"})

# Path to the uploaded docx file
import sys
if len(sys.argv) < 3:
    print("Usage: python textExtract.py <path_to_docx> <output_directory>")
    sys.exit(1)
docx_path = sys.argv[1] if len(sys.argv) > 1 else None
output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    
# Process the document
if len(sys.argv) == 4:
    inline_tags = set(sys.argv[3].split(","))
    print(f"Using inline tags: {inline_tags}")
else:
    inline_tags = None

tags = process_docx(docx_path, output_dir, inline_tags=inline_tags)

print(f"Processed {docx_path} into {output_dir}")
print("Done.")
