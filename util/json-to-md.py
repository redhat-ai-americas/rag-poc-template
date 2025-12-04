import json
import os

# --- CONFIG ---
input_folder = "./assets/raw-json"  # folder containing your Docling JSON files
output_folder = "./assets/markdown-out"  # folder to save generated Markdown files
heading_keywords = [
    "Business Objective",
    "Mid-Level Solution Requirement",
    "Detailed Solution Requirement",
]

# make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# --- HELPER FUNCTION TO DETECT HEADINGS ---
def format_heading(text):
    if text.strip() in heading_keywords:
        return f"## {text.strip()}"
    if text.strip().endswith(":"):
        return f"## {text.strip()}"
    return text.strip()


# --- PROCESS EACH JSON FILE ---
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)
    output_md_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.md")

    # load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # build lookup maps
    texts_map = {t["self_ref"]: t for t in data.get("texts", [])}
    pictures_map = {p["self_ref"]: p for p in data.get("pictures", [])}

    # extract Markdown lines
    md_lines = []
    body_children = data.get("body", {}).get("children", [])

    for child in body_children:
        ref = child.get("$ref")
        if ref in pictures_map:
            picture = pictures_map[ref]
            for t_ref in picture.get("children", []):
                t = texts_map.get(t_ref["$ref"])
                if t and t.get("label") != "page_footer":
                    line = format_heading(t["text"])
                    md_lines.append(line)

    # write Markdown
    markdown_content = "\n\n".join(md_lines)
    with open(output_md_path, "w") as f:
        f.write(markdown_content)

    print(f"Converted {filename} -> {output_md_path}")
