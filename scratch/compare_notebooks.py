import json

def get_code_comments(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    comments = []
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            lines = cell.get('source', [])
            comments_in_cell = [l.strip() for l in lines if l.strip().startswith('#')]
            comments.append(comments_in_cell[0] if comments_in_cell else "NO_COMMENT")
    return comments

c_upgraded = get_code_comments("Highlight_generation_FINAL_Upgraded.ipynb")
c_colab = get_code_comments("Highlight_generation_FINAL_colab.ipynb")

for idx, (c1, c2) in enumerate(zip(c_upgraded, c_colab)):
    if c1 != c2:
        print(f"Cell {idx} difference:")
        print(f"  Upgraded: {c1}")
        print(f"  Colab:    {c2}")
