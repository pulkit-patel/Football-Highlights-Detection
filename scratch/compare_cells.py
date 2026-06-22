import json

def dump_cell_info(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"=== {path} ===")
    for idx, cell in enumerate(data.get('cells', [])):
        cell_type = cell.get('cell_type')
        lines = cell.get('source', [])
        first_line = lines[0].strip() if lines else "EMPTY"
        # Convert to ascii safely
        first_line_ascii = first_line.encode('ascii', errors='replace').decode('ascii')
        print(f"{idx:02d} [{cell_type}]: {first_line_ascii[:60]}")

dump_cell_info("Highlight_generation_FINAL_Upgraded.ipynb")
dump_cell_info("Highlight_generation_FINAL_colab.ipynb")
dump_cell_info("Highlight_generation_FINAL_Upgraded_colab.ipynb")
