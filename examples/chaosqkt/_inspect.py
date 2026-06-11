import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = r'C:\Users\z5459034\OneDrive - UNSW\Documents\Gits\Psyduck\examples\chaosqkt\ClassicalSimulations.ipynb'
nb = json.load(open(path, encoding='utf-8'))

# Parse cell 0 as JSON (it contains an embedded notebook)
raw_src = ''.join(nb['cells'][0]['source'])
inner = json.loads(raw_src)

# Compare cells after cell 0 of current notebook with cells of inner notebook
# by source content.
current_cells = nb['cells'][1:]
inner_cells = inner['cells']

print(f'current code/markdown cells after cell 0: {len(current_cells)}')
print(f'inner cells: {len(inner_cells)}')

# Map inner cells by source -> index for cross-check
inner_sources = [(c['cell_type'], ''.join(c.get('source', []))) for c in inner_cells]
current_sources = [(c['cell_type'], ''.join(c.get('source', []))) for c in current_cells]

# Check which current cells are present in inner
for i, (t, s) in enumerate(current_sources):
    found_at = None
    for j, (it, isrc) in enumerate(inner_sources):
        if t == it and s == isrc:
            found_at = j
            break
    flag = 'MATCH' if found_at is not None else 'DIFFERENT'
    print(f'current[{i+1}] ({t}, {len(s)}ch) -> inner[{found_at}] : {flag}')
    if flag == 'DIFFERENT':
        print('  PREVIEW:', s[:200].replace('\n',' / '))
