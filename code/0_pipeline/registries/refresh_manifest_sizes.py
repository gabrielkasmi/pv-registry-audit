"""Refresh MANIFEST size_bytes from what is actually on disk."""
import pandas as pd, os
from pathlib import Path
m = pd.read_csv('data/MANIFEST.csv')
for i, r in m.iterrows():
    p = Path('data') / r['path']
    if not p.exists(): continue
    m.at[i,'size_bytes'] = (p.stat().st_size if p.is_file()
        else sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))
m.to_csv('data/MANIFEST.csv', index=False)
print('MANIFEST refreshed')
