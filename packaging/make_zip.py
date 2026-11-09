"""Packs the program, its guide and its full source into the zip people download.

Usage: make_zip.py <project folder> <folder with gui.exe and cli.exe> <README.txt> <out.zip>
"""
import os
import sys
import zipfile

project = sys.argv[1]
built = sys.argv[2]
readme = sys.argv[3]
out = sys.argv[4]
top = 'Demo Analysis'

# Source files: everything needed to build, nothing the tool wrote while running.
source_items = ['src', 'vendor', 'assets', 'packaging', '.github', 'Cargo.toml', 'Cargo.lock', 'build.rs', 'algo_costs.json', 'cosmetics.json', 'LICENSE.md', 'README.md']

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    archive.write(os.path.join(built, 'gui.exe'), f'{top}/Demo Analysis.exe')
    archive.write(os.path.join(built, 'cli.exe'), f'{top}/demo-analysis-cli.exe')
    archive.write(readme, f'{top}/README.txt')
    archive.write(os.path.join(project, 'LICENSE.md'), f'{top}/LICENSE.md')
    archive.write(os.path.join(project, 'CHANGES.txt'), f'{top}/CHANGES.txt')
    archive.write(os.path.join(project, 'GUIDE.md'), f'{top}/GUIDE.md')
    images = os.path.join(project, 'docs', 'images')
    for name in sorted(os.listdir(images)):
        archive.write(os.path.join(images, name), f'{top}/docs/images/{name}')
    count = 0
    for item in source_items:
        path = os.path.join(project, item)
        if os.path.isfile(path):
            archive.write(path, f'{top}/source/{item}')
            count += 1
            continue
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ('target', '.git')]
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, project).replace(os.sep, '/')
                archive.write(full, f'{top}/source/{rel}')
                count += 1
print('source files:', count)
print('zip bytes:', os.path.getsize(out))
