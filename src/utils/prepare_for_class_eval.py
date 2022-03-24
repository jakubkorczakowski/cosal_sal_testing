import pathlib
import shutil


INPUT_DIR = pathlib.Path('')
OUTPUT_DIR = pathlib.Path('')
SEPARATOR = '_'
IMAGE_SUFFIX = '.jpg'

if __name__ == "__main__":
    for current_dir in INPUT_DIR.iterdir():
        filename = current_dir.stem
        group_dir = OUTPUT_DIR / f'{INPUT_DIR.stem}{SEPARATOR}{filename}' / filename
        group_dir.mkdir(parents=True, exist_ok=True)
        for file in current_dir.rglob(f'*{IMAGE_SUFFIX}'):
            shutil.copy(file, group_dir / file.name)

