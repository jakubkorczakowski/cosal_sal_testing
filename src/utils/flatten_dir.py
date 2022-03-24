import pathlib
import shutil

# provide your config values
INPUT_DIR = pathlib.Path('')
OUTPUT_DIR = pathlib.Path('')
SEPARATOR = '&'
IMAGE_SUFFIX = '.jpg'

if __name__ == "__main__":
    for file in INPUT_DIR.rglob(f'*{IMAGE_SUFFIX}'):
        flatten_filename = f"{file.parent.stem}{SEPARATOR}{file.name}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = pathlib.Path(OUTPUT_DIR / flatten_filename)
        shutil.copy(file, dest)
