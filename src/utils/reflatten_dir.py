import pathlib
import shutil

# provide your config values
INPUT_DIR = pathlib.Path('')
OUTPUT_DIR = pathlib.Path('')
SEPARATOR = '&'
IMAGE_SUFFIX = '.jpg'

if __name__ == "__main__":
    for file in INPUT_DIR.rglob(f'*{IMAGE_SUFFIX}'):
        group_name = file.name.split(SEPARATOR)[0]
        group_dir = OUTPUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        filename = file.name.split(SEPARATOR)[-1]
        dest = pathlib.Path(OUTPUT_DIR / group_name / filename)
        shutil.copy(file, dest)
