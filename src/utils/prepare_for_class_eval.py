import pathlib
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--input_dir', type=pathlib.Path,
        help='Path to directory containing input images.'
    )
    parser.add_argument(
        '--output_dir', type=pathlib.Path,
        help='Path to directory where you want to save result images.'
    )   
    parser.add_argument(
        '--img_suffix', type=str, default=".png",
        help='Suffix of images that will be processed.'
    )
    parser.add_argument(
        '--separator', type=str, default="&",
        help='Sign that you want file names and categories names to be separated with.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for current_dir in args.input_dir.iterdir():
        filename = current_dir.stem
        group_dir = args.output_dir / f'{args.input_dir.stem}{args.separator}{filename}' / filename
        group_dir.mkdir(parents=True, exist_ok=True)
        for file in current_dir.rglob(f'*{args.img_suffix}'):
            shutil.copy(file, group_dir / file.name)

