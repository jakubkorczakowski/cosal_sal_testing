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
    for file in args.input_dir.rglob(f'*{args.img_suffix}'):
        group_name = file.name.split(args.separator)[0]
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        filename = file.name.split(args.separator)[-1]
        dest = pathlib.Path(args.output_dir / group_name / filename)
        shutil.copy(file, dest)
