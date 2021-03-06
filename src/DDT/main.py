from DDT_cosal import DDT
import pathlib
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--train_dir', type=pathlib.Path,
        help='Path to directory containing training images.'
    )
    parser.add_argument(
        '--sal_dir', type=pathlib.Path,
        help='Path to directory containing saliency map generated by your method.'
    )   
    parser.add_argument(
        '--save_dir', type=pathlib.Path,
        help='Path to directory where you want to save result images.'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for dir in args.train_dir.iterdir():
        if dir.is_dir():
            print(f'Processing {dir.stem} category.')
            ddt = DDT(use_cuda=True)
            trans_vectors, descriptor_means = ddt.fit(dir)
            save_dir_dir = args.save_dir / dir.stem
            save_dir_dir.mkdir(exist_ok=True, parents=True)
            ddt.co_locate(args.train_dir / dir.stem, save_dir_dir, args.sal_dir / dir.stem, trans_vectors, descriptor_means)

