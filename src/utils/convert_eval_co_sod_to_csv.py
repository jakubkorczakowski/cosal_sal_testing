import pandas as pd
import numpy as np
import pathlib
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--results', type=pathlib.Path,
        help='Path to directory containing input txt.'
    )
    parser.add_argument(
        '--results_csv', type=pathlib.Path,
        help='Path to directory where you want to save result csv.'
    )   

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    first_line = "dataset,method,mae,max-fm,mean-fm,max-Emeasure,mean-Emeasure,S-measure,AP,AUC\n"

    with open(args.results) as results_file, open(args.results_csv, "w") as results_csv_file:
        results_csv_file.write(first_line)
        for line in results_file.readlines():
            line = line\
                    .replace('|', '')\
                    .replace('mae', ',')\
                    .replace('max-fm', ',')\
                    .replace('mean-fm', ',')\
                    .replace('max-Emeasure', ',')\
                    .replace('mean-Emeasure', ',')\
                    .replace('S-measure', ',')\
                    .replace('AP', ',')\
                    .replace('AUC.', '')\
                    .replace(':', ',')\
                    .replace('(', ',')\
                    .replace(')', '')\
                    .replace(' ', '')
            results_csv_file.write(line)