from DDT_cosal import DDT
import pathlib


train_dir = pathlib.Path('/home/korczakowski/data/CoCA/image')
save_dir = pathlib.Path('/home/korczakowski/cosal_sal_testing/data/results/DDT_results/CoCA')
sal_dir = pathlib.Path('/home/korczakowski/data/LDF_Co2/CoCA')

if __name__ == "__main__":
    for dir in train_dir.iterdir():
        if dir.is_dir():
            print(f'Processing {dir.stem} category.')
            ddt = DDT(use_cuda=True)
            trans_vectors, descriptor_means = ddt.fit(dir)
            save_dir_dir = save_dir / dir.stem
            save_dir_dir.mkdir(exist_ok=True, parents=True)
            ddt.co_locate(train_dir / dir.stem, save_dir_dir, sal_dir / dir.stem, trans_vectors, descriptor_means)

