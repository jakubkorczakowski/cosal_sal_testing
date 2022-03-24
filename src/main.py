from DDT.DDT_cosal import DDT
import pathlib


train_dir = pathlib.Path('')
test_dir = train_dir
save_dir = pathlib.Path('')
sal_dir = pathlib.Path('')

if __name__ == "__main__":
    for dir in train_dir.iterdir():
        if dir.is_dir():
            print(f'Processing {dir.stem} category.')
            ddt = DDT(use_cuda=True)
            trans_vectors, descriptor_means = ddt.fit(dir)
            save_dir_dir = save_dir / dir.stem
            save_dir_dir.mkdir(exist_ok=True, parents=True)
            # print(f'Saving {dir} category.')
            ddt.co_locate(test_dir / dir.stem, save_dir_dir, sal_dir / dir.stem, trans_vectors, descriptor_means)

    