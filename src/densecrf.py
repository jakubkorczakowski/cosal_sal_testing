import denseCRF
import numpy as np
from PIL import Image
import pathlib

def densecrf(I, P, param):
    """
    input parameters:
        I    : a numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
        P    : a probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a numpy array of shape [H, W], where pixel values represent class indices. 
    """
    out = denseCRF.densecrf(I, P, param) 
    return out   

def convert_label_to_probability_map(label, color_list):
    [H, W, _] = label.shape
    C = len(color_list)
    prob  = np.zeros([H, W, len(color_list)], np.float32) 
    for h in range(H):
        for w in range(W):
            ca = label[h, w, :]
            if sum(ca) == 0:
                for c in range(C):
                    prob[h, w, c] = 1.0 / C
            else:
                for c in range(C):
                    cb = color_list[c]
                    if(ca[0]==cb[0] and ca[1]==cb[1] and ca[2]==cb[2]):
                        prob[h, w, c] = 1.0
                        break
    return prob 

def colorize_label_map(label, color_list):
    [H, W] = label.shape
    out = np.zeros((H, W, 3), np.uint8)
    for h in range(H):
        for w in range(W):
            idx = label[h, w] 
            color = np.asarray(color_list[idx])
            out[h, w, :] = color
    return out

def apply_crf_old(image_path, map_path, save_path):
    I  = Image.open(image_path)
    Iq = np.asarray(I)
    L  = Image.open(map_path)
    Lq = np.asarray(L, np.float32) / 255
    prob = Lq[:, :, :2]
    prob[:, :, 0] = 1.0 - prob[:, :, 0]

    w1    = 10.0  # weight of bilateral term
    alpha = 80    # spatial std
    beta  = 13    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 5.0   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    lab = densecrf(Iq, prob, param)
    lab = Image.fromarray(lab*255)
    lab = lab.convert('RGB')
    lab.save(save_path / image_path.parts[-2] / image_path.parts[-1], 'JPEG')

def apply_crf(image_path, map_array, save_path):
    I  = Image.open(image_path)
    Iq = np.asarray(I)
    Lq = map_array
    prob = Lq[:, :, :2]
    prob[:, :, 0] = 1.0 - prob[:, :, 0]

    w1    = 10.0  # weight of bilateral term
    alpha = 80    # spatial std
    beta  = 13    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 5.0   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    lab = densecrf(Iq, prob, param)
    return lab


if __name__ == "__main__":
    # provide your config values
    image_dir = pathlib.Path("")
    map_dir = pathlib.Path("")
    save_dir = pathlib.Path("")
    for file in image_dir.rglob('*.jpg'):
        (save_dir / file.parts[-2]).mkdir(exist_ok=True)
        apply_crf(file, map_dir / file.parts[-2] / file.parts[-1], save_dir)
