# coding:utf-8
from glob import glob
import os
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import imageio
import pandas as pd
DATASET_FOLDER = "D:\compation\kaggle"
def generate_2D_png():
    path = glob(r"D:\compation\kaggle\3D_preprocess\a\*") # 获取到该文件夹下所有的标签（3D nii文件）
    save = Path(r"D:\compation\kaggle\3D_preprocess")  # 保存路径
    for i in range(len(path)):
        file = path[i]
        file_name = file.split("\\")[-1].split("_seg")[0]
        case = file_name.split("_")[0]
        print("case:{}, file_name:{}".format(case, file_name))
        seg = sitk.ReadImage(file)
        seg = sitk.GetArrayFromImage(seg)
        for j in range(seg.shape[0]):
            # png = np.zeros((seg.shape[1:]))
            # print("j:",j)
            save_path = save / case / file_name /"scans"
            name = "slice_"+str(j)+str(seg.shape[1]) +"_"+str(seg.shape[2])+"_"+str(1.5)+"_"+str(1.5)
            output = seg[j, ...]
            Snapshot_img = np.zeros(shape=(seg.shape[1],seg.shape[2],3), dtype=np.uint8)  # png设置为3通道
            Snapshot_img[:, :, 0][np.where(output == 1)] = 255   #我们也有3个标签，其中值分别为1,2,3，所以我们需要给每个标签都赋予不同的通道
            Snapshot_img[:, :, 1][np.where(output == 2)] = 255
            Snapshot_img[:, :, 2][np.where(output == 3)] = 255
            os.makedirs(save_path, exist_ok=True)
            imageio.imwrite(os.path.join(save_path, name + '.png'), Snapshot_img[:, :, :])

def enrich_data(df, sdir="train"):
    imgs = glob(os.path.join(DATASET_FOLDER, sdir, "case*", "case*_day*", "scans", "*.png"))
    img_folders = [os.path.dirname(p).split(os.path.sep) for p in imgs]
    img_names = [os.path.splitext(os.path.basename(p))[0].split("_") for p in imgs]
    img_keys = [f"{f[-2]}_slice_{n[1]}" for f, n in zip(img_folders, img_names)]

    # print(img_keys[:5])
    df["img_path"] = df["id"].map({k: p for k, p in zip(img_keys, imgs)})
    df["Case_Day"] = df["id"].map({k: f[-2] for k, f in zip(img_keys, img_folders)})
    df["Case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["Day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["Slice"] = df["id"].map({k: int(n[1]) for k, n in zip(img_keys, img_names)})
    df["width"] = df["id"].map({k: int(n[2]) for k, n in zip(img_keys, img_names)})
    df["height"] = df["id"].map({k: int(n[3]) for k, n in zip(img_keys, img_names)})
    df["spacing1"] = df["id"].map({k: float(n[4]) for k, n in zip(img_keys, img_names)})
    df["spacing2"] = df["id"].map({k: float(n[5]) for k, n in zip(img_keys, img_names)})

def rle_encode(mask, bg = 0) -> dict:
    vec = mask.flatten()
    nb = len(vec)
    where = np.flatnonzero
    starts = np.r_[0, where(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 2]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = {}
    for start, length, val in zip(starts, lengths, values):
        if val == bg:
            continue
        rle[val] = rle.get(val, []) + [str(start), length]
    # post-processing
    rle = {lb: " ".join(map(str, id_lens)) for lb, id_lens in rle.items()}
    return rle

def generate_rel(LABELS, path):
    preds = []

    for i in range(len(path)):
        file = path[i]
        file_name = file.split("\\")[-1].split("_seg")[0]
        case = file_name.split("_")[0]
        print("case:{}, file_name:{}".format(case, file_name))
        seg = sitk.ReadImage(file)
        seg = sitk.GetArrayFromImage(seg)
        for j in range(seg.shape[0]):
            if j>=0 and j<9:
                number = str(0)+str(0)+str(0)+str(j+1)
            elif j>=9 and j<99:
                number = str(0)+str(0) + str(j+1)
            else:
                number = str(0) + str(j+1)
            name = file_name+"_slice_"+number
            output = seg[j, ...]
            Snapshot_img = np.zeros(shape=(seg.shape[1],seg.shape[2],3), dtype=np.uint8)  # png设置为3通道
            Snapshot_img[:, :, 0][np.where(output == 1)] = 1   #我们也有3个标签，其中值分别为1,2,3，所以我们需要给每个标签都赋予不同的通道
            Snapshot_img[:, :, 1][np.where(output == 2)] = 1
            Snapshot_img[:, :, 2][np.where(output == 3)] = 1
            rle_lb = rle_encode(Snapshot_img[:, :, 0]) if np.sum(Snapshot_img[:, :, 0]) > 1 else {}
            rle_sb = rle_encode(Snapshot_img[:, :, 1]) if np.sum(Snapshot_img[:, :, 1]) > 1 else {}
            rle_sto = rle_encode(Snapshot_img[:, :, 2]) if np.sum(Snapshot_img[:, :, 2]) > 1 else {}
            index = (0,1,2)
            rel = [rle_lb, rle_sb, rle_sto]
            preds += [{"id": name, "class": lb, "predicted": rle.get(1, "")} for i, rle, lb in zip(index, rel, LABELS)]
        df_pred = pd.DataFrame(preds)
        df_pred.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    # df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
    # df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv","traines.csv"))
    # enrich_data(df_ssub,"traines")
    # df_ssub.to_csv("df.csv")
    # print(df_ssub["Case_Day"][4])
    # generate_2D_png()
    #
    pred_file = glob(r"D:\compation\kaggle\3D_preprocess\a\*")  # 获取到该文件夹下所有的标签（3D nii文件）
    LABELS = ("large_bowel", "small_bowel", "stomach")
    generate_rel(LABELS, pred_file)