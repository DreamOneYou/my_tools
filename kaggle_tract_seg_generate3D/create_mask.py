#coding:utf-8
#主要用于实现从csv路径文件中生成3Dnii文件。
import os.path
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold
import SimpleITK as sitk

KAGGLE_DIR = Path(r"D:\compation\kaggle")
INPUT_DIR = KAGGLE_DIR / "train"
OUTPUT_DIR = KAGGLE_DIR/ "working"

INPUT_DATA_DIR =  KAGGLE_DIR / "train.csv"
N_SPLITS = 5
RANDOM_SEED = 2022

OUTPUT_DATA_DIR = OUTPUT_DIR
if not os.path.exists(OUTPUT_DATA_DIR):
    os.mkdir(OUTPUT_DATA_DIR)

train_df = pd.read_csv(INPUT_DATA_DIR / "train.csv")

DEBUG = True
if DEBUG:
    train_df = train_df.head(999)
    # print(train_df)

# -----------------From https://www.kaggle.com/code/awsaf49?scriptVersionId=93500248&cellId=18
def merge_ids(df):
    df["empty"] = df["segmentation"].map(lambda x: int(pd.isna(x)))
    # print("df2\n", df)
    df2 = df.groupby(["id"])["class"].agg(list).to_frame().reset_index()
    # print("df2\n",df2)
    df2 = df2.rename(columns={"class": "classes"})
    df2 = df2.merge(df.groupby(["id"])["segmentation"].agg(list), on=["id"])

    df = df.drop(columns=["segmentation", "class"])
    df = df.groupby(["id"]).head(1).reset_index(drop=True)
    df = df.merge(df2, on=["id"])

    return df



# ----------------------------Extract metadata from id
def extract_metadata_from_id(df):
    dfs = df
    dfs[["case_day", "slice"]] = dfs["id"].str.split("_slice", n=1, expand=True)
    s = dfs
    df[["case", "day", "slice"]] = df["id"].str.split("_", n=2, expand=True)
    df["case"] = df["case"].str.replace("case", "").astype(int)
    df["day"] = df["day"].str.replace("day", "").astype(int)
    df["case_and_day"] = s["case_day"]
    df["slice"] = df["slice"].str.replace("slice_", "").astype(int)

    return df



# -------------From https://www.kaggle.com/code/awsaf49?scriptVersionId=93500248&cellId=27
def create_folds(df, n_splits, random_seed):
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["empty"], groups=df["case"])):
        df.loc[val_idx, "fold"] = fold

    return df

#------------------------Add Scan Path
def add_scan_path(df, data_dir, stage="train"):
    scan_paths = [str(path) for path in (data_dir / stage).rglob("*.png")]

    path_df = pd.DataFrame(scan_paths, columns=["scan_path"])
    print(path_df.head())
    print(path_df["scan_path"][0].split("\\slice")[0])
    path_df = extract_metadata_from_path(path_df, stage)

    df = df.merge(path_df, on=["case", "day", "slice"])

    return df


def extract_metadata_from_path(path_df, stage="train"):
    path_df[["parent", "case_day", "scans", "file_name"]] = path_df["scan_path"].str.rsplit("\\", n=3, expand=True)

    path_df[["case", "day"]] = path_df["case_day"].str.split("_", expand=True)
    path_df["case"] = path_df["case"].str.replace("case", "")
    path_df["day"] = path_df["day"].str.replace("day", "")

    path_df[["slice", "width", "height", "spacing", "spacing_"]] = (
        path_df["file_name"].str.replace("slice_", "").str.replace(".png", "").str.split("_", expand=True)
    )
    path_df = path_df.drop(columns=["parent", "case_day", "scans", "file_name", "spacing_"])

    if stage == "test":
        path_df["id"] = "case" + path_df["case"] + "_day" + path_df["day"] + "_slice_" + path_df["slice"]

    numeric_cols = ["case", "day", "slice", "width", "height", "spacing"]
    path_df[numeric_cols] = path_df[numeric_cols].apply(pd.to_numeric)
    return path_df



def add_npy_paths(df, stage="train"):
    df["img_npy_path"] = df["scan_path"].str.replace(".png", ".npy").str.replace("scans", "imgs_npy")
    df["img_npy_path"] = df["img_npy_path"].str.replace("train", "working")

    if stage == "train":
        df["seg_npy_path"] = df["img_npy_path"].str.replace("imgs", "segs")

    return df
#---------------------保存nii文件
def create_nii(image_details,pred_path=""):
    case_no = image_details['case_and_day'].tolist()
    paths = image_details['scan_path'].tolist()
    segs = image_details['segmentation'].tolist()
    Width = image_details['width'].tolist()
    Height = image_details['height'].tolist()
    Case_no_And_Day = image_details['case_and_day'].tolist()
    i, j = 0, 0
    sor = []
    while i < len(case_no):
        file_app = []
        seg_app = []
        k = 0
        while j < len(case_no):
            if case_no[i] == case_no[j]:
                # print("{},{},{}".format(k, case_no[j], Case_no_And_Day[i]))
                file = paths[j]
                image = load_img_npy(file)
                seg = load_seg_npy(segs[j], Width[j], Height[j])

                # image = np.array(image)
                file_app.append(image)
                seg_app.append(seg)
                k += 1
                j += 1
            else:
                break
        T = len(file_app)
        file_in = np.zeros((T, Height[i], Width[i]))
        seg_in = np.zeros((T, Height[i], Width[i]))

        Large_Bowel =  np.zeros((Height[i], Width[i]))
        Small_Bowel =  np.zeros((Height[i], Width[i]))
        for s in range(T):
            # print("s:{},max:{},min:{},shape:{}".format(s, seg_app[s].max(), seg_app[s].min(), seg_app[s].shape))
            # print("s:{},shape:{}".format(s, file_app[s].shape))
            file_in[s, :, :] = file_app[s]
            label = np.zeros((Height[i], Width[i]))
            seg_Label = seg_app[s].transpose((2, 0, 1))
            # output = seg_Label.argmax(0)
            label[np.where(seg_Label[0] == 1)] = 1
            label[np.where(seg_Label[1] == 1)] = 2
            label[np.where(seg_Label[2] == 1)] = 3
            seg_in[s, :, :] = label
        # save image nii
        file_in = file_in.astype(np.uint16)
        predict_img = sitk.GetImageFromArray(file_in)
        # save seg nii
        seg_in = seg_in.astype(np.uint8)
        predict_seg = sitk.GetImageFromArray(seg_in)
        print("pred:", os.path.join(pred_path, Case_no_And_Day[i] + ".nii.gz"))
        os.makedirs(os.path.join(out_dir, Case_no_And_Day[i]), exist_ok=True)
        sitk.WriteImage(predict_img,os.path.join(pred_path, Case_no_And_Day[i], Case_no_And_Day[i] + "_org.nii.gz"))
        sitk.WriteImage(predict_seg,os.path.join(pred_path, Case_no_And_Day[i], Case_no_And_Day[i] + "_seg.nii.gz"))
        i = j
        sor.append(seg_in.shape[0])
    print(sorted(sor))

#-----------------Create NumPy Data
def save_img_and_seg(row):
    img_npy = load_img_npy(row.scan_path)
    save_array(row.img_npy_path, img_npy)

    seg_npy = load_seg_npy(row)
    save_array(row.seg_npy_path, seg_npy)


# From https://www.kaggle.com/code/awsaf49?scriptVersionId=93141541&cellId=12
def load_img_npy(scan_path):

    scan_path = str(scan_path)
    # print("scan:", scan_path)
    img = cv2.imread(scan_path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")  # original is uint16
    img = (img - img.min()) / (img.max() - img.min()) * 255.0  # scale image to [0, 255]
    img = img.astype("uint8")

    return img


def save_array(file_path, array):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


# Based on https://www.kaggle.com/code/awsaf49?scriptVersionId=93141541&cellId=10
def load_seg_npy(row, width, height):
    shape = (height, width, 3)
    mask = np.zeros(shape, dtype=np.uint8)

    for i, rle in enumerate(row):
        if not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
            # mask[np.where(mask==1)] = i+1
    return mask
    # return mask * 255


# From https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)  # Needed to align to RLE direction


# From https://www.kaggle.com/code/awsaf49?scriptVersionId=93141541&cellId=12
def show_img(img, seg=None, use_clahe=False):
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    plt.imshow(img, cmap="bone")

    if seg is not None:
        plt.imshow(seg, alpha=0.5)

        handles = [
            Rectangle((0, 0), 1, 1, color=_c) for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Stomach", "Large Bowel", "Small Bowel"]

        plt.legend(handles, labels)
    # plt.show()
    plt.axis("off")


# Slightly adapted from https://www.kaggle.com/code/awsaf49?scriptVersionId=93141541&cellId=19
def show_grid(train_df, nrows, ncols):
    fig, _ = plt.subplots(figsize=(5 * ncols, 5 * nrows))

    train_df_sampled = train_df[train_df["empty"] == 0].sample(n=nrows * ncols)
    for i, row in enumerate(train_df_sampled.itertuples()):
        img = np.load(row.img_npy_path)
        seg = np.load(row.seg_npy_path)

        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.title(row.id)

        show_img(img, seg, use_clahe=True)
    plt.show()
if  __name__ == "__main__":
    #----------------------------------Merge IDs-------------------------------
    train_df = merge_ids(train_df)

    #------------------------------------Extract metadata from id--------------
    train_df = extract_metadata_from_id(train_df)

    # ------------------------------------create_folds-------------------------
    train_df = create_folds(train_df, N_SPLITS, RANDOM_SEED)

    # ------------------------------------add_scan_path------------------------
    train_df = add_scan_path(train_df, KAGGLE_DIR, stage="train")
    train_df.to_csv("path_df.csv")

    # ------------------------------------add_npy_paths-------------------------
    train_df = add_npy_paths(train_df, stage="train")
    train_df.to_csv("path_df_npy.csv")

    # ------------------------------------save npy------------------------------
    # for row in tqdm(train_df.itertuples(), total=len(train_df)):
    #     save_img_and_seg(row)

    # ----------------------------------save nii--------------------------------
    # out_dir = r"D:\compation\kaggle_tract_seg_generate3D\3D_preprocess"
    # print(f"file_number:{len(train_df)}")
    # create_nii(train_df, out_dir)

    #----------------------展示图像和label结果-----------------------------------
    # nrows = 1
    # ncols = 4
    # show_grid(train_df, nrows, ncols)


