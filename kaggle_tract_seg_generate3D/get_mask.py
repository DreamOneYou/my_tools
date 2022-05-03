# coding:utf-8
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from glob import glob
import warnings
import SimpleITK as sitk
warnings.filterwarnings('ignore')


def show_image(list_images, show=False, create_nii=False):
    # 只展示有分割像素值的case有多少。
    # sb.countplot(df[df['segmentation'].notnull()]['class'])
    # plt.show()

    pred_path = r"D:\compation\kaggle\3D_preprocess"

    image_details = pd.DataFrame({'Path':list_images})
    splits = image_details['Path'].str.split("\\", n = 10, expand = True)
    image_details['Case_no_And_Day'] = splits[8]
    image_details['Slice_Info'] = splits[10]


    splits = image_details['Case_no_And_Day'].str.split("_", n = 2, expand = True)
    image_details['Case_no'] = splits[0].str[4:].astype(int)
    image_details['Day'] = splits[1].str[3:].astype(int)

    # print(image_details.head())
    splits = image_details['Slice_Info'].str.split("_", n = 5, expand = True)
    image_details['Slice_no'] = splits[1].astype(int)
    image_details['Width'] = splits[2].astype(int)
    image_details['Height'] = splits[3].astype(int)
    image_details['Pixel1'] = splits[4].astype(float)
    image_details['Pixel2'] = splits[5].str[:-4].astype(float)
    # image_details.to_csv("train_information.csv")
    # print(uni_case)
    if create_nii:
        case_no = image_details['Case_no_And_Day'].tolist()
        paths = image_details['Path'].tolist()
        Width = image_details['Width'].tolist()
        Height = image_details['Height'].tolist()
        Case_no_And_Day = image_details['Case_no_And_Day'].tolist()
        i, j = 0, 0
        while i < len(case_no):
            file_app = []
            k = 0
            while j < len(case_no):
                if case_no[i] == case_no[j]:
                    # print("{},{},{}".format(k, case_no[j], Case_no_And_Day[i]))
                    file = paths[j]
                    image = Image.open(file)
                    image = np.array(image)
                    file_app.append(image)
                    k += 1
                    j += 1
                else:
                    break
            T = len(file_app)
            file_in = np.zeros((T, Height[i], Width[i]))
            for s in range(T):
                print("s:{},shape:{}".format(s, file_app[s].shape))
                file_in[s, :, :] = file_app[s]
            file_in = file_in.astype(np.uint16)
            predict_seg = sitk.GetImageFromArray(file_in)
            print("pred:", os.path.join(pred_path, Case_no_And_Day[i] + ".nii.gz"))
            sitk.WriteImage(predict_seg,
                            os.path.join(pred_path, Case_no_And_Day[i] + ".nii.gz"))
            i = j


    # print('Height of the images having 1.63 pixel spacing are ==>>'
    #       ,list(image_details[image_details['Pixel1']==1.63]['Height'].unique()))
    # print('Height of the images having 1.5 pixel spacing are  ==>>'
    #       ,list(image_details[image_details['Pixel1']==1.5]['Height'].unique()))

    # for col in image_details.loc[:,'Case_no':'Height']:
    #   k = len(image_details[col].unique())
    #   print(f'{col} has {k} unique items.')
    #   print(image_details[col].unique())
    #   print()
    #-----------------------------------展示原始图像-------------------------------------------
    if show:
        plt.subplots(figsize=(15, 15))
        for i in range(12):
            index = np.random.randint(0, image_details.shape[0])
            image = Image.open(image_details.loc[index, 'Path'])
            image = np.array(image)

            plt.subplot(3, 4, i + 1)

            title = (image_details.loc[index, 'Case_no_And_Day'] +
                     '_Slice_no_' + str(image_details.loc[index, 'Slice_no']))

            plt.title(title)
            # plt.imshow(np.interp(image, [np.min(image), np.max(image)], [0, 255]))
            plt.imshow(image / 65535)
            # plt.imshow(image / image.max())  #This will also serve the purpose.

        plt.show()
        return  image_details
    else:
        return image_details
  #-----------------------------------展示原始图像-------------------------------------------

#-----------------------------------展示分割图像-------------------------------------------
# 获取训练集csv路径，并进行读取
def get_pixel_loc(rle_string, img_shape):
  rle = [int(i) for i in rle_string.split(' ')]
  pairs = list(zip(rle[0::2], rle[1::2]))

  # This for loop will help to understand better the above command.
  # pairs = []
  # for i in range(0, len(rle), 2):
  #   a.append((rle[i], rle[i+1])

  p_loc = []  # Pixel Locations

  for start, length in pairs:
    for p_pos in range(start, start + length):
      p_loc.append((p_pos % img_shape[1], p_pos // img_shape[0]))

  return p_loc


def get_mask(mask, img_shape):
  canvas = np.zeros(img_shape).T
  canvas[tuple(zip(*mask))] = 1

  # This is the Equivalent for loop of the above command for better understanding.
  # for pos in range(len(p_loc)):
  #   canvas[pos[0], pos[1]] = 1

  return canvas.T


def apply_mask(image, mask, img_shape):
  image = image / image.max()
  image = np.dstack((image, get_mask(mask, img_shape), get_mask(mask, img_shape)))

  return image

def show_mask(index_list,image_details, mask_data,show=True, create_nii=False ):

    if show:
      for i in range(5):
        index = index_list[np.random.randint(0, len(index_list) - 1)]

        curr_id = mask_data.loc[index, 'id']
        class_of_scan = mask_data.loc[index, 'class']

        splits = curr_id.split('_')
        x = image_details[(image_details['Case_no'] == int(splits[0][4:]))
                          & (image_details['Day'] == int(splits[1][3:]))
                          & (image_details['Slice_no'] == int(splits[3]))]

        image = np.array(Image.open(x['Path'].values[0]))
        k = image.shape

        rle_string = mask_data.loc[index, 'segmentation']
        p_loc = get_pixel_loc(rle_string, k)

        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].set_title('Image')
        ax[0].imshow(image)

        ax[1].set_title('Mask')
        ax[1].imshow(get_mask(p_loc, k))

        ax[2].set_title(f'{class_of_scan} Segmented')
        ax[2].imshow(apply_mask(image, p_loc, k))
        plt.show()

      plt.show()
    else:
        pass

if __name__ == "__main__":
  df = pd.read_csv(r'D:/compation/kaggle/train.csv/train.csv')  # 路径文件
  list_images = glob(r'D:\\compation\\kaggle\\train\\*\\*\\scans\\*.png')  # 利用glob列出所有.png文件
  # print(df.head())
  mask_data_not_null = df[df['segmentation'].notnull()]  # 只提取有分割结果的mask
  print(mask_data_not_null)
  index_list = list(mask_data_not_null.index)
  image_details = show_image(list_images,show=True,create_nii=True)   # 展示image,可以选择是否保存成nii文件（3D文件）
  show_mask(index_list, image_details,mask_data_not_null, show=True)   #展示mask，暂时不知道怎么搞成3D，有会的帮忙补充一下。

