##kaggle_tract_seg_generate3D
这个文件主要是用于kaggle：UW-Madison GI Tract Image Segmentation比赛数据处理用的。官方给的数据是2D png图像，所以我试着将其转换为3D图像进行分割。
主要包括三个文件，分别是train、sample_submission.csv、train.csv三个文件。
具体文件格式：\
--train \
&emsp; --case2 \
&emsp;&emsp;&emsp;--ase2_day1 \
&emsp;&emsp;&emsp;&emsp;&emsp;--scans \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;slice_0001_266_266_1.50_1.50.png \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;slice_0002_266_266_1.50_1.50.png \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;... \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;slice_0144_266_266_1.50_1.50.png

train.csv文件格式 \
![train.csv](kaggle_tract_seg_generate3D/train.png)