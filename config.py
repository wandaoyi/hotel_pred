#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/18 22:47
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

from easydict import EasyDict as edict
import os

__C = edict()

cfg = __C

# common options 公共配置文件
__C.COMMON = edict()
# windows 获取文件绝对路径, 方便 windows 在黑窗口 运行项目
__C.COMMON.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# # 获取当前窗口的路径, 当用 Linux 的时候切用这个，不然会报错。(windows也可以用这个)
# __C.COMMON.BASE_PATH = os.getcwd()

# 训练集，验证集，测试集占的百分比
__C.COMMON.TRAIN_PERCENT = 0.9
__C.COMMON.VAL_PERCENT = 0.1

# 酒店来源
__C.COMMON.HOTEL_SOURCE_LIST = ["xie_cheng", "long_teng", "tai_tan"]

# 原始训练数据路径
__C.COMMON.ORIGINAL_TRAIN_DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "dataset/original_data/train_data")
# 原始测试数据路径
__C.COMMON.ORIGINAL_TEST_DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "dataset/original_data/test_data")
# 清晰后数据路径
__C.COMMON.CLEAN_DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "dataset/cleaned_data")

# 训练集，验证集，测试集占的百分比
__C.COMMON.TRAIN_PERCENT = 0.7
__C.COMMON.VAL_PERCENT = 0.3

# 模型训练配置文件
__C.TRAIN = edict()

# 是否绘制 ROC 曲线，绘制为 True
__C.TRAIN.ROC_FLAG = True

__C.TRAIN.POSITIVE_SAMPLING = 1000
__C.TRAIN.NEGATIVE_SAMPLING = 3000

# prepare 好的训练数据路径
__C.TRAIN.TRAIN_PATH = os.path.join(__C.COMMON.BASE_PATH, "info/train/")

__C.TRAIN.POSITIVE_TRAIN_DATA = os.path.join(__C.COMMON.BASE_PATH, "info/train/positive.txt")
__C.TRAIN.NEGATIVE_TRAIN_DATA = os.path.join(__C.COMMON.BASE_PATH, "info/train/negative.txt")

__C.TRAIN.TRAIN_DATA_INFO_PATH = os.path.join(__C.TRAIN.TRAIN_PATH, "train_data.txt")
__C.TRAIN.VAL_DATA_INFO_PATH = os.path.join(__C.TRAIN.TRAIN_PATH, "val_data.txt")

# 模型保存路径
__C.TRAIN.MODEL_SAVE_PATH = os.path.join(__C.COMMON.BASE_PATH, "models/model_")

# 模型预测配置文件
__C.TEST = edict()
# 使用 acc 高的模型，当模型 acc 大于 0.5 时，用 True，否则用 False
__C.TEST.ACC_FLAG = False

# prepare 好的预测数据路径
__C.TEST.TEST_DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "info/test/")
# 预测结果输出路径
__C.TEST.TEST_OUTPUT_PATH = os.path.join(__C.COMMON.BASE_PATH, "outputs/")
__C.TEST.TEST_STANDARD_HEADER = ["标准酒店来源", "标准酒店id", "标准酒店名", "标准酒店地址",
                                 "标准酒店电话", "标准酒店国家", "标准酒店省", "标准酒店市",
                                 "标准酒店区", "标准酒店经度", "标准酒店纬度"
                                 ]
__C.TEST.TEST_SUPPLIER_HEADER = ["供应商酒店来源", "供应商酒店id", "供应商酒店名",
                                 "供应商酒店地址", "供应商酒店电话", "供应商酒店国家",
                                 "供应商酒店省", "供应商酒店市", "供应商酒店区",
                                 "供应商酒店经度", "供应商酒店纬度"
                                 ]
__C.TEST.TEST_DISTANCE_HEADER = ["两酒店的经纬度距离"]
# 模型路径
__C.TEST.MODEL_PATH = os.path.join(__C.COMMON.BASE_PATH, "models/model_acc=0.996667.m")
