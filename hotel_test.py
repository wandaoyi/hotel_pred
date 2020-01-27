#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/18 22:48
# @Author   : WanDaoYi
# @FileName : hotel_test.py
# ============================================

from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import os
import shutil
from utils import Utils
from config import cfg


class HotelPred(object):

    def __init__(self):

        self.test_data_path = cfg.TEST.TEST_DATA_PATH

        self.clean_data_path = cfg.COMMON.CLEAN_DATA_PATH

        self.pred_data_output = cfg.TEST.TEST_OUTPUT_PATH

        self.standard_header = cfg.TEST.TEST_STANDARD_HEADER
        self.supplier_header = cfg.TEST.TEST_SUPPLIER_HEADER
        self.distance_header = cfg.TEST.TEST_DISTANCE_HEADER

        # 对标准数据的表头进行拼接
        self.standard_header.extend(self.supplier_header)
        self.standard_header.extend(self.distance_header)

        self.acc_flag = cfg.TEST.ACC_FLAG
        self.model = joblib.load(cfg.TEST.MODEL_PATH)

        self.utils = Utils()
        pass

    # 读取数据
    def read_data(self, data_path):
        with open(data_path, "r") as file:
            data_info = file.readlines()
            data_list = [data.strip().split(",") for data in data_info]

            data_list = np.array(data_list)
            data_source = data_list[:, :4]
            data_info = data_list[:, 4:]

            data_info = np.array(data_info, dtype=float)
            # print(data_info[: 5])

            # ravel() 是列转行，用于解决数据转换警告。
            return data_source.tolist(), data_info

        pass

    def predict_data(self):

        # 每个城市预测正确的数据
        pred_city_data_list = []

        # 如果之前存在数据，则将文件夹和数据一起删除
        if os.path.exists(self.pred_data_output):
            shutil.rmtree(self.pred_data_output)

        # 获取省份名字list
        province_list = os.listdir(self.test_data_path)
        for province in province_list:
            # 获取城市名字list
            province_path = os.path.join(self.test_data_path, province)
            city_list = os.listdir(province_path)

            for city in city_list:
                # 获取文件名字list
                city_path = os.path.join(province_path, city)
                file_list = os.listdir(city_path)

                # 清洗后城市文件名
                object_clean_path = os.path.join(self.clean_data_path, "test/" + province + "/" + city)

                # 获取清晰后的 xie_cheng 数据
                xie_cheng_clean_path = os.path.join(object_clean_path, "xie_cheng.txt")
                xie_cheng_data_list = self.utils.read_txt2json_data(xie_cheng_clean_path)

                # 获取清晰后的 long_teng 数据
                long_teng_clean_path = os.path.join(object_clean_path, "long_teng.txt")
                long_teng_data_list = self.utils.read_txt2json_data(long_teng_clean_path)

                # 获取清晰后的 tai_tan 数据
                tai_tan_clean_path = os.path.join(object_clean_path, "tai_tan.txt")
                tai_tan_data_list = self.utils.read_txt2json_data(tai_tan_clean_path)

                print(xie_cheng_data_list[: 2])

                # 给三个来源的酒店数据以 酒店 id 建立索引
                clean_data_dir = {}
                clean_data_dir = self.deal_clean_data_dir(xie_cheng_data_list, clean_data_dir)
                clean_data_dir = self.deal_clean_data_dir(long_teng_data_list, clean_data_dir)
                clean_data_dir = self.deal_clean_data_dir(tai_tan_data_list, clean_data_dir)

                for file_name in file_list:
                    # 单个文件的路径
                    data_path = os.path.join(city_path, file_name)

                    data_source, data_info = self.read_data(data_path)
                    y_pred_list = self.model.predict(data_info)
                    y_pred_list_len = len(y_pred_list)

                    for i in range(0, y_pred_list_len):
                        y_pred = y_pred_list[i]
                        if y_pred == 1.0:
                            source = data_source[i]
                            source.append(y_pred)
                            pred_city_data_list.append(source)

                print(pred_city_data_list)

                xie_cheng_result, long_teng_result, tai_tan_result = self.deal_result_data(pred_city_data_list,
                                                                                           clean_data_dir)

                output_path = os.path.join(self.pred_data_output, province + "/" + city)
                # 判断路径是否存在，如果不存在则创建
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                xie_cheng_output_path = os.path.join(output_path, "xie_cheng.csv")
                long_teng_output_path = os.path.join(output_path, "long_teng.csv")
                tai_tan_output_path = os.path.join(output_path, "tai_tan.csv")

                print(self.standard_header)
                print(self.supplier_header)

                xie_cheng_samples = pd.DataFrame(columns=self.standard_header, data=xie_cheng_result)
                xie_cheng_samples.to_csv(xie_cheng_output_path, mode='a+', index=False, encoding="utf-8-sig")

                long_teng_samples = pd.DataFrame(columns=self.supplier_header, data=long_teng_result)
                long_teng_samples.to_csv(long_teng_output_path, mode='a+', index=False, encoding="utf-8-sig")

                tai_tan_samples = pd.DataFrame(columns=self.supplier_header, data=tai_tan_result)
                tai_tan_samples.to_csv(tai_tan_output_path, mode='a+', index=False, encoding="utf-8-sig")

    # 将字典部分信息转为list
    def deal_hotel_data(self, hotel_data_dir):

        hotel_source = hotel_data_dir["hotel_source"]
        hotel_id = hotel_data_dir["hotel_id"]
        hotel_name = hotel_data_dir["hotel_name"]
        address = hotel_data_dir["address"]
        telephone = hotel_data_dir["telephone"]
        country = hotel_data_dir["country_name"]
        province = hotel_data_dir["province_name"]
        city = hotel_data_dir["city_name"]
        district = hotel_data_dir["district_name"]
        lng = hotel_data_dir["lng"]
        lat = hotel_data_dir["lat"]

        data_list = [hotel_source, hotel_id, hotel_name, address,
                     telephone, country, province, city, district,
                     lng, lat]

        return data_list

    # 处理结果数据
    def deal_result_data(self, pred_city_data_list, clean_data_dir):
        xie_cheng_result = []
        long_teng_result = []
        tai_tan_result = []

        # 获取已经匹配的数据 的 id，用于后面数据整合
        xie_cheng_same = []
        long_teng_same = []
        tai_tan_same = []

        for pred_data in pred_city_data_list:
            # ['xie_cheng', '393346', 'long_teng', '450327', 1.0]
            xie_cheng_source = pred_data[0]
            xie_cheng_id = pred_data[1]

            hotel_source = pred_data[2]
            hotel_id = pred_data[3]

            # 获取已经匹配的数据 的 id，用于后面数据整合
            xie_cheng_same.append(xie_cheng_id)
            if "long_teng".__eq__(hotel_source):
                long_teng_same.append(hotel_id)
            else:
                tai_tan_same.append(hotel_id)

            # 标准库携程信息
            xie_cheng_dir = clean_data_dir[xie_cheng_source]
            xie_cheng_data = xie_cheng_dir[xie_cheng_id]

            xie_cheng_data_list = self.deal_hotel_data(xie_cheng_data)

            # 供应商其他方
            hotel_dir = clean_data_dir[hotel_source]
            hotel_data = hotel_dir[hotel_id]

            hotel_data_list = self.deal_hotel_data(hotel_data)

            lng = xie_cheng_data["lng"]
            lat = xie_cheng_data["lat"]
            lng2 = hotel_data["lng"]
            lat2 = hotel_data["lat"]

            distance = self.utils.distance_computer(lng, lat, lng2, lat2)

            xie_cheng_data_list.extend(hotel_data_list)
            xie_cheng_data_list.append(distance)

            xie_cheng_result.append(xie_cheng_data_list)

        # 遍历数据清洗的字典信息：有 携程、龙腾、泰坦
        for source_key in clean_data_dir:

            data_info_dir = clean_data_dir[source_key]
            for hotel_id_key in data_info_dir:
                data_list = self.deal_hotel_data(data_info_dir[hotel_id_key])
                # 分别携程没匹配的数据添加到集合后面去
                if "xie_cheng".__eq__(source_key):
                    if hotel_id_key not in xie_cheng_same:
                        data_len = len(data_list)
                        for i in range(0, data_len + 1):
                            data_list.append("")
                        xie_cheng_result.append(data_list)
                        pass

                # 将龙腾 和 泰坦 没匹配的数据另外用集合接收
                elif "long_teng".__eq__(source_key):
                    if hotel_id_key not in long_teng_result:
                        long_teng_result.append(data_list)
                    pass
                elif "tai_tan".__eq__(source_key):
                    if hotel_id_key not in tai_tan_result:
                        tai_tan_result.append(data_list)
                    pass
                pass
        # 返回三个来源的数据信息，其中携程前面部分为匹配数据，后面部分为没匹配的数据。
        # 龙腾 和 泰坦 的数据为没匹配的信息
        return xie_cheng_result, long_teng_result, tai_tan_result
        pass

    # 给清洗好的数据添加索引,每个来源对应的 hotel_id 是唯一的
    def deal_clean_data_dir(self, clean_data_list, clean_data_dir):

        data_source = clean_data_list[0]["hotel_source"]
        data_dir = {}
        for clean_data in clean_data_list:
            hotel_id = clean_data["hotel_id"]
            data_dir.update({str(hotel_id): clean_data})
            pass
        clean_data_dir.update({data_source: data_dir})
        return clean_data_dir


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.utcnow()
    print("开始时间: ", start_time)

    demo = HotelPred()
    demo.predict_data()

    # 代码结束时间
    end_time = datetime.utcnow()
    print("结束时间: ", end_time, ", 训练模型耗时: ", end_time - start_time)
    pass
