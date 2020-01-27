#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/18 22:47
# @Author   : WanDaoYi
# @FileName : prepare.py
# ============================================

from datetime import datetime
from utils import Utils
import os
from data_clean import DataClean
import shutil
from config import cfg


class PrepareData(object):

    def __init__(self):

        self.train_path = cfg.TRAIN.TRAIN_PATH
        self.test_path = cfg.TEST.TEST_DATA_PATH

        self.train_data_path = cfg.TRAIN.TRAIN_DATA_INFO_PATH
        self.val_data_path = cfg.TRAIN.VAL_DATA_INFO_PATH

        self.utils = Utils()

        self.data_clean = DataClean()

        self.train_percent = cfg.COMMON.TRAIN_PERCENT
        pass

    # 判断较短的字符串是否被较长的字符串包含，如果包含则返回 Ture，否则为 False
    # 如果至少有一个字符串为空，则也返回 Ture
    # 用来判断，国家，省份，城市是否同一个。如：str1=广州，str2=广州市
    def is_sampling(self, str1, str2):

        str_len1 = len(str1)
        str_len2 = len(str2)
        # 不空的时候，要进行匹配校验
        if str_len1 > 0 and str_len2 > 0:
            # 判断短字符是否在长字符内
            if str_len1 > str_len2:
                flag = str2 in str1
            else:
                flag = str1 in str2

            return flag
        # 有空的时候，无法判断是否是同一个地方，所以，进行采样
        # str_len1 == 0 or str_len2 == 0
        else:
            return True

    # 对两个来源的酒店数据进行量化
    def quantification_data(self, data_save_path, data_list, data_list2, train_flag=True):
        """
        :param data_save_path: 保存路径
        :param data_list: 标准数据集，xie_cheng
        :param data_list2: 供应商数据集，tai_tan or long_teng
        :param train_flag: 是否是训练数据，是为 True，测试为 False
        :return:
        """

        file_name = ""

        # xie_cheng 数据长度
        data_list_len = len(data_list)

        # 训练时候数据集，区分正负样本, 因为，会生成大量的负样本
        # 区分正负样本，为后面训练的时候采样做准备
        positive_samples = []
        negative_samples = []
        # 测试时候 数据集，不分正负样本
        sample_list = []
        count = 0
        save_num = 0

        # 用来命名文件序号用
        name_num = 1

        # i 用来计算 data_list 的第几个元素
        i = 0
        for data in data_list:

            i += 1

            hotel_source = data["hotel_source"]
            hotel_id = data["hotel_id"]
            hotel_name = data["hotel_name"]
            address = data["address"]
            telephone = data["telephone"]
            fax = data["fax"]
            lng = data["lng"]
            lat = data["lat"]
            country_name = data["country_name"]
            province_name = data["province_name"]
            city_name = data["city_name"]
            district_name = data["district_name"]
            postal_code = data["postal_code"]
            open_year = data["open_year"]

            # j 用来计算 data_list2 的第几个元素
            j = 0
            for data2 in data_list2:

                j += 1

                country_name2 = data2["country_name"]
                province_name2 = data2["province_name"]
                city_name2 = data2["city_name"]

                # 判断是否生成样本(同城市地区的才生成，如果城市地区为空的，也生成)
                country_flag = self.is_sampling(country_name, country_name2)
                province_flag = self.is_sampling(province_name, province_name2)
                city_flag = self.is_sampling(city_name, city_name2)
                # bool(1 - True) 布尔值取反，为 False；
                # 当国家、省份、或城市有一个不相同时，不生成量化数据
                if bool(1 - (country_flag and province_flag and city_flag)):
                    continue

                lng2 = data2["lng"]
                lat2 = data2["lat"]

                # 计算两酒店经纬度之间的距离
                if len(lng) > 0 and len(lat) > 0 and len(lng2) > 0 and len(lat2) > 0:
                    lng_lat_distance = self.utils.distance_computer(lng, lat, lng2, lat2)

                # 缺失某经纬度值，缺失值为 -1
                else:
                    lng_lat_distance = self.utils.missing_value

                # 如果两个酒店距离大于 1000m，就不生成比对数据。因为相距太远的酒店可能是同一家的可能性太低
                if lng_lat_distance > 1000:
                    # print("距离大于 1000m 的酒店: {} - {} 相差 {}m".format(hotel_name, hotel_name2, lng_lat_distance))
                    continue

                hotel_source2 = data2["hotel_source"]
                hotel_id2 = data2["hotel_id"]
                hotel_name2 = data2["hotel_name"]
                address2 = data2["address"]
                telephone2 = data2["telephone"]
                fax2 = data2["fax"]
                district_name2 = data2["district_name"]
                postal_code2 = data2["postal_code"]
                open_year2 = data2["open_year"]

                # 酒店名余弦相似度得分
                if len(hotel_name) > 0 and len(hotel_name2) > 0:
                    name_score = self.utils.cosine_similarity(hotel_name, hotel_name2)
                else:
                    name_score = self.utils.missing_value

                # 地址余弦相似度得分
                if len(address) > 0 and len(address2) > 0:
                    address_score = self.utils.cosine_similarity(address, address2)
                else:
                    address_score = self.utils.missing_value

                # 电话号码比较
                if len(telephone) > 0 and len(telephone2) > 0:
                    telephone_score = self.utils.number_compare(telephone, telephone2)
                else:
                    telephone_score = self.utils.missing_value

                # 传真号码比较
                if len(fax) > 0 and len(fax2) > 0:
                    fax_score = self.utils.number_compare(fax, fax2)
                else:
                    fax_score = self.utils.missing_value

                # 区域名余弦相似度得分
                if len(district_name) > 0 and len(district_name2) > 0:
                    district_score = self.utils.cosine_similarity(district_name, district_name2)
                else:
                    district_score = self.utils.missing_value

                # 邮政编码余弦相似度得分
                if len(postal_code) > 0 and len(postal_code2) > 0:
                    postal_code_score = self.utils.cosine_similarity(postal_code, postal_code2)
                else:
                    postal_code_score = self.utils.missing_value

                # 开业年余弦相似度得分
                if len(open_year) > 0 and len(open_year2) > 0:
                    open_year_score = self.utils.cosine_similarity(open_year, open_year2)
                else:
                    open_year_score = self.utils.missing_value

                pass

                # 训练集数据
                if train_flag:
                    # 训练数据，xie_cheng、long_teng、tai_tan 对应的行，是同一家酒店
                    if i == j:
                        label = 1
                    else:
                        label = 0
                    pass

                    sample = [hotel_source, hotel_id,
                              hotel_source2, hotel_id2,
                              name_score, address_score, lng_lat_distance,
                              telephone_score, fax_score, district_score,
                              postal_code_score, open_year_score, label
                              ]
                    if label == 1:
                        positive_samples.append(sample)
                    else:
                        negative_samples.append(sample)
                    pass

                # 测试集数据
                else:
                    sample = [hotel_source, hotel_id,
                              hotel_source2, hotel_id2,
                              name_score, address_score, lng_lat_distance,
                              telephone_score, fax_score, district_score,
                              postal_code_score, open_year_score
                              ]
                    sample_list.append(sample)
                    pass

            # 如果 sample_list 为空，则不做后面的操作
            if bool(1 - train_flag) and len(sample_list) == 0:
                print("{}生成对应的量化数据集为空: {}".format(data, sample_list))
                continue

            if save_num == 0:
                first_data = data_list[0]
                source_a = first_data["hotel_source"]

                second_data = data_list2[0]
                source_b = second_data["hotel_source"]

                n = 6 - len(str(name_num))
                file_name = source_a + "_to_" + source_b + "_" + str(0) * n + str(name_num) + ".txt"
                name_num += 1

                pass

            print("percent: {}/{}, file_name: {}".format(count, data_list_len, file_name))

            # 是否是训练
            if train_flag:

                # 设置每个文件保存 50W 条左右数据
                save_num += len(negative_samples) + len(positive_samples)

                positive_samples_path = os.path.join(data_save_path, "positive.txt")
                negative_samples_path = os.path.join(data_save_path, "negative.txt")

                self.utils.batch_data_save(positive_samples, positive_samples_path)
                self.utils.batch_data_save(negative_samples, negative_samples_path)

                positive_samples.clear()
                negative_samples.clear()
                pass
            else:

                # 设置每个文件保存 50W 条左右数据
                save_num += len(sample_list)

                save_path = os.path.join(data_save_path, file_name)
                self.utils.batch_data_save(sample_list, save_path)
                sample_list.clear()
                pass

            if save_num > 500000:
                save_num = 0

            count += 1
            pass
        pass

    # 生成数据
    def generate_data(self, data_list, train_flag=True):

        for list_info in data_list:
            data_info = list_info["data_info"]

            if train_flag:
                file_dumn_path = self.train_path
                pass
            else:
                province_pinyin = list_info["province_pinyin"]
                city_pinyin = list_info["city_pinyin"]
                file_dumn_path = os.path.join(self.test_path, province_pinyin + "/" + city_pinyin + "/")
                pass

            # 判断路径是否存在，如果不存在则创建
            if not os.path.exists(file_dumn_path):
                os.makedirs(file_dumn_path)

            for data in data_info:

                xie_cheng_data = data["xie_cheng"]
                long_teng_data = data["long_teng"]
                tai_tan_data = data["tai_tan"]

                print(len(xie_cheng_data))

                self.quantification_data(file_dumn_path, xie_cheng_data, long_teng_data, train_flag)
                self.quantification_data(file_dumn_path, xie_cheng_data, tai_tan_data, train_flag)
                pass
        pass

    # 将训练数据分割为训练集和验证集
    def train_val_split(self):

        positive_data_list = []
        negative_data_list = []

        # 正负样本文件路径
        positive_data_path = cfg.TRAIN.POSITIVE_TRAIN_DATA
        negative_data_path = cfg.TRAIN.NEGATIVE_TRAIN_DATA

        with open(positive_data_path, "r", encoding="utf-8") as file:
            data_info = file.readlines()
            data_list = [data.strip().split(",") for data in data_info]

            positive_data_list.extend(data_list)
            pass

        with open(negative_data_path, "r", encoding="utf-8") as file:
            data_info = file.readlines()
            data_list = [data.strip().split(",") for data in data_info]

            negative_data_list.extend(data_list)
            pass

        positive_sampling = cfg.TRAIN.POSITIVE_SAMPLING
        negative_sampling = cfg.TRAIN.NEGATIVE_SAMPLING

        # 正负样本采样量，对样本进行有放回采样
        sampling_list = self.utils.sampling_data(positive_data_list, negative_data_list, positive_sampling, negative_sampling)

        print("数据分割开始!")
        train_data, val_data = self.utils.data_split(sampling_list, self.train_percent)
        print("数据分割结束!")

        print("train_data_len: {}".format(len(train_data)))
        print("val_data_len: {}".format(len(val_data)))

        self.utils.batch_data_save(train_data, self.train_data_path)
        self.utils.batch_data_save(val_data, self.val_data_path)

        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.utcnow()
    print("开始时间: ", start_time)

    demo = PrepareData()

    # 如果之前存在数据，则将文件夹和数据一起删除
    if os.path.exists(demo.train_path):
        shutil.rmtree(demo.train_path)

    if os.path.exists(demo.test_path):
        shutil.rmtree(demo.test_path)

    # 生成训练数据
    train_data_list = demo.data_clean.deal_train_data()
    demo.generate_data(train_data_list)
    demo.train_val_split()

    # 生产预测数据
    test_data_list = demo.data_clean.deal_test_data()
    demo.generate_data(test_data_list, train_flag=False)

    # 代码结束时间
    end_time = datetime.utcnow()
    print("结束时间: ", end_time, ", 训练模型耗时: ", end_time - start_time)
    pass

