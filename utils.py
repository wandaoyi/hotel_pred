#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/18 22:48
# @Author   : WanDaoYi
# @FileName : utils.py
# ============================================

from datetime import datetime
import numpy as np
from pypinyin import lazy_pinyin
from math import sin, asin, cos, radians, fabs, sqrt
import jieba
import re
import json
import ast


class Utils(object):

    def __init__(self):

        # 地球半径 6371.393km
        self.radius = 6371393

        # 丢失值默认为 -1.0
        self.missing_value = -1.0

        # 对中文字符匹配的正则表达式
        self.pattern = re.compile(r'[\u4e00-\u9fa5]')
        pass

    # 将中文名转为拼音名
    def chinese_2_pinyin(self, chinese_str):
        pinyin_str = ""
        chinese_str_list = lazy_pinyin(chinese_str)
        pinyin_str_len = len(chinese_str_list)
        for i in range(0, pinyin_str_len):
            if len(pinyin_str) > 0:
                pinyin_str += "_"
            pinyin_str += chinese_str_list[i]
        return pinyin_str

    # string 不含中文(去除字符串中的中文)
    def str_un_chinese(self, old_str):

        new_str = ""
        if len(old_str) > 0:
            un_chinese = re.sub(self.pattern, ",", old_str)
            un_chinese_list = un_chinese.split(",")
            # len(word) > 0 去除单符号; len(word) > 3 此处去掉 号码 的短数字
            un_chinese_cut = [word for word in un_chinese_list if len(word) > 3]

            new_str = ""
            str_len = len(un_chinese_cut)
            for i in range(str_len):
                new_str += un_chinese_cut[i]
                if i == str_len - 1:
                    break

                new_str += ","
            pass

        return new_str

    # string 标准化，将特殊符号都转成 ',' 为分词做准备
    def str_standard(self, old_str):
        new_str = ""
        if len(old_str) > 0:
            try:
                new_str = old_str.replace("，", ",").replace("。", ",") \
                    .replace("、", ",").replace("（", ",").replace("）", ",") \
                    .replace("：", ",").replace("(", ",").replace(")", ",") \
                    .replace(":", ",").replace("/", ",").replace("#", ",") \
                    .replace("[", ",").replace("]", ",").replace("【", ",") \
                    .replace("】", ",").replace("{", ",").replace("}", ",") \
                    .replace("<", ",").replace(">", ",").replace("《", ",") \
                    .replace("》", ",").replace(" ", ",")

            except Exception as e:
                print("str_replace异常: {}\n{}".format(e, old_str))

        return new_str

    # 计算球面两点的距离, 用于计算两家酒店的经纬度距离
    def distance_computer(self, lng, lat, lng2, lat2):

        # 字符串转float
        # 将经纬度转换为弧度(将角度转为弧度)
        lng1 = radians(float(lng))
        lat1 = radians(float(lat))
        lng2 = radians(float(lng2))
        lat2 = radians(float(lat2))

        # 获取两点间弧度的绝对值
        # 两点间经度弧度的绝对值
        dlng = fabs(lng1 - lng2)
        # 两点间纬度弧度的绝对值
        dlat = fabs(lat1 - lat2)

        h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
        distance = 2 * self.radius * asin(sqrt(h))

        return distance

    # 两个字符串的余弦相似度计算
    def cosine_similarity(self, deal_str1, deal_str2):
        # 分词
        # print("deal_str1: {}".format(deal_str1))
        str1_list = jieba.cut(deal_str1)
        word_list1 = [word for word in str1_list if "," not in word]

        # print("deal_str2: {}".format(deal_str2))
        str2_list = jieba.cut(deal_str2)
        word_list2 = [word for word in str2_list if "," not in word]

        # 列出所有的词
        word_dict = []
        for word in word_list1:
            # 如果当前的词没有加入词汇表，则将该词加入词汇表
            if word not in word_dict:
                word_dict.append(word)
            else:
                continue

        for word in word_list2:
            # 如果当前的词没有加入词汇表，则将该词加入词汇表
            if word not in word_dict:
                word_dict.append(word)
            else:
                continue

        # 计算词频，写出词频向量
        word_count1 = {}
        word_count2 = {}
        word_vec1 = []
        word_vec2 = []
        # 对于词汇表中的每一个词，统计它在每句话中出现的次数
        # 关键词统计和词频统计，以列表形式返回
        for word in word_dict:
            num1 = deal_str1.count(word)
            num2 = deal_str2.count(word)
            word_count1[word] = num1
            word_count2[word] = num2
            word_vec1.append(num1)
            word_vec2.append(num2)

        # 计算相似度
        vec1 = np.array(word_vec1)
        vec2 = np.array(word_vec2)
        vector_mul = np.dot(vec1, vec2)
        vec_absolute1 = np.sqrt(np.dot(vec1, vec1))
        vec_absolute2 = np.sqrt(np.dot(vec2, vec2))
        cosine_value = vector_mul / (vec_absolute1 * vec_absolute2)
        # print("cosine_value: {}".format(cosine_value))

        return cosine_value

    # 号码比较, 这里用于电话号码比较，只要有一个号码相同则返回1，否则返回0
    def number_compare(self, num, num2):

        score = 0

        if len(num) > 0 or len(num2) > 0:
            num_list = num.split(",")
            num_list2 = num2.split(",")

            a = 0
            b = 0
            for i in num_list:
                for j in num_list2:
                    b += 1
                    if i == j:
                        score = 1
                        a = 1
                        break
                if a == 1:
                    break

        return score

    # 数据划分方法
    def data_split(self, data_list, split_percent):
        """
        :param data_list: 要划分的list
        :param split_percent: 划分的百分比
        :return:
        """

        # 数据的长度
        data_len = len(data_list)
        # 划分的长度
        n_split = int(split_percent * data_len)

        i = 0
        n_split_index_list = []
        while True:
            random_num = np.random.randint(0, data_len)

            if random_num in n_split_index_list:
                continue

            n_split_index_list.append(random_num)
            i += 1
            if i == n_split:
                break
            pass

        n_split_data_list = []
        leave_data_list = []

        for index_number, value_info in enumerate(data_list):
            if index_number in n_split_index_list:
                n_split_data_list.append(value_info)
            else:
                leave_data_list.append(value_info)
                pass
            pass

        print(n_split_data_list[0])
        print(leave_data_list[0])

        return n_split_data_list, leave_data_list

    # 正负样本采样量，对样本进行有放回采样
    def sampling_data(self, positive_data_list, negative_data_list, positive_sampling, negative_sampling):
        """
        :param positive_data_list: 正样本集
        :param negative_data_list: 负样本集
        :param positive_sampling: 正样本采样量
        :param negative_sampling: 负样本采样量
        :return:
        """

        positive_sampling_len = len(positive_data_list)
        negative_sampling_len = len(negative_data_list)

        sampling_list = []
        for i in range(0, positive_sampling):
            positive_random_num = np.random.randint(0, positive_sampling_len)
            sampling_list.append(positive_data_list[positive_random_num])
            pass

        for i in range(0, negative_sampling):
            negative_random_num = np.random.randint(0, negative_sampling_len)
            sampling_list.append(negative_data_list[negative_random_num])
            pass

        print(sampling_list[0])

        # 将list顺序打乱，防止list前面都是label=1 的数据，后面都是label=0的数据
        np.random.shuffle(sampling_list)

        return sampling_list
        pass

    # 批量保存数据
    def batch_data_save(self, sample_list, save_path):

        # a 以追加的模式打开(在原文件的末尾追加要写入的数据，不覆盖原文件)
        with open(save_path, "a", encoding="utf-8") as file:
            for sample in sample_list:
                file.write(",".join([str(data_info) for data_info in sample]))
                file.write("\n")

                pass
            pass

        pass

    def read_txt2json_data(self, data_path):

        with open(data_path, encoding="utf-8") as file:
            data_list = []
            for line in file.readlines():
                cur_line = line.strip()
                if cur_line == "":
                    continue
                # str to json
                # 不使用 data_json = json.loads(cur_line) 将字符串转 json，
                # 存在隐患，它只能处理内部 key or value 为 "" 的值，如果是 '' 则会报错
                data_json = ast.literal_eval(cur_line)

                # 如果数据为空，则跳过
                if len(data_json) == 0:
                    continue
                data_list.append(data_json)

            return data_list
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.utcnow()
    print("开始时间: ", start_time)

    demo = Utils()

    # 代码结束时间
    end_time = datetime.utcnow()
    print("结束时间: ", end_time, ", 训练模型耗时: ", end_time - start_time)
