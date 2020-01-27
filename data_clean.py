#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/18 22:47
# @Author   : WanDaoYi
# @FileName : data_clean.py
# ============================================

from datetime import datetime
import os
import json
from utils import Utils
from config import cfg


class DataClean(object):

    def __init__(self):

        self.ori_train_data_path = cfg.COMMON.ORIGINAL_TRAIN_DATA_PATH
        self.ori_test_data_path = cfg.COMMON.ORIGINAL_TEST_DATA_PATH

        self.clean_data_path = cfg.COMMON.CLEAN_DATA_PATH
        self.hotel_source_list = cfg.COMMON.HOTEL_SOURCE_LIST

        self.utils = Utils()

        pass

    # 数据清洗
    def clean_data(self, data_source, data_json, province, city):

        hotel_id = ""
        hotel_name = ""
        address = ""
        telephone = ""
        fax = ""
        lng = ""
        lat = ""
        country_name = ""
        province_name = ""
        city_name = ""
        district_name = ""
        postal_code = ""
        open_year = ""

        if "HotelID" in data_json:
            hotel_id = data_json["HotelID"]

        if "HotelName" in data_json:
            hotel_name = self.utils.str_standard(data_json["HotelName"])

        if "GeoInfo" in data_json:
            geo_info = data_json["GeoInfo"]

            if "Address" in geo_info:
                address = self.utils.str_standard(geo_info["Address"])

            if "PostalCode" in geo_info:
                postal_code = self.utils.str_standard(geo_info["PostalCode"])

            if "Country" in geo_info:
                country = geo_info["Country"]
                if "Name" in country:
                    country_name = self.utils.str_standard(country["Name"])

            if "City" in geo_info:
                city_info = geo_info["City"]
                if "Name" in city_info:
                    city_name = self.utils.str_standard(city_info["Name"])

            if "Province" in geo_info:
                province_info = geo_info["Province"]
                if len(province_info):
                    province_name = self.utils.str_standard(province_info["Name"])

            if "BusinessDistrict" in geo_info:
                business_district = geo_info["BusinessDistrict"]
                if len(business_district):
                    district = business_district[0]
                    district_name = self.utils.str_standard(district["Name"])

        if "LNG" in data_json:
            lng = data_json["LNG"]
            if "-1" == lng or "-1.0" == lng or "0" == lng:
                lng = ""

        if "LAT" in data_json:
            lat = data_json["LAT"]
            if "-1" == lat or "-1.0" == lat or "0" == lat:
                lat = ""

        if "ContactInfo" in data_json:

            contact_info = data_json["ContactInfo"]
            if "Telephone" in contact_info:
                tel = contact_info["Telephone"]
                tel_standard = self.utils.str_standard(tel)
                telephone = self.utils.str_un_chinese(tel_standard)

            if "Fax" in contact_info:
                fa = contact_info["Fax"]
                fax_standard = self.utils.str_standard(fa)
                fax = self.utils.str_un_chinese(fax_standard)

        if "OpenYear" in data_json:
            open_year = data_json["OpenYear"].replace("-", ",")

        if len(province_name) == 0:
            province_name = province
        if len(city_name) == 0:
            city_name = city

        data_info = {"hotel_source": data_source,
                     "hotel_id": hotel_id, "hotel_name": hotel_name,
                     "address": address, "telephone": telephone,
                     "fax": fax, "lng": lng, "lat": lat,
                     "country_name": country_name, "province_name": province_name,
                     "city_name": city_name, "district_name": district_name,
                     "postal_code": postal_code, "open_year": open_year,
                     }

        return data_info
        pass

    # 读取数据
    def read_data(self, data_dic):

        clean_data_list = []
        data_dir = {}
        for data_source in data_dic:
            data_info = data_dic[data_source]
            data_path = data_info["data_path"]
            province = data_info["province_name"]
            city = data_info["city_name"]

            data_list = []
            with open(data_path, encoding="utf-8") as file:

                for line in file.readlines():
                    cur_line = line.strip()
                    if cur_line == "":
                        continue
                    # str to json
                    data_json = json.loads(cur_line)
                    # 如果数据为空，则跳过
                    if len(data_json) == 0:
                        continue

                    data_info_list = self.clean_data(data_source, data_json, province, city)
                    data_list.append(data_info_list)

                pass

            data_dir.update({data_source: data_list})

        clean_data_list.append(data_dir)
        # print(clean_data_list[0]["xie_cheng"][0])

        return clean_data_list
        pass

    # 将清洗的数据写下来
    def dumn_data(self, data_list, train_flag=True):

        for list_info in data_list:
            data_info = list_info["data_info"]

            if train_flag:
                file_dumn_path = os.path.join(self.clean_data_path, "train/")
                pass
            else:
                province_pinyin = list_info["province_pinyin"]
                city_pinyin = list_info["city_pinyin"]
                file_dumn_path = os.path.join(self.clean_data_path, "test/" + province_pinyin + "/" + city_pinyin + "/")
                pass

            # 判断路径是否存在，如果不存在则创建
            if not os.path.exists(file_dumn_path):
                os.makedirs(file_dumn_path)

            for data in data_info:

                for source_key in data:

                    file_write = open(os.path.join(file_dumn_path, source_key + ".txt"),
                                      "w", encoding="utf-8")
                    hotel_info = data[source_key]
                    print(hotel_info[0])
                    for hotel in hotel_info:
                        file_write.write(str(hotel))
                        file_write.write('\n')
                    file_write.close()
                pass

            pass
        pass

    # 处理测试数据
    def deal_test_data(self):
        file_list = os.listdir(self.ori_test_data_path)

        data_list = []
        # 获取省份名称信息
        for province_name in file_list:
            data_province_path = os.path.join(self.ori_test_data_path, province_name)
            province_path = os.listdir(data_province_path)

            # 将中文转为拼音，用来创建路径用
            province_pinyin = self.utils.chinese_2_pinyin(province_name)

            # 获取城市名称信息
            for city_name in province_path:
                data_city_path = os.path.join(data_province_path, city_name)
                city_path = os.listdir(data_city_path)

                # 将中文转为拼音，用来创建路径用
                city_pinyin = self.utils.chinese_2_pinyin(city_name)

                data_dic = {}

                # 获取文件路径
                for file_name in city_path:
                    data_path = os.path.join(data_city_path, file_name)
                    data_info_dic = {"data_path": data_path,
                                     "province_name": province_name,
                                     "city_name": city_name
                                     }
                    if "携程" in file_name:
                        data_dic.update({"xie_cheng": data_info_dic})
                        pass
                    elif "龙腾" in file_name:
                        data_dic.update({"long_teng": data_info_dic})
                        pass
                    else:
                        data_dic.update({"tai_tan": data_info_dic})
                    pass
                pass

                clean_data_list = self.read_data(data_dic)

                clean_data_dir = {"data_info": clean_data_list,
                                  "province_pinyin": province_pinyin,
                                  "city_pinyin": city_pinyin
                                  }

                data_list.append(clean_data_dir)

        return data_list
        pass

    # 处理训练数据
    def deal_train_data(self):
        file_list = os.listdir(self.ori_train_data_path)

        data_list = []
        data_dic = {}
        for file_name in file_list:
            data_path = os.path.join(self.ori_train_data_path, file_name)
            data_info_dic = {"data_path": data_path,
                             "province_name": "",
                             "city_name": ""
                             }
            if "携程" in file_name:
                data_dic.update({"xie_cheng": data_info_dic})
                pass
            elif "龙腾" in file_name:
                data_dic.update({"long_teng": data_info_dic})
                pass
            else:
                data_dic.update({"tai_tan": data_info_dic})
            pass

        clean_data_list = self.read_data(data_dic)

        clean_data_dir = {"data_info": clean_data_list}
        data_list.append(clean_data_dir)
        return data_list
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.utcnow()
    print("开始时间: ", start_time)

    demo = DataClean()
    train_data_list = demo.deal_train_data()
    demo.dumn_data(train_data_list)
    test_data_list = demo.deal_test_data()
    demo.dumn_data(test_data_list, train_flag=False)
    # 代码结束时间
    end_time = datetime.utcnow()
    print("结束时间: ", end_time, ", 训练模型耗时: ", end_time - start_time)
    pass
