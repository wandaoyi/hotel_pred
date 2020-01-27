#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/18 22:48
# @Author   : WanDaoYi
# @FileName : hotel_train.py
# ============================================

from datetime import datetime
import os
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import joblib
from config import cfg

import matplotlib.pyplot as plt
import matplotlib

# 用于解决画图中文乱码
font = {"family": "SimHei"}
matplotlib.rc("font", **font)


class HotelTrain(object):

    def __init__(self):
        self.train_data_path = cfg.TRAIN.TRAIN_DATA_INFO_PATH
        self.val_data_path = cfg.TRAIN.VAL_DATA_INFO_PATH

        self.model_save_path = cfg.TRAIN.MODEL_SAVE_PATH

        self.x_train, self.y_train = self.read_data(self.train_data_path)
        self.x_val, self.y_val = self.read_data(self.val_data_path)

        self.roc_flag = cfg.TRAIN.ROC_FLAG
        pass

    # 读取数据
    def read_data(self, data_path):
        with open(data_path, "r") as file:
            data_info = file.readlines()
            data_list = [data.strip().split(",") for data in data_info]

            data_list = np.array(data_list)
            data_info = data_list[:, 4:-1]
            label_info = data_list[:, -1:]

            data_info = np.array(data_info, dtype=float)
            label_info = np.array(label_info, dtype=float)
            # print(data_info[: 5])

            # ravel() 是列转行，用于解决数据转换警告。
            return data_info, label_info.ravel()
        pass

    def best_estimators_depth(self):
        # np.arange 可以生成 float 类型，range 只能生成 int 类型
        best_param = {'n_estimators': range(10, 201, 5),
                      'max_depth': range(1, 20, 1)
                      }
        best_gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                            gamma=0,
                                                            subsample=0.8,
                                                            colsample_bytree=0.8,
                                                            objective='binary:logistic',
                                                            nthread=4,
                                                            min_child_weight=5,
                                                            seed=27
                                                            ),
                                    param_grid=best_param, scoring='roc_auc', iid=False, cv=10)

        best_gsearch.fit(self.x_train, self.y_train)
        print("best_param:{0}".format(best_gsearch.best_params_))
        print("best_score:{0}".format(best_gsearch.best_score_))
        # best_param: {'max_depth': 1, 'n_estimators': 10}
        # best_score: 1.0
        return best_gsearch.best_params_
        pass

    def best_lr_gamma(self, estimator=10, depth=1):
        # np.arange 可以生成 float 类型，range 只能生成 int 类型
        best_param = {'learning_rate': np.arange(0.1, 1.1, 0.1),
                      'gamma': np.arange(0.1, 5.1, 0.2)
                      }
        best_gsearch = GridSearchCV(estimator=XGBClassifier(n_estimators=estimator,
                                                            max_depth=depth,
                                                            # learning_rate=0.1,
                                                            # gamma=0,
                                                            subsample=0.8,
                                                            colsample_bytree=0.8,
                                                            objective='binary:logistic',
                                                            nthread=4,
                                                            min_child_weight=5,
                                                            seed=27
                                                            ),
                                    param_grid=best_param, scoring='roc_auc', iid=False, cv=10)

        best_gsearch.fit(self.x_train, self.y_train)
        print("best_param:{0}".format(best_gsearch.best_params_))
        print("best_score:{0}".format(best_gsearch.best_score_))
        # best_param: {'gamma': 0.1, 'learning_rate': 0.1}
        # best_score: 1.0
        return best_gsearch.best_params_
        pass

    def best_subsmaple_bytree(self, estimator=10, depth=1, lr=0.1, gama=0.1):
        # np.arange 可以生成 float 类型，range 只能生成 int 类型
        # 调整subsample(行),colsample_bytree（列）
        best_param = {'subsample': np.arange(0.1, 1.1, 0.1),
                      'colsample_bytree': np.arange(0.1, 1.1, 0.1)
                      }
        best_gsearch = GridSearchCV(estimator=XGBClassifier(n_estimators=estimator,
                                                            max_depth=depth,
                                                            learning_rate=lr,
                                                            gamma=gama,
                                                            # subsample=1.0,
                                                            # colsample_bytree=0.8,
                                                            objective='binary:logistic',
                                                            nthread=4,
                                                            min_child_weight=5,
                                                            seed=27
                                                            ),
                                    param_grid=best_param, scoring='roc_auc', iid=False, cv=10)

        best_gsearch.fit(self.x_train, self.y_train)
        print("best_param:{0}".format(best_gsearch.best_params_))
        print("best_score:{0}".format(best_gsearch.best_score_))
        # best_param: {'colsample_bytree': 0.30000000000000004, 'subsample': 1.0}
        # best_score: 1.0
        return best_gsearch.best_params_
        pass

    def best_nthread_weight(self, estimator=10, depth=1, lr=0.1,
                            gama=0.1, subsamples=1.0, bytree=0.3):
        # np.arange 可以生成 float 类型，range 只能生成 int 类型
        best_param = {'nthread': range(1, 20, 1),
                      'min_child_weight': range(1, 20, 1)
                      }
        best_gsearch = GridSearchCV(estimator=XGBClassifier(n_estimators=estimator,
                                                            max_depth=depth,
                                                            learning_rate=lr,
                                                            gamma=gama,
                                                            subsample=subsamples,
                                                            colsample_bytree=bytree,
                                                            objective='binary:logistic',
                                                            # nthread=4,
                                                            # min_child_weight=5,
                                                            seed=27
                                                            ),
                                    param_grid=best_param, scoring='roc_auc', iid=False, cv=10)

        best_gsearch.fit(self.x_train, self.y_train)
        print("best_param:{0}".format(best_gsearch.best_params_))
        print("best_score:{0}".format(best_gsearch.best_score_))
        # best_param: {'min_child_weight': 1, 'nthread': 1}
        # best_score: 1.0
        return best_gsearch.best_params_
        pass

    def best_seek(self, estimator=10, depth=1, lr=0.1,
                  gama=0.1, subsamples=1.0, bytree=0.3,
                  n_thread=1, child_weight=1):
        # np.arange 可以生成 float 类型，range 只能生成 int 类型
        best_param = {'seed': range(1, 1000, 1)}
        best_gsearch = GridSearchCV(estimator=XGBClassifier(n_estimators=estimator,
                                                            max_depth=depth,
                                                            learning_rate=lr,
                                                            gamma=gama,
                                                            subsample=subsamples,
                                                            colsample_bytree=bytree,
                                                            nthread=n_thread,
                                                            min_child_weight=child_weight,
                                                            # seed=27,
                                                            objective='binary:logistic'
                                                            ),
                                    param_grid=best_param, scoring='roc_auc', iid=False, cv=10)

        best_gsearch.fit(self.x_train, self.y_train)
        print("best_param:{0}".format(best_gsearch.best_params_))
        print("best_score:{0}".format(best_gsearch.best_score_))
        # best_param: {'seed': 7}
        # best_score: 1.0
        return best_gsearch.best_params_
        pass

    # 绘制 ROC 曲线
    def plt_roc(self, model):
        if self.roc_flag:
            y_proba = model.predict_proba(self.x_val)
            # 预测为 0 的概率
            y_zero = y_proba[:, 0]
            # 预测为 1 的概率
            y_one = y_proba[:, 1]
            print("AUC Score2: {}".format(metrics.roc_auc_score(self.y_val, y_one)))
            # 得到误判率、命中率、门限
            fpr, tpr, thresholds = metrics.roc_curve(self.y_val, y_one)
            # 计算auc
            roc_auc = metrics.auc(fpr, tpr)
            # 对ROC曲线图正常显示做的参数设定
            # 用来正常显示中文标签, 上面设置过
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # 用来正常显示负号
            plt.rcParams['axes.unicode_minus'] = False

            plt.plot(fpr, tpr, label='{0}_AUC = {1:.5f}'.format("xgboost", roc_auc))

            plt.title('ROC曲线')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.ylabel('命中率: TPR')
            plt.xlabel('误判率: FPR')
            plt.show()
        pass

    # 较好的 模型参数 进行训练
    def best_param_xgboost(self, estimator=10, depth=1, lr=0.1,
                           gama=0.1, subsamples=1.0, bytree=0.3,
                           n_thread=1, child_weight=1, seed_num=7):
        best_model = XGBClassifier(n_estimators=estimator,
                                   max_depth=depth,
                                   learning_rate=lr,
                                   gamma=gama,
                                   subsample=subsamples,
                                   colsample_bytree=bytree,
                                   nthread=n_thread,
                                   min_child_weight=child_weight,
                                   seed=seed_num,
                                   objective='binary:logistic'
                                   )

        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_val)
        acc_score = metrics.accuracy_score(self.y_val, y_pred)
        print("acc_score: {}".format(acc_score))
        print("score: {}".format(best_model.score(self.x_val, self.y_val)))

        save_path = self.model_save_path + "acc={:.6f}".format(acc_score) + ".m"
        # 判断模型是否存在，存在则删除
        if os.path.exists(save_path):
            os.remove(save_path)
            pass

        # 保存模型
        joblib.dump(best_model, save_path)

        print("AUC Score: {}".format(metrics.roc_auc_score(self.y_val, y_pred)))

        # 绘制 ROC 曲线
        self.plt_roc(best_model)

        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.utcnow()
    print("开始时间: ", start_time)

    demo = HotelTrain()

    estimators_depth = demo.best_estimators_depth()
    estimator = estimators_depth["n_estimators"]
    depth = estimators_depth["max_depth"]

    lr_gamma = demo.best_lr_gamma(estimator=estimator, depth=depth)
    lr = lr_gamma["learning_rate"]
    gamma = lr_gamma["gamma"]

    subsmaple_bytree = demo.best_subsmaple_bytree(estimator=estimator,
                                                  depth=depth,
                                                  lr=lr,
                                                  gama=gamma
                                                  )
    subsmaple = subsmaple_bytree["subsample"]
    bytree = subsmaple_bytree["colsample_bytree"]

    nthread_weight = demo.best_nthread_weight(estimator=estimator,
                                              depth=depth,
                                              lr=lr,
                                              gama=gamma,
                                              subsamples=subsmaple,
                                              bytree=bytree
                                              )
    nthread = nthread_weight["nthread"]
    weight = nthread_weight["min_child_weight"]

    seek_num = demo.best_seek(estimator=estimator,
                              depth=depth,
                              lr=lr,
                              gama=gamma,
                              subsamples=subsmaple,
                              bytree=bytree,
                              n_thread=nthread,
                              child_weight=weight
                              )
    seek = seek_num["seed"]

    demo.best_param_xgboost(estimator=estimator,
                            depth=depth,
                            lr=lr,
                            gama=gamma,
                            subsamples=subsmaple,
                            bytree=bytree,
                            n_thread=nthread,
                            child_weight=weight,
                            seed_num=seek
                            )

    # 代码结束时间
    end_time = datetime.utcnow()
    print("结束时间: ", end_time, ", 训练模型耗时: ", end_time - start_time)
