# hotel_pred
XGBoost 酒店预测 2020-1-18
- 项目自带少量数据，如果想要更多数据，则自己想办法去搞了。
- 数据下载：链接：https://pan.baidu.com/s/17UAInz7C14uSTambA6Eatw 
            提取码：kgmt 
- 运行 data_clean.py 对数据进行清洗
- 运行 prepare.py 将数据集划为训练集，验证集和测试集。准备数据的时候，直接运行 prepare.py 会调用 data_clean.py 中的方法。无需单独运行 data_clean.py 文件

## 训练模型
- 开始训练模型之前，先进行调参
- 运行 hotel_train.py 中的下列方法
- best_estimators_depth() # 调节模型的迭代次数和深度
- best_lr_gamma()	# 调节模型的学习率和gamma值
- best_subsmaple_bytree()	# 调节子样本集和叶子数(即调节行和列)
- best_nthread_weight()	# 调节最小子权重和线程数
- best_seek()	# 调节随机种子
- best_param_xgboost()	# 方法为上面网格搜索后得到较好的参数后，对数据进行训练

- 将调参寻找到的较优的参数，配置到 demo.best_param_xgboost() 方法中，开始训练
- 训练模型的时候，会绘制 ROC 曲线，方面目测效果的好坏。
- 如果不想展示，可以在 config.py 中将 __C.TRAIN.ROC_FLAG 设置为 False
- 多训练几次，寻找一个较好的模型。

## 预测
- 加载权重，将训练好的权重 .m 文件放入models文件夹
- 运行 hotel_test.py，对数据集进行预测，将预测结果整理后，输出到 outputs 文件夹中
- 预测结果，请参详 outputs 文件夹中的文件：
- long_teng.csv	# 龙腾没匹配上的剩余酒店信息
- tai_tan.csv	# 泰坦没匹配上的剩余酒店信息
- xie_cheng.csv	# 携程全部信息，和匹配上龙腾、泰坦酒店的信息


