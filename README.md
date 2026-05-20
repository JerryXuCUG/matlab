#基于改进SVM模型的微裂缝识别与预测工作流
本仓库包含了论文《Micro-Fracture Identification and Prediction with Improved SVM Model: A Case Study of Complex Carbonate and Clastic Rock Reservoirs in the Niuxintuo Area, Liaohe Depression, Bohai Bay Basin》的源代码，发表于《Computers & Geosciences》（2026年）。

该代码库实现了一套基于常规测井曲线、利用支持向量机（SVM）进行裂缝识别的完整工作流。它包含：敏感测井曲线的筛选与标准化、裂缝/非裂缝样本的极值法提取、SMOTE数据增强、多测井曲线交叉图阈值确定、SVM模型的训练与5折交叉验证，以及针对老旧井缺乏部分测井曲线类型（DEN、CNL）时采用替代曲线（SP）的两种预测模型。
