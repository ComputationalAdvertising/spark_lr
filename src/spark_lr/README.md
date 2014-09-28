spark_logistic regression using lbfgs
================

这里使用LBFGS算法训练LR模型，主要包括3个部分：
1. lbfgs和lr的封装；
2. 程序与spark集群环境的交互（参数解析）；
3. LR模型的保存与统计/评估指标
