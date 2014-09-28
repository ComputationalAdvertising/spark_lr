#/bin/bash

/data0/spark-1.0.1/bin/spark-submit \
  --master yarn-cluster \
  --class com.sina.adtech.mobilealliance.LR_LBFGS \
  --jars /usr/local/hadoop-2.4.0/share/hadoop/common/lib/hadoop-lzo-cdh4-0.4.15-gplextras.jar \
  --driver-memory 2G \
  --executor-memory 1G \
  --executor-cores 2 \
  --num-executors 60 \
  spark_sparklr-assembly-0.0.1.jar \
  --training_data hdfs://ns1/user/zeus/sample_binary_classification_data.txt \
  --testing_data hdfs://ns1/user/zeus/sample_binary_classification_data.txt \
  --model_path hdfs://ns1/user/zeus/model_for_lr.txt \
  --evaluation_path hdfs://ns1/user/zeus/evaluation_for_lr.txt \
  --num_corrections 10 \
  --num_iterations 20 \
  --convergence_tol 1e-4 \
  --reg_param 0.1 \
  --job_description LRModelonBinaryClassification.1

#> `pwd`/spark_lr.log 2>&1 &
