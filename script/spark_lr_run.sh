#/bin/bash

######################################################
#                                                    #
# @name:    spark_lr_run.sh                          #
# @author:  zhouyong@staff.sina.com.cn               #
# @date:    20140920                                 #
# @desc:    interaction between program and spark    #
#           environment                              #
#                                                    #
######################################################

################################################
#dir variable
cd $(dirname `ls -l $0 | awk '{print $NF;}'`)/..
WK_DIR=`pwd`

ALERT_DIR=${WK_DIR}alert
DATA_DIR=${WK_DIR}/data             # data下保存中间数据
FLAG_DIR=${WK_DIR}/flag             # crontab_label标志文件
LOG_DIR=${WK_DIR}/log               # log日志
HIVE_DIR=${WK_DIR}/hive
SCRIPT_DIR=${WK_DIR}/script
JAR_DIR=${WK_DIR}/jar
SRC_DIR=${WK_DIR}/src

CONF_DIR=${WK_DIR}/conf
DEFAULT_FILE=${CONF_DIR}/default.conf
MONITOR_FILE=${CONF_DIR}/monitor.conf

################################################
source ${DEFAULT_FILE}

echo $convergenceTol
{
/data0/spark-1.0.1/bin/spark-submit \
  --master yarn-cluster \
  --class com.sina.adtech.mobilealliance.LR_LBFGS \
  --jars /usr/local/hadoop-2.4.0/share/hadoop/common/lib/hadoop-lzo-cdh4-0.4.15-gplextras.jar \
  --driver-memory 2G \
  --executor-memory 1G \
  --executor-cores 2 \
  --num-executors 60 \
  ${JAR_DIR}/spark_sparklr-assembly-0.0.1.jar \
  --training_data ${training_data_path} \
  --testing_data ${testing_data_path} \
  --model_path ${model_path} \
  --num_corrections $numCorrections \
  --num_iterations $maxNumIterations \
  --convergence_tol $convergenceTol \
  --reg_param $regParam \
  --date_arg 20140928 \
  --job_description LRModelonBinaryClassification.1
} > ${LOG_DIR}/spark_lr.log 2>&1 &
