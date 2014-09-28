/**
 * 两件事需要解决：
 *    （1）参数解析
 *    （2）模型评估
 *    （3）spark on yarn测试
 *    都需要参考 刚哥 的代码
 */

package com.sina.adtech.mobilealliance

/**
 * Created by zhouyong on 2014/9/23.
 */

import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.mllib.util.MLUtils
import scala.util.Properties

import scopt.OptionParser
import org.apache.spark.Logging

/**
 * 修改object: LR_LBFGS_test --> LR_LBFGS
 * 修改master: 去除master部分，只保留setAppName
 *
 */

/**
 * static method in java roughly correspond to singleton method in Scala
 * namely:
 *   Java: public class HelloWorld { public static void main(String args[]) {...} }
 *   Scala: object HelloWorld { def main(args: Array[String]) {...} }
 *   not: class HelloWorld {def main(args: Array[String]) {...} }
 */
object LR_LBFGS_1 extends Logging {

  case class Params(
    trainingData: String = null,
    testingData: String = null,
    modelPath: String = null,
    numCorrections: Int = 10,
    numIterations: Int = 20,
    convergenceTol: Double = 1e-4,
    regParam: Double = 0.1,
    jobDescription: String = "Spark LR for CTR Prediction using LBFGS")

  def createSparkContext(): SparkContext = {
    val conf = new SparkConf().setAppName("Spark Logistic Regression")
    val sparkContext = new SparkContext(conf)
    sparkContext
  }

  def trainLRParamByLBFGS(training: RDD[(Double, Vector)], numCorrections: Int, convergenceTol: Double,
    maxNumIterations: Int, regParam: Double, initialWeightsWithIntercept: Vector) = {
    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
      training,
      new LogisticGradient(),
      new SquaredL2Updater(),
      numCorrections, convergenceTol, maxNumIterations, regParam, initialWeightsWithIntercept)
    (weightsWithIntercept, loss)
  }

  def getLogisticRegressionModel(weightsWithIntercept: Vector): LogisticRegressionModel = {
    val model = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))
    model
  }

  def run(params: Params) {
    val sc = createSparkContext()
    //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "D:\\spark-1.1.0-bin-hadoop\\spark_data_zhouyong\\mllib\\sample_libsvm_data.txt")
    //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/user/zeus/sample_binary_classification_data.txt")
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, params.trainingData)
    val numFeatures = data.take(1)(0).features.size

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()

    val test = splits(1)

    val numCorrections = 10
    val convergenceTol = 1e-4
    val maxNumIterations = 20
    val regParam = 0.1

    val initialWeightsWithIntercept: Vector = Vectors.dense(new Array[Double](numFeatures + 1))

    val (weightsWithIntercept, loss) = trainLRParamByLBFGS( // get lr model param
      training,
      numCorrections, convergenceTol, maxNumIterations, regParam,
      initialWeightsWithIntercept)

    val model: LogisticRegressionModel = getLogisticRegressionModel(weightsWithIntercept) // get lr mode
    model.clearThreshold()

    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Loss of each step in training process")

    loss.foreach(println)
    println("Area under ROC = " + auROC)
  }

  def main(args: Array[String]): Unit = {

    val parser = new OptionParser[Params]("LR_LBFGS") {
      head("Logistic Regression: A binary-class, binary-label classifier.")
      opt[String]("training_data")
        .text(s"input paths to training data of binary-labeled example.")
        .required()
        .action((x, c) => c.copy(trainingData = x))
    }

    parser.parse(args, Params()) map {
      case params: Params =>
        logInfo(s"params: $params")
        run(params)
    } getOrElse {
      sys.exit(1)
    }
    // run()
  }
}
