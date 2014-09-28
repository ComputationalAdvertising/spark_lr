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
object LR_LBFGS_2 extends Logging {

  case class Params(
    trainingData: String = null,
    testingData: String = null,
    modelPath: String = null,
    evaluationPath: String = null,
    numCorrections: Int = 10,
    numIterations: Int = 20,
    convergenceTol: Double = 1e-4,
    regParam: Double = 0.1,
    jobDescription: String = "Spark CTR Prediction using LR_LBFGS")

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
    val training_data = MLUtils.loadLibSVMFile(sc, params.trainingData)
    val numFeatures = training_data.take(1)(0).features.size

    val splits = training_data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()

    //val testing_data = MLUtils.loadLibSVMFile(sc, params.testingData)
    //val testing = testing_data.map(x => (x.label, MLUtils.appendBias(x.features))).cache()
    val testing = splits(1)

    val numCorrections = 10
    val convergenceTol = 1e-4
    val maxNumIterations = 20
    val regParam = 0.1

    val initialWeightsWithIntercept: Vector = Vectors.dense(new Array[Double](numFeatures + 1))

    val (weightsWithIntercept, loss) = trainLRParamByLBFGS( // get lr model param
      training,
      params.numCorrections, params.convergenceTol, params.numIterations, params.regParam,
      initialWeightsWithIntercept)

    val model: LogisticRegressionModel = getLogisticRegressionModel(weightsWithIntercept) // get lr mode
    model.clearThreshold()

    val scoreAndLabels = testing.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    val pr_value = metrics.areaUnderPR()

    println("Loss of each step in training process")

    loss.foreach(println)
    println("Testing AUC is: " + auROC)
    println("Testing PR is: " + pr_value)

  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[Params]("LR_LBFGS") {
      head("Logistic Regression: A binary-class, binary-label classifier.")
      opt[String]("training_data")
        .text(s"input paths to training data of binary-labeled example.")
        .required()
        .action((x, c) => c.copy(trainingData = x))
      opt[String]("testing_data")
        .text(s"input paths to testing data of binary-labeled example.")
        .required()
        .action((x, c) => c.copy(testingData = x))
      opt[String]("model_path")
        .text(s"model path to save logistic regression model(include params using lbfgs).")
        .required()
        .action((x, c) => c.copy(modelPath = x))
      opt[String]("evaluation_path")
        .text(s"input paths to training data of binary-labeled example.")
        .required()
        .action((x, c) => c.copy(evaluationPath = x))
      opt[Int]("num_corrections")
        .text(s"num of corrections for the lbfgs. default = 10")
        .required()
        .action((x, c) => c.copy(numCorrections = x))
      opt[Int]("num_iterations")
        .text(s"num of interations for the lbgfs. default = 20.")
        .required()
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("convergence_tol")
        .text(s"covegence threshold for the lbfgs. default = 1e-4.")
        .required()
        .action((x, c) => c.copy(convergenceTol = x))
      opt[Double]("reg_param")
        .text(s"weights of regulation items for the loss function. default = 0.1.")
        .required()
        .action((x, c) => c.copy(regParam = x))
      opt[String]("job_description")
        .text(s"the description info in the Name field of spark/hadoop cluster ui.")
        .required()
        .action((x, c) => c.copy(jobDescription = x))
    }

    parser.parse(args, Params()) map {
      case params: Params =>
        logInfo(s"params: $params")
        run(params)
    } getOrElse {
      sys.exit(1)
    }
  }
}
