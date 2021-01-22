package smote

import org.apache.spark.SparkContext
import breeze.linalg._
import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object loadData {

  //  def readDelimitedData(sc: SparkContext, path: String, numFeatures: Int, delimiter: String, numPartitions: Int): RDD[(LabeledPoint, Int, Int)] = {
  def readDelimitedData(originalDf:DataFrame, numFeatures: Int, delimiter: String, numPartitions: Int): RDD[(LabeledPoint, Int, Int)] = {
    val target="IS_TARGET"
    val rdd: RDD[Row] = originalDf.filter(s"$target=1").rdd
    val data: RDD[Array[String]] = originalDf.filter(s"$target=1").rdd.map(r => r.toString().replace("[","").replace("]",""))
      .repartition(numPartitions).mapPartitions { x => Iterator(x.toArray) }
    val formatData = data.mapPartitionsWithIndex { (partitionId, iter) =>
      var result = List[(LabeledPoint, Int, Int)]()
      val dataArray = iter.next
      val dataArraySize = dataArray.size - 1
      var rowCount = dataArraySize
      for (i <- 0 to dataArraySize) {
        val parts = dataArray(i).split(delimiter)
        result.::=((LabeledPoint(parts(0).toDouble, DenseVector(parts.slice(1, numFeatures+1)).map(_.toDouble)), partitionId.toInt, rowCount))
        rowCount = rowCount - 1
      }
      result.iterator
    }

    formatData
  }

}

object runLoadData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("Summary Statistics")
      .config("spark.sql.warehouse.dir", ".")
      .getOrCreate()
    val sc = spark.sparkContext
    println(spark.version)
    val rawSamples = spark.read.format("csv").option("sep", ",").option("inferSchema", "true")
      .option("header", "true").load("/opt/temp/iris4minority.csv")

    //    val formatData: RDD[(LabeledPoint, Int, Int)] = loadData.readDelimitedData(sc,"/opt/temp/small_test.txt",54," ",2)
    val formatData: RDD[(LabeledPoint, Int, Int)] = loadData.readDelimitedData(rawSamples,4,",",2)
    //    formatData.foreach(d=>println(d.mkString(",")))
    formatData.foreach(d=>println(d._1,d._2,d._3))

  }
}
