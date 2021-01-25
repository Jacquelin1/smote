package smote

import breeze.linalg.Vector
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


class SMOTE extends Serializable {

  def runSMOTE(spark:SparkSession,
               originalDf:DataFrame,
               oversamplingPctg: Double, //将少数样本筛出来后取多少比例参与过采样计算，为1则用所有少数样本进行过采样
               kNN: Int,  //近邻点个数
               delimiter: String,  //文件分隔符
               numPartitions: Int  //分区个数
              ): DataFrame = {

    val labelCol:String="IS_TARGET"
    var featureCols = ArrayBuffer(originalDf.columns.filter(!_.contains(labelCol)):_*)
    val dataCols: ArrayBuffer[String] = featureCols.+=:(labelCol)
    val correctDF=originalDf.select(dataCols.map(f => col(f).cast("double")): _*)
    val posDF=correctDF.filter(s"$labelCol=1")
    val numPos = posDF.count().toDouble
    val numAll=correctDF.count()
    val numNeg = numAll - numPos
    val pnPerc=(numPos/numNeg).toDouble
    println("numPos",numPos)
    println("numNeg",numNeg)
    println(pnPerc)
    if (pnPerc < 0.3){
      val numSyn = numNeg - numPos
      val creationFactor = math.ceil(numSyn / numPos).toInt

      correctDF.show(5)
      correctDF.printSchema()
      val schema: StructType = correctDF.schema
      val numFeatures=featureCols.size
      val rand = new Random()

      val data: RDD[(LabeledPoint, Int, Int)] = loadData.readDelimitedData(correctDF, numFeatures, delimiter, numPartitions)

      val dataArray: RDD[Array[(LabeledPoint, Int, Int)]] = data.mapPartitions(x => Iterator(x.toArray)).cache()

      //    val value = dataArray
      //    value.foreach(r=>println(r.mkString(",")))
      val numObs = dataArray.map(x => x.size).reduce(_+_)

      println("Number of Filtered Observations "+numObs.toString)

      val roundPctg = oversamplingPctg
      val fm: RDD[(LabeledPoint, Int, Int)] = dataArray.flatMap(x => x)
      val sampleData: Array[(LabeledPoint, Int, Int)] = dataArray.flatMap(x => x)
        .sample(withReplacement = false, fraction = roundPctg, seed = 1L).collect().sortBy(r => (r._2, r._3)) //without Replacement
      sampleData.foreach(t=>println(t._1,t._2,t._3))
      println("Sample Data Count "+sampleData.size.toString)

      val globalNearestNeighbors: Array[(String, Array[((Int, Int), Double)])] = NearestNeighbors.runNearestNeighbors(dataArray, kNN, sampleData)
      globalNearestNeighbors.foreach(m=>println(m._1,m._2.mkString(",")))

      val randomNearestNeighbor: Array[(Int, Int, ((Int, Int), Double))] = globalNearestNeighbors.map(x => (x._1.split(",")(0).toInt,
        x._1.split(",")(1).toInt, x._2(rand.nextInt(kNN)))).sortBy(r => (r._1, r._2))
      println(rand.nextInt(kNN))//定一个参数n，nextInt(n)将返回一个大于等于0小于n的随机数，即：0 <= nextInt(n) < n
      randomNearestNeighbor.foreach(r=>println(r._1,r._2,r._3._1._1,r._3._1._2,r._3._2))

      val t1: Array[((Int, Int, ((Int, Int), Double)), (LabeledPoint, Int, Int))] = randomNearestNeighbor.zip(sampleData)
      for (t<-t1){
        println(t._1._1,t._1._2,t._1._3,t._2)
      }
      val sampleDataNearestNeighbors: Array[(Int, Int, Int, LabeledPoint)] = randomNearestNeighbor.zip(sampleData)
        .map(x => (x._1._3._1._1, x._1._2, x._1._3._1._2, x._2._1))
      sampleDataNearestNeighbors.foreach(s=>println(s._1,s._2,s._3,s._4))

      val syntheticData= dataArray.mapPartitionsWithIndex(createSyntheticData(_, _, sampleDataNearestNeighbors, delimiter, creationFactor,spark, schema)).persist()
      syntheticData.foreach(r=>println("syntheticData",r.split(",").mkString(",")))
      val value: RDD[Row] = syntheticData.map(_.split(",")).map(p=>Row.fromSeq(p.map(_.toDouble)))
      val syntheticDF: DataFrame = spark.createDataFrame(value ,schema)
      syntheticDF.show()
      println("Synthetic Data Count "+syntheticData.count.toString)
      val resDF = syntheticDF.union(correctDF)
      resDF.show(100)

      resDF
    }else{
      correctDF
    }
  }

  private def createSyntheticData(partitionIndex: Long,
                                  iter: Iterator[Array[(LabeledPoint,Int,Int)]],
                                  sampleDataNN: Array[(Int,Int,Int,LabeledPoint)],
                                  delimiter: String,
                                  creationFactor:Int,
                                  spark:SparkSession,
                                  schema:StructType): Iterator[String]  = {

    var result = List[String]()
    val dataArr = iter.next
    val nLocal = dataArr.size - 1
    val sampleDataNNSize = sampleDataNN.size - 1
    for (i <- 0 to creationFactor) {
      val rand = new Random().nextDouble()
      println("rand***",rand)
      for (j <- 0 to sampleDataNNSize) {
        val partitionId = sampleDataNN(j)._1
        val neighborId = sampleDataNN(j)._3
        val sampleFeatures: Vector[Double] = sampleDataNN(j)._4.features
        if (partitionId == partitionIndex.toInt) {
          val currentPoint = dataArr(neighborId)
          val features = currentPoint._1.features
          //					println("sampleFeatures,features",sampleFeatures.toArray.mkString(",")," ** ",features.toArray.mkString(","),rand.nextDouble)
          sampleFeatures += (features - sampleFeatures) * rand

          println(sampleFeatures.toArray.mkString(","))
          result.::=("1.0" + delimiter + sampleFeatures.toArray.mkString(delimiter))
        }
      }
    }
    result.iterator
  }
}


object RunSmote {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("Summary Statistics")
      .config("spark.sql.warehouse.dir", ".")
      .getOrCreate()
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //    val sc = spark.sparkContext
    var rawSamples = spark.read.format("csv").option("sep", ",").option("inferSchema", "true")
      .option("header", "true").load("/opt/temp/iris4minority.csv")
    //      .option("header", "true").load("/opt/temp/train_data.csv")
    rawSamples = rawSamples.drop("CUST_NO")
    val t1 = System.currentTimeMillis()
    val s=new SMOTE
    val resDF=s.runSMOTE(spark,rawSamples,1.0,3,",",3)

    val t2 = System.currentTimeMillis()
    resDF.coalesce(1).write.option("header", "true").mode("overwrite").format("csv").save("/opt/temp/overSample.csv")
    println("smote TIME COST：" + (t2 - t1) / 1000)
  }
}