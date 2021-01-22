package smote

import org.apache.spark.SparkContext
import breeze.linalg._
import breeze.linalg.{DenseVector,Vector,SparseVector}
import com.github.fommil.netlib.BLAS
import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable.ArrayBuffer

object NearestNeighbors {

	def runNearestNeighbors(data: RDD[Array[(LabeledPoint,Int,Int)]],
                          kNN: Int,
                          sampleData: Array[(LabeledPoint,Int,Int)]): Array[(String,Array[((Int,Int),Double)])] = {

//		val r1: RDD[(String, Iterable[((Int, Int), Double)])] = data.mapPartitionsWithIndex(localNearestNeighbors(_, _, kNN, sampleData)).groupByKey()
//		r1.foreach(rr=>println("localNearestNeighbors groupByKey",rr._1,rr._2.mkString(",")))
		val globalNearestNeighborsByIndex = data.mapPartitionsWithIndex(localNearestNeighbors(_,_,kNN,sampleData))
      .groupByKey().map(x => (x._1,x._2.toArray.sortBy(r => r._2).take(kNN))).collect()

		globalNearestNeighborsByIndex
	}


	private def localNearestNeighbors(partitionIndex: Long,
                                    iter: Iterator[Array[(LabeledPoint,Int,Int)]],
                                    kNN: Int,
                                    sampleData: Array[(LabeledPoint,Int,Int)]): Iterator[(String,((Int,Int),Double))] = {

			var result = List[(String,((Int,Int),Double))]()
			val dataArr: Array[(LabeledPoint, Int, Int)] = iter.next
			val nLocal = dataArr.size - 1
			val sampleDataSize = sampleData.size - 1


		val kLocalNeighbors: Array[distanceIndex] = Array.fill[distanceIndex](sampleDataSize + 1)(null)
		for (i1 <- 0 to sampleDataSize){
//			distanceIndex(val sampleRowId: Int, val partitionId: Int,
			// 							val distanceVector: DenseVector[Double], val neighborRowId: DenseVector[Int])
			kLocalNeighbors(i1) = distanceIndex(sampleData(i1)._3.toInt, sampleData(i1)._2.toInt,
				DenseVector.zeros[Double](kNN) + Int.MaxValue.toDouble, DenseVector.zeros[Int](kNN))
		}

		for (i <- 0 to nLocal) {  //every Array((LabeledPoint(1.0,DenseVector(5.0, 3.4, 1.6, 0.4)),1,11))
			val currentPoint = dataArr(i)
//			println(currentPoint._1,currentPoint._2,currentPoint._3)    //(LabeledPoint(1.0,DenseVector(5.0, 3.3, 1.4, 0.2)),0,0)
			val features = currentPoint._1.features
			val rowId = currentPoint._3.toInt
			for (j <- 0 to sampleDataSize) {
				val samplePartitionId = sampleData(j)._2
				val sampleRowId = sampleData(j)._3
				val sampleFeatures = sampleData(j)._1.features
				if (!((rowId == sampleRowId) & (samplePartitionId == partitionIndex))) {    //按位与:两边都是1才是1（https://www.runoob.com/scala/scala-operators.html）。本例中两边都是1就是同一条样本了，是不需要计算距离的
//					println(rowId,sampleRowId,samplePartitionId,partitionIndex)

					val d: Double = Math.sqrt(sum((sampleFeatures - features) *:* (sampleFeatures - features)))
					val distance = d
//					println("cal distance",samplePartitionId,sampleRowId,d,sampleFeatures.toArray.mkString(",")," ** ",partitionIndex,rowId,"features",features.toArray.mkString(","))
					if (distance < max(kLocalNeighbors(j).distanceVector)) {
						val indexToReplace = argmax(kLocalNeighbors(j).distanceVector)
						kLocalNeighbors(j).distanceVector(indexToReplace) = distance
						kLocalNeighbors(j).neighborRowId(indexToReplace) = rowId
					}
				}
			}
		}

		kLocalNeighbors.foreach(k=>println(k.partitionId,k.sampleRowId,k.neighborRowId,k.distanceVector))

		for (m <- 0 to sampleDataSize){
			for (l <-0 to kNN-1) {

				val key = kLocalNeighbors(m).partitionId.toString+","+kLocalNeighbors(m).sampleRowId.toString
				val tup = (partitionIndex.toInt,kLocalNeighbors(m).neighborRowId(l))
				result.::=(key,(tup,kLocalNeighbors(m).distanceVector(l)))
			}
		}
//		result.foreach(r=>println("key: "+r._1,"dataArr partitionIndex: "+r._2._1._1,"neighborRowId: "+r._2._1._2,"distanceVector: "+r._2._2))
		result.iterator
	}
}
