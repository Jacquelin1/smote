## SMOTE（Scala）计算过程

## 一、转换源数据
将label列换到第一列，并分区

## 二、寻找k近邻
原理可参考sklearn knn(from sklearn.neighbors import KNeighborsClassifier)，改写也是根据Python源码走  
1.计算距离  
功能函数：NearestNeighbors.runNearestNeighbors  
以当前sampleData为基准数据计算，dataArr为对照数据，尽管这两样本内容是一样的，采用欧氏距离Math.sqrt(sum((sampleFeatures - features) *:* (sampleFeatures - features)))  
kLocalNeighbors即为某个点与除自己之外的点的距离数据集，初步计算后的结果解释：  
kLocalNeighbors.foreach(k=>println(k.partitionId,k.sampleRowId,k.neighborRowId,k.distanceVector))  
kLocalNeighbors某一行内容：(0,0,DenseVector(1, 2, 3, 0),DenseVector(0.5099019513592786, 0.4123105625617663, 0.5477225575051662, 2.147483647E9))  
四个参数解释：  
sampleData partitionId当前需要处理数据所在分区,  
sampleRowId 当前样本处在该partition里第几行（跟原CSV不一定对得住，经过了分区）  
neighborRowId vector格式，当前样本与除自己之外的样本计算距离时记录的另外样本所在的行index  
distanceVector vector格式，当前样本与除自己之外的样本计算距离时记录的距离，与index对应  
举例：sampleData里边第0个partition的第0条样本与dataArr里第1,2,3条样本计算了距离且按计算顺序保存  
	(cal distance,0,0,0.5099019513592786,5.0,3.3,1.4,0.2, ** ,0,1,features,5.3,3.7,1.5,0.2)  
	(cal distance,0,0,0.4123105625617663,5.0,3.3,1.4,0.2, ** ,0,2,features,4.6,3.2,1.4,0.2)  
	(cal distance,0,0,0.5477225575051662,5.0,3.3,1.4,0.2, ** ,0,3,features,5.1,3.8,1.6,0.2)  
上边三条出自距离计算过程，即下边数据：  
(cal distance,0,1,0.5099019513592786,5.3,3.7,1.5,0.2, ** ,0,0,features,5.0,3.3,1.4,0.2)  
(cal distance,0,2,0.4123105625617663,4.6,3.2,1.4,0.2, ** ,0,0,features,5.0,3.3,1.4,0.2)  
(cal distance,0,3,0.5477225575051662,5.1,3.8,1.6,0.2, ** ,0,0,features,5.0,3.3,1.4,0.2)  
(cal distance,0,0,0.5099019513592786,5.0,3.3,1.4,0.2, ** ,0,1,features,5.3,3.7,1.5,0.2)  
(cal distance,0,2,0.8660254037844388,4.6,3.2,1.4,0.2, ** ,0,1,features,5.3,3.7,1.5,0.2)  
(cal distance,0,3,0.24494897427831785,5.1,3.8,1.6,0.2, ** ,0,1,features,5.3,3.7,1.5,0.2)  
(cal distance,0,0,0.4123105625617663,5.0,3.3,1.4,0.2, ** ,0,2,features,4.6,3.2,1.4,0.2)  
(cal distance,0,1,0.8660254037844388,5.3,3.7,1.5,0.2, ** ,0,2,features,4.6,3.2,1.4,0.2)  
(cal distance,0,3,0.8062257748298548,5.1,3.8,1.6,0.2, ** ,0,2,features,4.6,3.2,1.4,0.2)  
(cal distance,0,0,0.5477225575051662,5.0,3.3,1.4,0.2, ** ,0,3,features,5.1,3.8,1.6,0.2)  
(cal distance,0,1,0.24494897427831785,5.3,3.7,1.5,0.2, ** ,0,3,features,5.1,3.8,1.6,0.2)  
(cal distance,0,2,0.8062257748298548,4.6,3.2,1.4,0.2, ** ,0,3,features,5.1,3.8,1.6,0.2)  

计算好之后需要进行距离排序，取前K个最近的点  

## 三、合成两个近邻点之间的采样点  
1.数据抽取：rand.nextInt(kNN)表示给定一个参数n，nextInt(n)将返回一个大于等于0小于n的随机数，即：0 <= nextInt(n) < n。在生成的K个近邻里随机取一个点与sampleData的正样本点合成采样点。  
此步骤的结果sampleDataNearestNeighbors: Array[(Int, Int, Int, LabeledPoint)]各列内容：  
[dataArr分区，sampleData样本所在行，dataArr某个分区里样本所在行，sampleData里的样本]  
2.合成两点连线上的随机点即为采样点  
功能函数：createSyntheticData  
点生成公式：sampleFeatures += (features - sampleFeatures) * rand  

## 四、过采样样本生成策略  
过采样目标一般而言需要保证最终正样本与负样本数量相当  
即 numSyn == numNeg - numPos,其中numNeg = numAll - numPos  
需要循环采样的次数creationFactor = math.ceil(numSyn / numPos).toInt  
循环采样并非重采样，random尽量保证每次生成的点不重合  
举例：iris数据numNeg=50,numPos=4,numSyn=46,creationFactor=46/4=12,在12轮里每次rand都会随机生成一次  



