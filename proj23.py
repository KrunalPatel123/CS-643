#####
# This is a training file
####
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler                    
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('CS643Proj2').getOrCreate()


traindt= spark.read.format("com.databricks.spark.csv").csv(
    's3://cs643pro2in/TrainingDataset.csv', header=True, sep=";")
traindt.printSchema()


validationdt= spark.read.format("com.databricks.spark.csv").csv(
    's3://cs643pro2in/ValidationDataset.csv', header=True, sep=";")
validationdt.printSchema()

traindt=traindt.distinct()
validationdt=validationdt.distinct()


trainTotalColumns = traindt.columns
validationTotalColumns=validationdt.columns

from pyspark.sql.functions import col
def prePrep(dataset):
    return dataset.select(*(col(c).cast("double").alias(c) for c in dataset.columns))


traindt = prePrep(traindt)
validationdt = prePrep(validationdt)


traindt.show(10)
validationdt.show(10)


from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

udfList = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
stages = []

for cName in trainTotalColumns[:-1]:
    stages = []
    va = VectorAssembler(inputCols=[cName],outputCol=cName+'_vect')
    stages.append(va)
    stages.append(MinMaxScaler(inputCol=cName+'_vect', outputCol=cName+'_scaled'))
    pipeline = Pipeline(stages=stages)
    traindt = pipeline.fit(traindt).transform(traindt).withColumn(cName+"_scaled", udfList(cName+"_scaled")).drop(cName+"_vect").drop(cName)

traindt.show(10)

for cName in trainTotalColumns[:-1]:
    stages = []
    va = VectorAssembler(inputCols=[cName],outputCol=cName+'_vect')
    stages.append(va)
    stages.append(MinMaxScaler(inputCol=cName+'_vect', outputCol=cName+'_scaled'))
    pipeline = Pipeline(stages=stages)
    validationdt = pipeline.fit(validationdt).transform(validationdt).withColumn(cName+"_scaled", udfList(cName+"_scaled")).drop(cName+"_vect").drop(cName)

validationdt.show(10)



va = VectorAssembler(inputCols=[cName+"_scaled" for cName in trainTotalColumns[:-1]],outputCol='features')
traindt.columns
proj2train = va.transform(traindt)
proj2valid = va.transform(validationdt)
proj2data_train = proj2train.select(['features',trainTotalColumns[-1]]).cache()
proj2data_valid = proj2valid.select(['features',trainTotalColumns[-1]]).cache()


proj2data_train.show(15)

proj2data_valid.show(15)


gbt = GBTRegressor(featuresCol='features', labelCol = trainTotalColumns[-1],maxIter=80,maxDepth=4,subsamplingRate=0.5,stepSize=0.1)
proj2model = gbt.fit(proj2data_train)
proj2predictions = proj2model.transform(proj2data_valid)
proj2predictions.select('prediction',trainTotalColumns[-1]).show(5)


proj2model.write().overwrite().save("s3://cs643pro2out/Proj2gbt.model")

gbtPrjo2Scorer= RegressionEvaluator(labelCol = trainTotalColumns[-1],predictionCol='prediction',metricName="rmse")
Scorermse = gbtPrjo2Scorer.evaluate(proj2predictions)


gbtPrjo2Scorer1= RegressionEvaluator(labelCol = trainTotalColumns[-1],predictionCol='prediction',metricName="r2")
Scorer2 = gbtPrjo2Scorer1.evaluate(proj2predictions)


print("RMS=%g" % Scorermse)
print("R squared = ",Scorer2)


