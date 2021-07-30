####
# This is a prediction file.
####
import pyspark
import argparse
from pyspark.ml.regression import GBTRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import PipelineModel, Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler                    
from pyspark.ml.regression import GBTRegressor
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import MulticlassMetrics


spark = SparkSession.builder.appName('CS643Proj2Prediction').getOrCreate()


def preprep(testdf):

    testTotalColumns = testdf.columns
    testdf = testdf.select(*(col(c).cast("double").alias(c) for c in testdf.columns))
    udfList = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stages = []

    for cName in testTotalColumns[:-1]:
        stages = []
        va = VectorAssembler(inputCols=[cName],outputCol=cName+'_vect')
        stages.append(va)
        stages.append(MinMaxScaler(inputCol=cName+'_vect', outputCol=cName+'_scaled'))
        pipeline = Pipeline(stages=stages)
        testdf = pipeline.fit(testdf).transform(testdf).withColumn(cName+"_scaled", udfList(cName+"_scaled")).drop(cName+"_vect").drop(cName)

    va = VectorAssembler(inputCols=[cName+"_scaled" for cName in testTotalColumns[:-1]],outputCol='features')
    testdf = vectorAssembler.transform(testdf)
    return testdf, testTotalColumns




def predict(testFilePath):
    testdf = spark.read.format("com.databricks.spark.csv").csv(
        testFilePath, header=True, sep=";")
    testdf, testTotalColumns = preprep(testdf)
    model = GBTRegressionModel.load("s3://cs643pro2out/Proj2gbt.model")
    testdf = model.transform(testdf)
    testdf = testdf.withColumn("prediction_with_round", func.round(testdf["prediction"], 0)).drop('prediction')
    testdf = testdf.select("prediction_with_round", testTotalColumns[-1])
    return testdf, testTotalColumns



def printAccuracyF1(testdf, testTotalColumns):
    labelCol = testTotalColumns[-1]
    predAndLabels = testdf.select(['prediction_with_round', testTotalColumns[-1]])
    l = testdf.select([labelCol]).distinct()
    predHeader = l.rdd.first()
    l = l.rdd.filter(lambda line: line !=predHeader)
    predHeader = predAndLabels.rdd.first()
    copyOfPredAndLabels = predAndLabels.rdd.filter(lambda line: line != predHeader)
    copyOfPredAndLabel = copyOfPredAndLabels.map(lambda lp: (float(lp[0]), float(lp[1])))
    scorer = MulticlassMetrics(copyOfPredAndLabel)
    pScore = scorer.precision()
    precallVal = scorer.recall()
    pf1Score = scorer.fMeasure()
    print("Precision = %s" % pScore)
    print("Recall = %s" % precallVal)
    print("F1 Score = %s" % pf1Score)

import argparse
arg_parser = argparse.ArgumentParser(description='Project 2 prediction')
arg_parser.add_argument('--test_file', required=True, help='Please pass in file path or S3 URL of your Test File')
args = arg_parser.parse_args()
df, testTotalColumns = predict(args.test_file)
printAccuracyF1(df, testTotalColumns)
