# A known issue: https://stackoverflow.com/questions/44403010/pyspak-ml-vectorindexer-not-working-the-way-it-is-supposed-to-work

import os
from os import listdir
from os.path import isfile, join
from pyspark.sql import *
import logging

from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql.functions import lit
import subprocess
from pyspark.sql.types import *
#from pyspark.sql.functions import udf,when

import pyspark.sql.functions as sf

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark as ps


conf = ps.SparkConf()
conf.set("spark.executor.heartbeatInterval","36000000s")
conf.set("spark.network.timeout","36000000s")
conf.set("spark.rpc.lookupTimeout","36000000s")
conf.set("spark.files.fetchTimeout","36000000s")
sc = ps.SparkContext(conf=conf)

f_schema = StructType([
StructField("c1",LongType(),True),
StructField("c2",IntegerType(),True),
StructField("c3",LongType(),True),
StructField("c4",StringType(),True),
StructField("c5",StringType(),True),
StructField("c6",StringType(),True),
StructField("c7",DoubleType(),True),
StructField("c8",IntegerType(),True)])

sqlContext = SQLContext(sc)
data_path = "gs://bucket/path"
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(f_schema).load(data_path)
#data.cache()

all_classes = ["label1","label2","label3","label4","label5","label6"]

filter_classes = ["label1","label3","label6",]

## filter data by failure type, this will also filter out all the data (failure+healthy)
## from that failure class
#data = data.filter(data.SubComponent.isin(filter_classes))

# Filter failure folder, before failure data only, for the selected classes
data = data.where((col("c6").isin(filter_classes)) & (sf.col('c2') == 1) & (sf.col('c7').cast(LongType()) >= sf.col('c3')))
#data = data.where((data.SubComponent.isin(filter_classes)) & (data.failed_folder == 1)& (data.failutc.cast(LongType()) >= data.AcqTime))

data = data.withColumn('failure_class', sf.when((data.failutc.cast(LongType()) <= (86400 + data.AcqTime)), data.SubComponent).otherwise("all_good"))

num_col_toUse_names= ["c1","c2","c3","c7","c8"]
char_col_toUse_names  = ["c4","c5","c6"]
bool_col_toUse_names  = []
class_label_name = "failure_class"

do_one_hot = True
do_vectorIndexer = False

max_cat_count = 1550

for name in num_col_toUse_names:
    data = data.withColumn(name, sf.when( data[name].isNull(), 0).otherwise(data[name]))

for name in char_col_toUse_names:
    data = data.withColumn(name, sf.when( data[name].isNull(), "empty").otherwise(data[name]))

data = data.withColumn(class_label_name, sf.when(data[class_label_name].isNull(), "empty").otherwise(data[class_label_name]))
data.cache()

labelIndexer = StringIndexer(inputCol = class_label_name, outputCol="indexedLabel").fit(data)

string_feature_indexers = [
   StringIndexer(inputCol=x, outputCol="int_{0}".format(x)).fit(data)
   for x in char_col_toUse_names
]

if do_one_hot:
    onehot_encoder = [
       OneHotEncoder(inputCol="int_"+x, outputCol="onehot_{0}".format(x))
       for x in char_col_toUse_names
    ]
    all_columns = num_col_toUse_names + bool_col_toUse_names + ["onehot_"+x for x in char_col_toUse_names]
    assembler = VectorAssembler(inputCols=[col for col in all_columns], outputCol="features")    
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=100)        
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

    pipeline = Pipeline(stages=[labelIndexer] + string_feature_indexers + onehot_encoder + [assembler, rf, labelConverter])

elif do_vectorIndexer:
    all_columns = num_col_toUse_names + bool_col_toUse_names + ["int_"+x for x in char_col_toUse_names]
    assembler = VectorAssembler(inputCols=[col for col in all_columns], outputCol="features")
    data_temp = Pipeline(stages=string_feature_indexers+[assembler]).fit(data).transform(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=max_cat_count).fit(data_temp)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=100, maxBins= max_cat_count)    
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
    
    pipeline = Pipeline(stages=[labelIndexer] + string_feature_indexers + [assembler, featureIndexer, rf, labelConverter])

else:
    print("be sure of what you doing...")
    sys.exit(0)


(trainingData, testData) = data.randomSplit([0.9, 0.1])

#print("----------------------------------------------")
#print("Model fitting")
#model = pipeline.fit(trainingData)
#predictions = model.transform(testData)
#predictions.select("predictedLabel", class_label_name, "features").show(5)
#predictions.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("gs://abhi-ml/data/first_prediction")
#
#
#print("Evaluation matrix calculations")
## Select (prediction, true label) and compute test error
#evaluator = MulticlassClassificationEvaluator(
#    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
#accuracy = evaluator.evaluate(predictions)
#print("----------------------------------------------------------")
#print("Test Error = %g" % (1.0 - accuracy))
#
#rfModel = model.stages[-2]
#print("----------------------------------------------------------")
#print(rfModel)  # summary only

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()


evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
#MulticlassClassificationEvaluator
#metricName can be (f1|weightedPrecision|weightedRecall|accuracy)
    
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

cvModel = crossval.fit(trainingData)

prediction = cvModel.transform(testData)
selected = prediction.select("predictedLabel", class_label_name, "features")
selected.show(5)

if do_one_hot:
    prediction.write.format("com.databricks.spark.csv").option("header", "true").save("gs://bucket/path1")
elif do_vectorIndexer:
    prediction.write.format("com.databricks.spark.csv").option("header", "true").save("gs://bucket/path2")
