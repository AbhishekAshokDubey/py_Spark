# A know issue: https://stackoverflow.com/questions/44403010/pyspak-ml-vectorindexer-not-working-the-way-it-is-supposed-to-work

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
StructField("_c0",LongType(),True),
StructField("WellEventID",IntegerType(),True),
StructField("WellStartTime",LongType(),True),
StructField("WellEndTime",LongType(),True),
StructField("WellName",StringType(),True),
StructField("Client",StringType(),True),
StructField("MobileHubID",StringType(),True),
StructField("MTID",StringType(),True),
StructField("FracCATVanSerial",StringType(),True),
StructField("Crew",StringType(),True),
StructField("AcqTime",LongType(),True),
StructField("PumpID",IntegerType(),True),
StructField("DischargePressure",IntegerType(),True),
StructField("PumpRate",DoubleType(),True),
StructField("ThrottleAct",IntegerType(),True),
StructField("EnginePercentLoad",IntegerType(),True),
StructField("EngineBoostPressure",DoubleType(),True),
StructField("EngineFuelPressure",DoubleType(),True),
StructField("EngineFuelRate",DoubleType(),True),
StructField("EngineOilPressure",DoubleType(),True),
StructField("EngineCoolantTemp",IntegerType(),True),
StructField("EngineVoltage",DoubleType(),True),
StructField("TransLockupPressure",DoubleType(),True),
StructField("TransConverterTemp",IntegerType(),True),
StructField("TransSumpTemp",IntegerType(),True),
StructField("TransMainOilPressure",DoubleType(),True),
StructField("TransFilterInPressure",DoubleType(),True),
StructField("TransFilterOutPressure",DoubleType(),True),
StructField("PowerEndOilTemp",IntegerType(),True),
StructField("PowerEndOilPressure",DoubleType(),True),
StructField("PowerEndStrokeCount",IntegerType(),True),
StructField("FluidEndSuctionPressure",DoubleType(),True),
StructField("FluidEndSuctionDampChargePressure",DoubleType(),True),
StructField("InstantIdle",StringType(),True),
StructField("LockupInhibit",StringType(),True),
StructField("EngineWarmup",StringType(),True),
StructField("TransmissionWarmup",StringType(),True),
StructField("PressureTest",StringType(),True),
StructField("PrimeUp",StringType(),True),
StructField("PumpBrake",StringType(),True),
StructField("SensorFailures",DoubleType(),True),
StructField("WarningsShutdowns",DoubleType(),True),
StructField("EngineHours",DoubleType(),True),
StructField("EngineGARState",StringType(),True),
StructField("Gear",DoubleType(),True),
StructField("AutoShutdownControl",DoubleType(),True),
StructField("DischargePSIRange",DoubleType(),True),
StructField("EngineType",DoubleType(),True),
StructField("PowerEndType",DoubleType(),True),
StructField("TransmissionType",DoubleType(),True),
StructField("FluidEndSize",DoubleType(),True),
StructField("ThrottleSetpoint",IntegerType(),True),
StructField("PressureSetpoint",IntegerType(),True),
StructField("SubComponent",StringType(),True),
StructField("failutc",DoubleType(),True),
StructField("failed_folder",IntegerType(),True)])

sqlContext = SQLContext(sc)
data_path = "gs://abhi-ml/data/phm_data_final"
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(f_schema).load(data_path)
#data.cache()

all_classes = ["Turbo", "Fuel - Injectors", "Cylinder block - Liner",
                "Engine - Gaskets", "Exahust - Piping", "Engine - Control - ECM",
                "Engine - Hydraulic - Starter", "Lubrication - Pump", "Engine - Sensors/switches",
                "Exhaust - Manifold", "Cylinder block", "Fuel - Filter", "Lubrication - Filters/Strainer",
                "Lubrication - Oil Pan", "Engine - Crankshaft - Seal", "Engine - Crankshaft",
                "Cylinder head", "Exahust - Muffler", "Inlet - Air cleaner housing", "nan",
                "Engine Assembly -", "Engine - Hydraulic - Hoses/piping/gasket", "Inlet - Aftercooler - Core",
                "Cylinder block - Cover", "Fuel - Priming pump", "Engine Assembly - Charged Air System",
                "Valve cover - Rocker Arm", "Camshaft", "Fuel - Hoses/lines/gaskets", "Cylinder head - Valve Assembly",
                "Fuel - Tank", "Inlet - Piping", "Engine - Cables/Harness/Wiring", "ECM -", "Fuel - Transfer pump",
                "Piston", "Valve cover - Pushrod", "Flywheel", "Exhaust - Piping", "Engine Fluids - Engine Lube Oil System",
                "Engine -  Aftercooler - Heat exchanger", "Engine Fluids - Engine Water System",
                "Inlet - Emergency shutoff valves", "Engine - Gear - Accessory drive", "Engine Fluids - Engine Fuel System",
                "Engine Assembly - Exhaust", "Vibration damper", "Camshaft - Bearing", "Exhaust - Muffler",
                "Engine - Crankshaft - Bearing", "Hydraulic Starter -", "Engine - Gear - Idler", "Engine - Mounting",
                "Engine - Hydraulic - Quick Disconects", "Tractor - Hydraulic - Pump", "Valve cover", "Fuel - Valves"]

filter_classes = ["Turbo", "Fuel - Injectors", "Cylinder block - Liner",
                "Engine - Gaskets", "Exahust - Piping", "Engine - Control - ECM",
                "Engine - Hydraulic - Starter", "Lubrication - Pump", "Engine - Sensors/switches",
                "Exhaust - Manifold", "Cylinder block", "Fuel - Filter", "Lubrication - Filters/Strainer",
                "Lubrication - Oil Pan", "Engine - Crankshaft - Seal"]

## filter data by failure type, this will also filter out all the data (failure+healthy)
## from that failure class
#data = data.filter(data.SubComponent.isin(filter_classes))

# Filter failure folder, before failure data only, for the selected classes
data = data.where((col("SubComponent").isin(filter_classes)) & (sf.col('failed_folder') == 1) & (sf.col('failutc').cast(LongType()) >= sf.col('AcqTime')))
#data = data.where((data.SubComponent.isin(filter_classes)) & (data.failed_folder == 1)& (data.failutc.cast(LongType()) >= data.AcqTime))

data = data.withColumn('failure_class', sf.when((data.failutc.cast(LongType()) <= (86400 + data.AcqTime)), data.SubComponent).otherwise("all_good"))

num_col_toUse_names= ["WellEventID", "WellStartTime","WellEndTime","AcqTime","PumpID","DischargePressure",
                      "PumpRate","ThrottleAct","EnginePercentLoad","EngineBoostPressure","EngineFuelPressure",
                      "EngineFuelRate","EngineOilPressure","EngineCoolantTemp","EngineVoltage",
                      "TransLockupPressure","TransConverterTemp","TransSumpTemp","TransMainOilPressure",
                      "TransFilterInPressure","TransFilterOutPressure","PowerEndOilTemp","PowerEndOilPressure",
                      "PowerEndStrokeCount","FluidEndSuctionPressure","FluidEndSuctionDampChargePressure",
                      "SensorFailures","WarningsShutdowns","EngineHours","Gear","AutoShutdownControl",
                      "DischargePSIRange","EngineType","PowerEndType","TransmissionType","FluidEndSize",
                      "ThrottleSetpoint","PressureSetpoint"]
char_col_toUse_names  = ["WellName","Client","MobileHubID","MTID","FracCATVanSerial", "Crew", "InstantIdle",
                         "LockupInhibit","EngineWarmup","TransmissionWarmup","PressureTest","PrimeUp","PumpBrake",
                         "EngineGARState"]
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
    prediction.write.format("com.databricks.spark.csv").option("header", "true").save("gs://abhi-ml/data/first_prediction/onehot")
elif do_vectorIndexer:
    prediction.write.format("com.databricks.spark.csv").option("header", "true").save("gs://abhi-ml/data/first_prediction/vectorIndexer")
