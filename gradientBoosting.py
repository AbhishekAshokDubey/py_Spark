from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Tokenizer, HashingTF, VectorIndexer


conf = (SparkConf()
         .setAppName("My app")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)



sqlContext = SQLContext(sc)

char_col_toUse_names = ['c1' ,'c2','c3','c4','c5']
num_col_toUse_names = ['x1','x2', 'x3', 'x4']

taxiFile = sc.textFile("hdfs://path/file.csv")
#taxiFile.count()
header = taxiFile.first()
#header


fields = [StructField(field_name, StringType(), True) for field_name in header.split(',')]

fields[7].dataType = IntegerType()
fields[8].dataType = IntegerType()

fields[9].dataType = FloatType()
fields[10].dataType = FloatType()
fields[11].dataType = FloatType()
fields[12].dataType = FloatType()

schema = StructType(fields)
taxiFile = taxiFile.filter(lambda line:line != header)
# taxiFile.subtract(header)


taxi_temp_string = taxiFile.map(lambda k: k.split(",")).filter(lambda l: len(l) == 18)
taxi_temp_typed = taxi_temp_string.map(lambda p: (p[0],p[1],p[2],p[3],p[4],p[5],p[6], int(p[7]), int(p[8]), float(p[9]), float(p[10]), float(p[11]), float(p[12]), p[13], p[14], p[15],p[16],p[17]))
#taxi_temp_typed.collect()

taxi_df = sqlContext.createDataFrame(taxi_temp_typed, schema)
#taxi_df.head(10)
distinct_values_in_each_cat_var = [taxi_df.select(x).distinct().count() for x in char_col_toUse_names] 


######################################################################### With One hot #########################################################################

string_indexers = [
   StringIndexer(inputCol=x, outputCol="int_{0}".format(x))
   for x in char_col_toUse_names
]

encoder = [
   OneHotEncoder(inputCol="int_{0}".format(x), outputCol="one_hot_{0}".format(x))
   for x in char_col_toUse_names
]

assembler = VectorAssembler(
    inputCols= ["one_hot_"+x for x in char_col_toUse_names] + num_col_toUse_names,
    outputCol="features"
)


pipeline = Pipeline(stages = string_indexers + encoder + [assembler])
model = pipeline.fit(taxi_df)
indexed = model.transform(taxi_df)
ml_df = indexed.select(col("Tool Days").cast("int").alias("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))

training, test = ml_df.randomSplit([0.8, 0.2], seed=0)

gbm = GradientBoostedTrees.trainRegressor(sc.parallelize(training.collect()), categoricalFeaturesInfo={},  numIterations=3)

predictions = gbm.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda x: x.label).zip(predictions)

error = 1.0 * labelsAndPredictions.filter(lambda (p, a): a!=0).map(lambda (p, a): abs(p-a)/a).reduce(lambda a, b: a+b) / test.count()
print(".........................-----------------------=================================================================== Error with one hot encoding: "+ str(error))


######################################################################### WithOUT One hot #########################################################################


string_indexers = [
   StringIndexer(inputCol=x, outputCol="int_{0}".format(x))
   for x in char_col_toUse_names
]

assembler = VectorAssembler(
    inputCols= ["int_"+x for x in char_col_toUse_names] + num_col_toUse_names,
    outputCol="features"
)


pipeline = Pipeline(stages=string_indexers + [assembler])
model = pipeline.fit(taxi_df)
indexed = model.transform(taxi_df)
ml_df = indexed.select(col("Tool Days").cast("int").alias("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))

training, test = ml_df.randomSplit([0.8, 0.2], seed=0)

gbm = GradientBoostedTrees.trainRegressor(sc.parallelize(training.collect()), categoricalFeaturesInfo={0:24,1:3,2:4,3:5,4:107},  numIterations=3, maxBins=120)

predictions = gbm.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda x: x.label).zip(predictions)

error = 1.0 * labelsAndPredictions.filter(lambda (p, a): a!=0).map(lambda (p, a): abs(p-a)/a).reduce(lambda a, b: a+b) / test.count()
print(".........................-----------------------=================================================================== Error withOUT one hot encoding: "+ str(error))
