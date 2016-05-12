from pyspark import SparkConf, SparkContext
conf = (SparkConf().setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)

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

from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint, RidgeRegressionWithSGD, LinearRegressionWithSGD
from pyspark.sql.functions import col



assembler = VectorAssembler(inputCols=['x1','x2', 'x3', 'x4'],outputCol='features')
assembled_output = assembler.transform(taxi_df)
ml_df = assembled_output.select(col("Tool Days").alias("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))

training, test = ml_df.randomSplit([0.8, 0.2], seed=0)

lrm = LinearRegressionWithSGD.train(sc.parallelize(training.collect()), iterations=500, step=0.00001)

predictionAndLabel = test.map(lambda p: (lrm.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (p, a): a!=0).map(lambda (p, a): abs(p-a)/a).reduce(lambda a, b: a+b) / test.count()
accuracy

#model.save(sc, "target/tmp/myNaiveBayesModel")
#sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
