########## PART 1 ##########

df1 = spark.createDataFrame([Row(a="a",b=1),Row(a="b",b=1),Row(a="c",b=3),Row(a="c",b=4)])
df2 = spark.createDataFrame([Row(a="b",b=1),Row(a="b",b=1),Row(a="c",b=3)])
df_out = spark.createDataFrame([Row(k="l",b=1),Row(k="m",b=1),Row(k="n",b=3)])

s = StringIndexer(inputCol="a", outputCol="int_a").fit(df1)

pipeline = Pipeline(stages=[s])
m = pipeline.fit(df_out)
m.transform(df2).collect()


########## PART 2 ##########

train = spark.createDataFrame([Row(col1="b",col2=1,col3="a"),Row(col1="b",col2=1,col3="b"),Row(col1="c",col2=3,col3="a")])
test = spark.createDataFrame([Row(col1="a",col2=1,col3="c"),Row(col1="a",col2=1,col3="a"),Row(col1="c",col2=3,col3="a")])
all_data = train.unionAll(test)

s = [StringIndexer(inputCol=x, outputCol="int_{0}".format(x)).fit(all_data) for x in ["col1", "col3"]]

all_data_temp = all_data
temp_pipeline = Pipeline(stages=s)
all_data_temp1 = temp_pipeline.fit(all_data_temp).transform(all_data_temp)

all_data_temp = all_data
for k in s:
    all_data_temp = k.transform(all_data_temp)

all_data_temp1.show()
all_data_temp.show()
	
assembler = VectorAssembler(inputCols=["int_col1", "col2", "int_col3"], outputCol="features")

all_data_temp = assembler.transform(all_data_temp)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(all_data_temp)

pipeline = Pipeline(stages=s+[assembler,featureIndexer])

m = pipeline.fit(train)
m.transform(test).collect()
