from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


def oneHotEncoderExample(productSamples):
    samplesWithIdNumber = productSamples.withColumn("productIdNumber", F.col("productId").cast(IntegerType()))
    encoder = OneHotEncoderEstimator(inputCols=["productIdNumber"], outputCols=['productIdVector'], dropLast=False)
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.printSchema()
    oneHotEncoderSamples.show(10)


def array2vec(typeIndexes, indexSize):
    typeIndexes.sort()
    fill_list = [1.0 for _ in range(len(typeIndexes))]
    return Vectors.sparse(indexSize, typeIndexes, fill_list)


def multiHotEncoderExample(productSamples):
    samplesWithType = productSamples.select("productId", "title", explode(
        split(F.col("types"), "\\|").cast(ArrayType(StringType()))).alias('type'))
    typeIndexer = StringIndexer(inputCol="type", outputCol="typeIndex")
    StringIndexerModel = typeIndexer.fit(samplesWithType)
    typeIndexSamples = StringIndexerModel.transform(samplesWithType).withColumn("typeIndexInt",
                                                                                  F.col("typeIndex").cast(IntegerType()))
    indexSize = typeIndexSamples.agg(max(F.col("typeIndexInt"))).head()[0] + 1
    processedSamples = typeIndexSamples.groupBy('productId').agg(
        F.collect_list('typeIndexInt').alias('typeIndexes')).withColumn("indexSize", F.lit(indexSize))
    finalSample = processedSamples.withColumn("vector",
                                              udf(array2vec, VectorUDT())(F.col("typeIndexes"), F.col("indexSize")))
    finalSample.printSchema()
    finalSample.show(10)


def ratingFeatures(ratingSamples):
    ratingSamples.printSchema()
    ratingSamples.show()
    # calculate average product rating score and rating count
    productFeatures = ratingSamples.groupBy('productId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))
    productFeatures.show(10)
    # bucketing
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")
    # Normalization
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    productProcessedFeatures = featurePipeline.fit(productFeatures).transform(productFeatures)
    productProcessedFeatures.show(10)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = 'file:///Users/WigginsLu/Workspace/FinancialProductsRecSys'
    productResourcesPath = file_path + "/sampledata/products.csv"
    productSamples = spark.read.format('csv').option('header', 'true').load(productResourcesPath)
    print("Raw Product Samples:")
    productSamples.show(10)
    productSamples.printSchema()
    print("OneHotEncoder Example:")
    oneHotEncoderExample(productSamples)
    print("MultiHotEncoder Example:")
    multiHotEncoderExample(productSamples)
    print("Numerical features Example:")
    ratingsResourcesPath = file_path + "/sampledata/ratings.csv"
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingFeatures(ratingSamples)
