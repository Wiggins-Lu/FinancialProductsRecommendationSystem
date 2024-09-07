from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F

NUMBER_PRECISION = 2


def addSampleLabel(ratingSamples):
    ratingSamples.show(5, truncate=False)
    ratingSamples.printSchema()
    sampleCount = ratingSamples.count()
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         F.col('count') / sampleCount).show()
    ratingSamples = ratingSamples.withColumn('label', when(F.col('rating') >= 3.5, 1).otherwise(0))
    return ratingSamples


def extractReleaseYearUdf(title):
    # add realease year
    if not title or len(title.strip()) < 6:
        return 1990
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)


def addproductFeatures(productSamples, ratingSamplesWithLabel):
    # add product basic features
    samplesWithproducts1 = ratingSamplesWithLabel.join(productSamples, on=['productId'], how='left')
    # add releaseYear,title
    samplesWithproducts2 = samplesWithproducts1.withColumn('releaseYear',
                                                       udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')
    # split types
    samplesWithproducts3 = samplesWithproducts2.withColumn('productType1', split(F.col('types'), "\\|")[0]) \
        .withColumn('productType2', split(F.col('types'), "\\|")[1]) \
        .withColumn('productType3', split(F.col('types'), "\\|")[2])
    # add rating features
    productRatingFeatures = samplesWithproducts3.groupBy('productId').agg(F.count(F.lit(1)).alias('productRatingCount'),
                                                                    format_number(F.avg(F.col('rating')),
                                                                                  NUMBER_PRECISION).alias(
                                                                        'productAvgRating'),
                                                                    F.stddev(F.col('rating')).alias(
                                                                        'productRatingStddev')).fillna(0) \
        .withColumn('productRatingStddev', format_number(F.col('productRatingStddev'), NUMBER_PRECISION))
    # join product rating features
    samplesWithproducts4 = samplesWithproducts3.join(productRatingFeatures, on=['productId'], how='left')
    samplesWithproducts4.printSchema()
    samplesWithproducts4.show(5, truncate=False)
    return samplesWithproducts4


def extractTypes(types_list):
    '''
    pass in a list which format like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    count by each type，return type_list in reverse order
    eg:
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    return:['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    '''
    types_dict = defaultdict(int)
    for types in types_list:
        for type in types.split('|'):
            types_dict[type] += 1
    sortedTypes = sorted(types_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sortedTypes]


def addUserFeatures(samplesWithproductFeatures):
    extractTypesUdf = udf(extractTypes, ArrayType(StringType()))
    samplesWithUserFeatures = samplesWithproductFeatures \
        .withColumn('userPositiveHistory',
                    F.collect_list(when(F.col('label') == 1, F.col('productId')).otherwise(F.lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(F.col("userPositiveHistory"))) \
        .withColumn('userRatedproduct1', F.col('userPositiveHistory')[0]) \
        .withColumn('userRatedproduct2', F.col('userPositiveHistory')[1]) \
        .withColumn('userRatedproduct3', F.col('userPositiveHistory')[2]) \
        .withColumn('userRatedproduct4', F.col('userPositiveHistory')[3]) \
        .withColumn('userRatedproduct5', F.col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    F.count(F.lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', F.avg(F.col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', F.stddev(F.col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userAvgRating", format_number(
        F.avg(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)),
        NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", F.stddev(F.col("rating")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userTypes", extractTypesUdf(
        F.collect_list(when(F.col('label') == 1, F.col('types')).otherwise(F.lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userRatingStddev", format_number(F.col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userReleaseYearStddev", format_number(F.col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userType1", F.col("userTypes")[0]) \
        .withColumn("userType2", F.col("userTypes")[1]) \
        .withColumn("userType3", F.col("userTypes")[2]) \
        .withColumn("userType4", F.col("userTypes")[3]) \
        .withColumn("userType5", F.col("userTypes")[4]) \
        .drop("types", "userTypes", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)
    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(10)
    samplesWithUserFeatures.filter(samplesWithproductFeatures['userId'] == 1).orderBy(F.col('timestamp').asc()).show(
        truncate=False)
    return samplesWithUserFeatures


def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1)
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]
    training = smallSamples.where(F.col("timestampLong") <= splitTimestamp).drop("timestampLong")
    test = smallSamples.where(F.col("timestampLong") > splitTimestamp).drop("timestampLong")
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = 'file:///Users/WigginsLu/Workspace/FinancialProductsRecSys'
    productResourcesPath = file_path + "/sampledata/products.csv"
    ratingsResourcesPath = file_path + "/sampledata/ratings.csv"
    productSamples = spark.read.format('csv').option('header', 'true').load(productResourcesPath)
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)
    samplesWithproductFeatures = addproductFeatures(productSamples, ratingSamplesWithLabel)
    samplesWithUserFeatures = addUserFeatures(samplesWithproductFeatures)
    # save samples as csv format
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path + "/sampledata")
    # splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path + "/webroot/sampledata")
