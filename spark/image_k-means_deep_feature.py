from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
from StringIO import StringIO
from PIL import Image
import numpy as np
import csv
import os, tempfile
import boto
import datetime

sc = SparkContext()

# AWS S3 credentials:

AWS_KEY = ""
AWS_SECRET = ""
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_KEY)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET)

directory = 's3n://amlyelp/subset/deepfeature_trainnew.csv'

data = sc.textFile(directory)

first = data.take(1)[0]

data = data.filter(lambda x: x!=first)

data_parsed = data.map(lambda x: x.replace('[','').replace(']','').replace('"','').split(','))\
                    .map(lambda x: (x[0], np.array(map(float, x[1].split(' ')))))
del data

train = data_parsed.values().repartition(200).cache()

clusters = KMeans.train(train, 15, maxIterations=100)

clusters.save(sc, 's3n://amlyelp/subset/model/kmeans/deep_feature_15c_'+\
              str(datetime.datetime.now()).replace(' ', '_').replace(':','_')+'/')

image_cluster = data_parsed.map(lambda x: (x[0],clusters.predict(x[1])))

image_cluster.saveAsTextFile("s3n://amlyelp/subset/image_cluster_result/image_cluster_deep_feature_15c_%s" %\
                             str(datetime.datetime.now()).replace(' ', '_').replace('.','_').replace(':','_'))


