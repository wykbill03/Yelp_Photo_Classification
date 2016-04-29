from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
from StringIO import StringIO
# from PIL import Image
import numpy as np
import csv
import os, tempfile
import datetime

sc = SparkContext()
# AWS S3 credentials:

AWS_KEY = ""
AWS_SECRET = ""
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_KEY)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET)

directory = 's3n://amlyelp/subset/train_image_array/'

data = sc.textFile(directory)

data = data.map(lambda x: x.replace('(u','').replace(')','').replace("'",''))
data_parsed = data.map(lambda x: x.split(',')).map(lambda x: (x[0], np.fromstring(x[1], sep=' '))).cache()
del data

train = data_parsed.values().repartition(120).cache()

clusters = KMeans.train(train, 50, maxIterations=50)

clusters.save(sc, 's3n://amlyelp/subset/model/kmeans/50_iters_'+\
              str(datetime.datetime.now()).replace(' ', '_').replace('.','_').replace(':','_')+'/')

image_cluster = data_parsed.map(lambda x: (x[0],clusters.predict(x[1])))

image_cluster.saveAsTextFile("s3n://amlyelp/subset/image_cluster_result/image_cluster_50_iters_%s" %\
                             str(datetime.datetime.now()).replace(' ', '_').replace('.','_').replace(':','_'))

