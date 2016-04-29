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


directory = 's3n://amlyelp/subset/trainnew/'

images = sc.binaryFiles(directory)

image_to_array = lambda rawdata: np.asarray(Image.open(StringIO(rawdata)))

image_array = images.map(lambda x: (x[0],image_to_array(x[1])))

image_array_flatten = image_array.map(lambda x: (x[0],x[1].flatten())).cache()
del image_array
del images

train = image_array_flatten.values().repartition(200).cache()

clusters = KMeans.train(train, 50, maxIterations=50)

clusters.save(sc, 's3n://amlyelp/subset/model/kmeans/50_iters_'+\
              str(datetime.datetime.now()).replace(' ', '_')+'/')

image_cluster = image_array_flatten.map(lambda x: (x[0].split('/')[-1].replace('.jpg', ''),clusters.predict(x[1])))

image_cluster.saveAsTextFile("s3n://amlyelp/subset/image_cluster_result/image_cluster_"+\
                str(datetime.datetime.now()).replace(' ', '_').replace('.','_').replace(':','_'))


