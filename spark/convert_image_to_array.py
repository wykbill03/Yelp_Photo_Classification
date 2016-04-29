from pyspark import SparkContext
from pyspark import SparkConf
from StringIO import StringIO
from PIL import Image
import numpy as np
import os, tempfile
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

image_array_flatten = image_array.map(lambda x: (x[0],x[1].flatten()))

image_array_flatten = image_array_flatten.map(lambda x: (x[0].split('/')[-1].\
                                                         replace('.jpg', '')," ".join(np.char.mod('%d', x[1]))))\
                                         .repartition(120).cache()

image_array_flatten.saveAsTextFile("s3n://amlyelp/subset/train_image_array/")

