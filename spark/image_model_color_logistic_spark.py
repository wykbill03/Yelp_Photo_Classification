import os
import sys
import pandas as pd

from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext()

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_KEY)
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET)

#conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
#sc = SparkContext(conf = conf)
#sc = SparkContext()


deeptraindirectory = 's3n://usf-aml/project/F1_trainfinal.csv'
deeptestdirectory = 's3n://usf-aml/project/F1_testfinal.csv'
trainset = 's3n://usf-aml/project/pic_label_subbybus_train.csv'
testset = 's3n://usf-aml/project/pic_label_subbybus_test.csv'

deepfeature = sc.textFile(deeptraindirectory)
deepfeaturetest = sc.textFile(deeptestdirectory)
#print deepfeature.count() 61719
train_label = sc.textFile(trainset)
test_label = sc.textFile(testset)

noheaderfeature = deepfeature.filter(lambda x: 'pic_id' not in x)
noheadertrain = train_label.filter(lambda x: 'photo_id' not in x)
noheaderfeaturetest = deepfeaturetest.filter(lambda x: 'pic_id' not in x)
noheadertest = test_label.filter(lambda x: 'photo_id' not in x)

# line =  noheaderfeature.take(1)
# line_train = noheadertrain.take(1)

def parse_feature(line):
    image_id = line.split(',')[0]
    feature = line.split(',')[1].strip('"[]').strip("'")
    feature = [float(x) for x in feature.split(' ')]
    return {'image_id':image_id,'feature':feature}

def parse_y(line,labelnum):
    image_id = line.split(',')[0]
    label = int(line.split(',')[labelnum+2])
    return {'image_id':image_id,'label':label}

trainfeature = noheaderfeature.map(parse_feature).map(lambda x: (x['image_id'],x['feature']))
trainfeature.repartition(120).cache()
testfeature = noheaderfeaturetest.map(parse_feature).map(lambda x: (x['image_id'],x['feature']))
#print feature_rdd.count()
testfeature.repartition(120).cache()

######label0
trainy= noheadertrain.map(lambda x: parse_y(x,0)).map(lambda x: (x['image_id'],x['label']))
testy = noheadertest.map(lambda x: parse_y(x,0)).map(lambda x: (x['image_id'],x['label']))
#
train_set = trainfeature.join(trainy).map(lambda x: LabeledPoint(x[1][1], x[1][0]))
test_set = testfeature.join(testy).map(lambda x: LabeledPoint(x[0], x[1][0]))


Logisticmodel = LogisticRegressionWithLBFGS.train(train_set)
Logisticmodel.clearThreshold()
labelsAndPreds = test_set.map(lambda x: (x.label, Logisticmodel.predict(x.features)))
result0=pd.DataFrame(labelsAndPreds.collect())
result0.columns = ['ID', 'Label0']

#####label1
for i in range(1,9):
    trainy= noheadertrain.map(lambda x: parse_y(x,i)).map(lambda x: (x['image_id'],x['label']))
    testy = noheadertest.map(lambda x: parse_y(x,i)).map(lambda x: (x['image_id'],x['label']))
    #
    train_set = trainfeature.join(trainy).map(lambda x: LabeledPoint(x[1][1], x[1][0]))
    test_set = testfeature.join(testy).map(lambda x: LabeledPoint(x[0], x[1][0]))

    Logisticmodel = LogisticRegressionWithLBFGS.train(train_set)
    Logisticmodel.clearThreshold()
    labelsAndPreds = test_set.map(lambda x: (x.label, Logisticmodel.predict(x.features)))
    result1=pd.DataFrame(labelsAndPreds.collect())
    result1.columns = ['ID', 'Label1']
    result0 =  pd.merge(result0, result1, on='ID', how='left')

result0.columns = ['ID', 'Label0','Label1','Label2','Label3','Label4','Label5','Label6','Label7','Label8']
result0.to_csv('colorf1log.csv',index=False)
