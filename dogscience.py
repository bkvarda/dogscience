# # Dog Science a
#
# The story of a dog who is also a scientist 


from __future__ import print_function
import sys
from random import random
from operator import add
from pyspark.sql import SparkSession
import logging

spark = SparkSession\
    .builder\
    .appName("Dog Science")\
    .getOrCreate()


import urllib
from IPython.display import Image

# # Navi is a dog
Image("images/IMG_1379.jpg")

# # Kind of a hipster

Image("images/IMG_0571.jpg")

# # She's also an incredible data scientist
Image("images/file-4.jpeg")


# # Her areas of expertise are vast and all encompasing
# # As a King County resident, she's interested in the placement of adoptable pets
!hdfs dfs -put data/Lost__found__adoptable_pets.csv

df = spark.read.csv('Lost__found__adoptable_pets.csv',header=True,inferSchema=True)

# # Our Schema Looks Like

df.printSchema()

# # Navi wants a taste for what's in this dataset, first she'll filter some things out
df.createOrReplaceTempView("adoptable_pets")
df2 = spark.sql("SELECT animal_type, Animal_Name, Animal_Breed, Age FROM adoptable_pets")
adoptable_pets = df2.filter("animal_type IS NOT NULL")
adoptable_pets.createOrReplaceTempView("adoptable_pets")
adoptable_pets.filter("Animal_Name IS NOT NULL").show(n=150,truncate=30)

# # As a scientist dog, she wants to know which breeds are the most common
spark.sql("SELECT animal_type, Animal_Breed FROM adoptable_pets").filter('animal_type = "Dog"').groupBy("Animal_Breed").count().orderBy("count",ascending=False).show(truncate=40)

# # Sometimes Navi likes to use Deep Learning libraries, like TensorFlow
!pip install tensorflow
!git clone https://github.com/tensorflow/models.git
  
# # What kind of dog is Navi?
Image("images/IMG_1379.jpg")
!python models/tutorials/image/imagenet/classify_image.py --image_file /home/cdsw/images/IMG_1379.jpg

# # What is this? 
Image("images/bigmac.jpg")
!python models/tutorials/image/imagenet/classify_image.py --image_file /home/cdsw/images/bigmac.jpg

# # This is one of my favorites - can the model tell what it is? 
Image("images/bone.jpeg")
!python models/tutorials/image/imagenet/classify_image.py --image_file /home/cdsw/images/bone.jpeg



spark.stop()