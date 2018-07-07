#! python
import sys
import subprocess

JAR = "./recommender-lenskit-1.0-SNAPSHOT-jar-with-dependencies.jar"
MAIN_CLASS = "org.mraczynska.recommender.lenskit.RecommenderEval"
TASK = "evaluation"
DATASET = "./BX-Book-Ratings.csv"
ALG_CLASSES = ["org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity"]
NEIGHBORHOOD_SIZE = [80,90,100,150,200,250,300,400,500,750,1000,1500,2000,3000,4000,5000,10000]

for alg in ALG_CLASSES:
	for fold in range(10):
		for k in NEIGHBORHOOD_SIZE:
			params = ["java", "-Xmx4096m", "-Xms2048m", "-cp", JAR, MAIN_CLASS, TASK, DATASET, alg, str(k), "./BX-Book-Ratings" + str(fold) + "_training.csv", "./BX-Book-Ratings" + str(fold) + "_test.csv"]
			print(params)
			subprocess.call(params)
