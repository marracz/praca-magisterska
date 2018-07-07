#! python
import sys
import subprocess

TASK = "evaluation"

JAR = "./recommender-mahout-1.0-SNAPSHOT-jar-with-dependencies.jar"
MAIN_CLASS = "org.mraczynska.recommender.mahout.RecommenderEval"
ALG_CLASSES = ["RatingSGDFactorizer", "ALSWRFactorizer", "ParallelSGDFactorizer", "ParallelArraysSGDFactorizer"]

NUM_FEATURES = ["6000"]
NUM_ITERATIONS = ["80","90","100","150","200","300"]

# MOVIELENS
MODEL_CLASS = "org.mraczynska.recommender.mahout.MovieLensDataModel"
DATASET = "./MovieLens1M.csv"

for alg in ALG_CLASSES:
	for x in NUM_FEATURES:
		for y in NUM_ITERATIONS:
			for fold in range(10):
				params = ["java", "-Xmx14336m", "-Xms2048m", "-cp", JAR, MAIN_CLASS, TASK, DATASET, MODEL_CLASS, "svd", alg, x, y, "./MovieLens1M" + str(fold) + "_training.csv", "./MovieLens1M" + str(fold) + "_test.csv"]
				print(params)
				subprocess.call(params)

	
