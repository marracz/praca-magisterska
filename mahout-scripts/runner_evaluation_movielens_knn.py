#! python
import sys
import subprocess

TASK = "evaluation"

JAR = "./recommender-mahout-1.0-SNAPSHOT-jar-with-dependencies.jar"
MAIN_CLASS = "org.mraczynska.recommender.mahout.RecommenderEval"
ALG_CLASSES = ["org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity",
	       "org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity",
               "org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity",
               "org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity",
               "org.apache.mahout.cf.taste.impl.similarity.SpearmanCorrelationSimilarity",
	       "org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity"]
NEIGHBOURHOOD_SIZE = ["750","1000","1500","2000","3000","4000","5000","10000"]

# MOVIELENS
MODEL_CLASS = "org.mraczynska.recommender.mahout.MovieLensDataModel"
DATASET = "./MovieLens1M.csv"

for alg in ALG_CLASSES:
        for x in NEIGHBOURHOOD_SIZE:
                for fold in range(10):
                        params = ["java", "-Xmx4096m", "-Xms2048m", "-cp", JAR, MAIN_CLASS, TASK, DATASET, MODEL_CLASS, alg, x, "./MovieLens1M" + str(fold) + "_training.csv", "./MovieLens1M" + str(fold) + "_test.csv"]
                        print(params)
                        subprocess.call(params)	
