package org.mraczynska.recommender.mahout;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.example.kddcup.track1.svd.ParallelArraysSGDFactorizer;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.neighborhood.CachingUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.ParallelSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD;

/**
 * Author: Marta Raczyńska
 */
public class RecommenderEval {

    public static void main(String[] args) throws Exception {
        validateArgsCount(args);

        Task task = Task.valueOf(args[0].toUpperCase());

        System.out.println(new Date());

        RecommenderBuilder recBuilder = (DataModel dataModel) -> {
            if("svd".equalsIgnoreCase(args[3])) {
                switch (args[4]) {
                    case "ParallelArraysSGDFactorizer":
                        return new SVDRecommender(dataModel, new ParallelArraysSGDFactorizer(dataModel, Integer.valueOf(args[5]), Integer.valueOf(args[6])));
                    case "ParallelSGDFactorizer":
                        return new SVDRecommender(dataModel, new ParallelSGDFactorizer(dataModel, Integer.valueOf(args[5]), 0.1, Integer.valueOf(args[6])));
                    case "RatingSGDFactorizer":
                        return new SVDRecommender(dataModel, new RatingSGDFactorizer(dataModel, Integer.valueOf(args[5]), Integer.valueOf(args[6])));
                    case "ALSWRFactorizer":
                        return new SVDRecommender(dataModel, new ALSWRFactorizer(dataModel, Integer.valueOf(args[5]), 0.1, Integer.valueOf(args[6])));
                    default:
                        throw new RuntimeException(String.format("'%s' given as 4th parameter. When 3rd parameter is 'svd' then 4th should be algorithm name", args[4]));
                }
            } else {
                UserSimilarity similarity = instantiateUserSimilarity(args[3], dataModel);
                UserNeighborhood neighborhood = new CachingUserNeighborhood(
                        new NearestNUserNeighborhood(Integer.parseInt(args[4]), similarity, dataModel), dataModel); //k testing
                return new MyGenericUserBasedRecommender(dataModel, neighborhood, similarity);
            }
        };


        String evalResult = null;
        if (task.isEvaluation()) {
            String trainingFilePath, testFilePath;
            if("svd".equalsIgnoreCase(args[3])) {
                trainingFilePath = args[7];
                testFilePath = args[8];
            } else {
                trainingFilePath = args[5];
                testFilePath = args[6];
            }
            DataModel trainingDataModel = instantiateModel(args[2], readDataFile(trainingFilePath));
            DataModel testDataModel = instantiateModel(args[2], readDataFile(testFilePath));
            evalResult = evaluationTask(trainingDataModel, testDataModel, recBuilder);
        }


        File inputFile = readDataFile(args[1]);
        DataModel model = instantiateModel(args[2], inputFile);

        long start1 = System.currentTimeMillis();
        Recommender recommender = recBuilder.buildRecommender(model);

        long end1 = System.currentTimeMillis();
        System.out.println(String.format("LEARNING (SVD) time=%f", (end1 - start1)/1000.0));
        final List<Long> users = getUsers(model, false);

        String recoResult = null;
        if (task.isRecommendation()) recoResult = generateRecommendationsForAllUsers(users, recommender, false, true);

        String f1Result = null;
        if (task.isF1()) f1Result = calculateF1Measure(model, recBuilder);




        if (evalResult != null) System.out.println(evalResult);
        if (recoResult != null) System.out.println(recoResult);
        if (f1Result != null) System.out.println(f1Result);

        System.out.println(new Date());
    }

    private static List<Long> getUsers(DataModel model, boolean shuffle) throws TasteException {
        final List<Long> users = new ArrayList<>();
        model.getUserIDs().forEachRemaining(users::add);
        if(shuffle)
            Collections.shuffle(users);
        return users;
    }

    private static String generateRecommendationsForAllUsers(List<Long> users, Recommender recommender, boolean print, boolean parallel) {

        long start = System.currentTimeMillis();

        Stream<Long> usersStream = users.stream();
        if (parallel) usersStream = usersStream.parallel();
        usersStream.forEach(id ->
        {
            try {
                if (print) {
                    List<RecommendedItem> recommendedItems = recommender.recommend(id, 10, false);
                    String items = recommendedItems.stream()
                            .map(x -> String.format("('%d', %.3f)", x.getItemID(), x.getValue()))
                            .collect(Collectors.joining(", "));
                    System.out.println(String.format("%d [%s]", id, items));
                } else {
                    recommender.recommend(id, 10, false);
                }
            } catch (TasteException e) {
                e.printStackTrace();
            }
        });

        long end = System.currentTimeMillis();

        //[sec]
        return String.format("RECOMMENDATION time=%f", (end - start)/1000.0);
    }

    private static File readDataFile(String inputFilePath) {
        File inputFile = new File(inputFilePath);
        if (!inputFile.exists()) {
            System.out.println("Input file doesn't exist!: " + inputFilePath);
            System.exit(-1);
        }
        return inputFile;
    }


    private static void validateArgsCount(String[] args) {
        if (args.length < 3) {
            System.out.println("Insufficient number of arguments given");
            System.out.println("Usage: recommender.jar DATA_FILE DATA_MODEL_CLASS SIMILARITY_MEASURE_CLASS");
            System.exit(-1);
        }
    }

    private static String calculateF1Measure(DataModel model, RecommenderBuilder recBuilder) {
        RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
        IRStatistics stats = null;
        try {
            stats = evaluator.evaluate(recBuilder,
                    null, model, null, 10, CHOOSE_THRESHOLD, 1.0);
        } catch (TasteException e) {
            e.printStackTrace();
        }
        return String.format("F1 SCORE=%s, PRECISION=%s , RECALL=%s, nDCG=%s", stats.getF1Measure(), stats.getPrecision(), stats.getRecall(), stats.getNormalizedDiscountedCumulativeGain());
    }


    private static DataModel instantiateModel(String modelClassName, File inputFile) {
        try {
            return (DataModel) Class.forName(modelClassName)
                    .getConstructor(File.class, boolean.class)
                    .newInstance(inputFile, false);
        } catch (InstantiationException | IllegalAccessException e) {
            System.out.println(String.format("Cannot create class '%s'. Maybe class has private constructor only?", modelClassName));
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            System.out.println(String.format("Cannot create class '%s'. Maybe class constructor has thrown exception?", modelClassName));
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            System.out.println(String.format("Cannot create class '%s'. Maybe class has no constructor with parameters (File, boolean)?", modelClassName));
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            System.out.println(String.format("Cannot create class '%s'. Remember to use full class name (with package). Ensure class is on classpath.", modelClassName));
            e.printStackTrace();
        } catch (ClassCastException e) {
            System.out.println(String.format("Cannot cast class '%s' to '%s'.", modelClassName, DataModel.class.getName()));
            e.printStackTrace();
        }

        System.out.println("Cannot create DataModel. Details above. Exiting...");
        System.exit(-1);
        return null;
    }

    private static UserSimilarity instantiateUserSimilarity(String similarityClassName, DataModel dataModel) {
        try {
            return (UserSimilarity) Class.forName(similarityClassName).getConstructor(DataModel.class).newInstance(dataModel);
        } catch (InstantiationException | IllegalAccessException e) {
            System.out.println(String.format("Cannot create class '%s'. Maybe class has private constructor only?", similarityClassName));
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            System.out.println(String.format("Cannot create class '%s'. Maybe class constructor has thrown exception?", similarityClassName));
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            System.out.println(String.format("Cannot create class '%s'. Maybe class has no constructor with parameters (DataModel)?", similarityClassName));
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            System.out.println(String.format("Cannot create class '%s'. Remember to use full class name (with package). Ensure class is on classpath.", similarityClassName));
            e.printStackTrace();
        } catch (ClassCastException e) {
            System.out.println(String.format("Cannot cast class '%s' to '%s'.", similarityClassName, DataModel.class.getName()));
            e.printStackTrace();
        }

        System.out.println("Cannot create UserSimilarity. Details above. Exiting...");
        System.exit(-1);
        return null;
    }

    private static String evaluationTask(DataModel trainingDataModel, DataModel testDataModel, RecommenderBuilder recBuilder) throws TasteException {
        MAEAndRankRecommenderEvaluator recEvaluator = new MAEAndRankRecommenderEvaluator();
        Map<String, Double> metrics = recEvaluator.evaluateAllMetrics(recBuilder, trainingDataModel, testDataModel);
        return String.format("EVALUATION MAE=%s, DCG=%s, nDCG=%s, NoEstimates=%s, avgEstimateBaseOnCount=%s",
                metrics.get("MAE.ByRating"), metrics.get("Predict.DCG"), metrics.get("Predict.nDCG"), metrics.get("NoEstimates"), metrics.get("avgEstimateBaseOnCount"));
    }

    private static void get100RecommendationsPerUserId(Long userId, DataModel model, RecommenderBuilder recBuilder) throws TasteException {
        Recommender recommender = recBuilder.buildRecommender(model);
        final HashMap<Long, Float> recommendations = new HashMap<>();
        final AtomicInteger i = new AtomicInteger(0);
        final List<Long> items = new ArrayList<>();
        model.getItemIDs().forEachRemaining(items::add);
        items.stream().limit(10000).forEach(itemId -> {
            if(i.get()%100==0){
                System.out.println(i);
            }
            i.incrementAndGet();
            try {
                Float rate = recommender.estimatePreference(userId, itemId);
                if(!rate.isNaN()){
                    System.out.println(itemId + " -> " + rate);
                    recommendations.put(itemId, rate);
                }
            } catch (TasteException e) {
                e.printStackTrace();
            }
        });
        List<Map.Entry<Long, Float>> collect = recommendations.entrySet().stream()
                .sorted((x, y) -> (int) (y.getValue() - x.getValue()))
                .limit(100)
                .collect(Collectors.toList());

        System.out.println(collect);
    }
}