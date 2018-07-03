package org.mraczynska.recommender.lenskit;


import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import org.lenskit.LenskitConfiguration;
import org.lenskit.LenskitRecommender;
import org.lenskit.api.*;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.dao.file.StaticDataSource;
import org.lenskit.data.entities.EntityType;
import org.lenskit.eval.traintest.AlgorithmInstance;
import org.lenskit.eval.traintest.DataSet;
import org.lenskit.eval.traintest.EvalTask;
import org.lenskit.eval.traintest.TrainTestExperiment;
import org.lenskit.eval.traintest.predict.NDCGPredictMetric;
import org.lenskit.eval.traintest.predict.PredictEvalTask;
import org.lenskit.eval.traintest.recommend.RecommendEvalTask;
import org.lenskit.eval.traintest.recommend.TopNPrecisionRecallMetric;
import org.lenskit.knn.NeighborhoodSize;
import org.lenskit.knn.user.NeighborFinder;
import org.lenskit.knn.user.UserUserItemScorer;
import org.lenskit.util.table.Table;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Author: Marta Raczyńska
 */
public class RecommenderEval implements Runnable {

    public static void main(String[] args) {

        RecommenderEval eval = null;
        try {
            eval = new RecommenderEval(args);
        } catch (ClassNotFoundException e) {
            e.printStackTrace(System.err);
            System.exit(1);
        }

        try {
            eval.run();
        } catch (RuntimeException e) {
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static final String WORKING_DIR = "lenskitEvaluationDataSets";

    private final Task task;
    private final Mode mode;
    private final String dataSetFileName;
    private final Class<? extends VectorSimilarity> similarityMeasure;
    private final int neighborhoodSize;
    private final String trainFileName;
    private final String testFileName;

    public RecommenderEval(String[] args) throws ClassNotFoundException {
        task = Task.valueOf(args[0].toUpperCase());
        mode = Mode.valueOf(args[1].toUpperCase());
        dataSetFileName = args[2];
        similarityMeasure = (Class<? extends VectorSimilarity>) Class.forName(args[3]);
        neighborhoodSize = Integer.parseInt(args[4]);
        trainFileName = args[5];
        testFileName = args[6];
    }

    public void run() {
        LenskitConfiguration pearsonConfig = new LenskitConfiguration();
        pearsonConfig.bind(VectorSimilarity.class).to(similarityMeasure);
        switch (mode) {
            case ORIGINAL:
                pearsonConfig.bind(ItemScorer.class).to(UserUserItemScorer.class);
                break;
            case MAHOUT:
                pearsonConfig.bind(ItemScorer.class).to(MyUserUserItemScorer.class);
                pearsonConfig.bind(NeighborFinder.class).to(MyLiveNeighborFinder.class);
                break;
            default:
                throw new RuntimeException("Unknown mode given");
        }

        pearsonConfig.set(NeighborhoodSize.class).to(neighborhoodSize);

        String evalResult = null;
        if (task.isEvaluation()) evalResult = evaluate(pearsonConfig, trainFileName, testFileName);

        String f1Result = null;
        if (task.isF1()) f1Result = f1(pearsonConfig);

        String recoResult = null;
        if (task.isRecommendation()) recoResult = recommend(pearsonConfig, false, true);

        if (evalResult != null) System.out.println(evalResult);
        if (f1Result != null) System.out.println(f1Result);
        if (recoResult != null) System.out.println(recoResult);
    }

    private String recommend(LenskitConfiguration pearsonConfig, boolean print, boolean parallel) {
        DataAccessObject dao = StaticDataSource.csvRatingFile(Paths.get(dataSetFileName)).get();
        pearsonConfig.addComponent(dao);

        Recommender rec;
        try {
            rec = LenskitRecommender.build(pearsonConfig);
        } catch (RecommenderBuildException e) {
            throw new RuntimeException("recommender build failed", e);
        }

        ItemRecommender irec = rec.getItemRecommender();
        assert irec != null;

        LongSet users = dao.getEntityIds(EntityType.forName("user"));
        List<Long> every10thUser = new ArrayList<>();
        int i = 0;
        for (Long user : users) {
            if (i%10 == 0)
                every10thUser.add(user);
            ++i;
        }

        long start = System.currentTimeMillis();

        Stream<Long> usersStream = every10thUser.stream();
        if (parallel) usersStream = usersStream.parallel();
        usersStream.forEach(user -> {
            if (print) {
                ResultList recommendedItems = irec.recommendWithDetails(user, 10, null, null);
                String items = recommendedItems.stream()
                        .map(x -> String.format("('%d', %.3f)", x.getId(), x.getScore()))
                        .collect(Collectors.joining(", "));
                System.out.println(String.format("%d [%s]", user, items));
            } else {
                irec.recommendWithDetails(user, 10, null, null);
            }
        });

        long end = System.currentTimeMillis();

        return String.format("RECOMMENDATION time=%f", (end - start)/1000.0);
    }

    private String evaluate(LenskitConfiguration lenskitConfig, String trainFileName, String testFileName) {
        PredictEvalTask task = new PredictEvalTask();
        task.addMetric(new DCGPredictMetric());
        task.addMetric(new NDCGPredictMetric());

        Table table = runExperiment(lenskitConfig, trainFileName, testFileName, task);

        return String.format("EVALUATION MAE=%s, DCG=%s, nDCG=%s",
                table.column("MAE.ByRating"), table.column("Predict.DCG"), table.column("Predict.nDCG"));
    }

    private Table runExperiment(LenskitConfiguration lenskitConfig, String trainFileName, String testFileName, EvalTask task) {
        StaticDataSource trainDS = StaticDataSource.csvRatingFile(Paths.get(trainFileName));
        StaticDataSource testDS = StaticDataSource.csvRatingFile(Paths.get(testFileName));
        DataSet dataSet = new DataSet("fake", trainDS, null, testDS, UUID.randomUUID(), null);

        TrainTestExperiment trainTestExperiment = new TrainTestExperiment();
        trainTestExperiment.addDataSet(dataSet);
        trainTestExperiment.addAlgorithm(new AlgorithmInstance("algo", lenskitConfig));
        trainTestExperiment.setUserOutputFile(Paths.get("lenskit-eval-out.txt"));
        trainTestExperiment.addTask(task);

        return trainTestExperiment.execute();
    }

    private String f1(LenskitConfiguration lenskitConfig) {
        new File(WORKING_DIR).mkdir();

        long now = new Date().getTime();
        String trainFileName = WORKING_DIR + "\\" + now + "_f1_train.csv";
        String testFileName = WORKING_DIR + "\\" + now + "_f1_test.csv";

        divideDataSetForF1(10, trainFileName, testFileName);


        RecommendEvalTask task = new RecommendEvalTask();
        task.addMetric(new TopNPrecisionRecallMetric());

        Table table = runExperiment(lenskitConfig, trainFileName, testFileName, task);

        return String.format("F1 SCORE F1=%s, PRECISION=%s, RECALL=%s, nDCG=%s",
                table.column("F1"), table.column("Precision"), table.column("Recall"), table.column("TopN.nDCG"));
    }

    /**
     *
     * @param at count of ratings that will be written to test file for each user (best ratings of user)
     * @param trainFileName
     * @param testFileName
     */
    private void divideDataSetForF1(int at, String trainFileName, String testFileName) {
        try {
            PrintWriter trainFile = new PrintWriter(new FileWriter(trainFileName));
            PrintWriter testFile = new PrintWriter(new FileWriter(testFileName));

            Map<Long, List<Rating>> userToListOfRatings = Files.lines(Paths.get(dataSetFileName))
                    .map(line -> line.split(",|;"))
                    .map(Rating::new)
                    .collect(Collectors.groupingBy(x -> x.user));

            userToListOfRatings.values()
                    .forEach(ratings -> {
                        ratings.sort(Comparator.comparingInt(a -> -a.rate));

                        for (int i=0; i<at && i<ratings.size(); ++i) {
                            testFile.println(ratings.get(i));
                        }
                        for (int i=at; i<ratings.size(); ++i) {
                            trainFile.println(ratings.get(i));
                        }
                    });

            trainFile.close();
            testFile.close();

        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    private class Rating {
        private final long user;
        private final long item;
        private final int rate;

        public Rating(String[] strings) {
            user = Long.parseLong(strings[0]);
            item = Long.parseLong(strings[1]);
            rate = Integer.parseInt(strings[2]);
        }

        @Override
        public String toString() {
            return String.format("%s,%s,%s", user, item, rate);
        }
    }
}
