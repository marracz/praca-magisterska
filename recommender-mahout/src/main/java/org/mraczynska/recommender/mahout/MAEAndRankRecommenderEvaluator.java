package org.mraczynska.recommender.mahout;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.*;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Author: Marta Raczyńska
 */
public class MAEAndRankRecommenderEvaluator /*implements RecommenderEvaluator*/ {

    private static final Logger log = LoggerFactory.getLogger(MAEAndRankRecommenderEvaluator.class);

    private final Random random;
    private float maxPreference;
    private float minPreference;

    private RunningAverage runningMAE;
    private RunningAverage runningDCG;
    private RunningAverage runningNDCG;
    private int noEstimatesCount = 0;
    private double avgEstimateBaseOnCount;

    protected MAEAndRankRecommenderEvaluator() {
        random = RandomUtils.getRandom();
        maxPreference = Float.NaN;
        minPreference = Float.NaN;
    }

//    @Override
    public final float getMaxPreference() {
        return maxPreference;
    }

//    @Override
    public final void setMaxPreference(float maxPreference) {
        this.maxPreference = maxPreference;
    }

//    @Override
    public final float getMinPreference() {
        return minPreference;
    }

//    @Override
    public final void setMinPreference(float minPreference) {
        this.minPreference = minPreference;
    }

//    @Override
    public double evaluate(RecommenderBuilder recommenderBuilder,
                           DataModel trainingDataModel,
                           DataModel testDataModel) throws TasteException {
        Preconditions.checkNotNull(recommenderBuilder);
        Preconditions.checkNotNull(trainingDataModel);
        Preconditions.checkNotNull(testDataModel);

        int numUsers = trainingDataModel.getNumUsers();
        FastByIDMap<PreferenceArray> trainingPrefs = new FastByIDMap<>(numUsers);
        FastByIDMap<PreferenceArray> testPrefs = new FastByIDMap<>(numUsers);

        LongPrimitiveIterator it = trainingDataModel.getUserIDs();
        while (it.hasNext()) {
            long userID = it.nextLong();
            readOneUserFromModelToPrefMap(trainingPrefs, userID, trainingDataModel);
            readOneUserFromModelToPrefMap(testPrefs, userID, testDataModel);
        }

        DataModel trainingModel = new GenericDataModel(trainingPrefs);

        Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);

        double result = getEvaluation(testPrefs, recommender);
        return result;

//        if (!(recommender instanceof MyGenericUserBasedRecommender)) {
//            throw new AssertionError("error, recommender should be instanceof MyGenericUserBasedRecommender");
//        }
//        MyGenericUserBasedRecommender myRecommender = (MyGenericUserBasedRecommender) recommender;
//
//        double result = getEvaluation(testPrefs, recommender);
//
//        avgEstimateBaseOnCount = myRecommender.getAvgEstimateBaseOnCount();
        //return result;

    }

    public Map<String, Double> evaluateAllMetrics(RecommenderBuilder recommenderBuilder,
                                                  DataModel trainingDataModel,
                                                  DataModel testDataModel) throws TasteException {
        evaluate(recommenderBuilder, trainingDataModel, testDataModel);
        return ImmutableMap.of(
                "MAE.ByRating", computeMAE(),
                "Predict.DCG", computeDCGRankError(),
                "Predict.nDCG", computeNDCGRankError(),
                "NoEstimates", (double)noEstimatesCount,
                "avgEstimateBaseOnCount", avgEstimateBaseOnCount);
    }

    private void readOneUserFromModelToPrefMap(FastByIDMap<PreferenceArray> preferencesMap,
                                               long userID,
                                               DataModel dataModel) throws TasteException {

        List<Preference> oneUserPrefs = new ArrayList<>();
        try {
            PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
            int size = prefs.length();
            for (int i = 0; i < size; i++) {
                Preference newPref = new GenericPreference(userID, prefs.getItemID(i), prefs.getValue(i));
                oneUserPrefs.add(newPref);
            }
            if (!oneUserPrefs.isEmpty()) {
                preferencesMap.put(userID, new GenericUserPreferenceArray(oneUserPrefs));
            }
        } catch (NoSuchUserException e) {
//            System.out.println(e.getMessage());
        }
    }

    private float capEstimatedPreference(float estimate) {
        if (estimate > maxPreference) {
            return maxPreference;
        }
        if (estimate < minPreference) {
            return minPreference;
        }
        return estimate;
    }

    private double getEvaluation(FastByIDMap<PreferenceArray> testPrefs, Recommender recommender)
            throws TasteException {
        reset();
        Collection<Callable<Void>> estimateCallables = new ArrayList<>();
        AtomicInteger noEstimateCounter = new AtomicInteger();
        for (Map.Entry<Long,PreferenceArray> entry : testPrefs.entrySet()) {
            estimateCallables.add(
                    new PreferenceEstimateCallable(recommender, entry.getKey(), entry.getValue(), noEstimateCounter));
        }
        log.info("Beginning evaluation of {} users", estimateCallables.size());
        RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
        execute(estimateCallables, noEstimateCounter, timing);


        double mae = computeMAE();
//        log.info("Evaluation result, MAE: {}", mae);
//        System.out.println("Evaluation result, MAE: " + mae);

        double dcg = computeDCGRankError();
//        log.info("Evaluation result, DCG: {}", dcg);
//        System.out.println("Evaluation result, DCG: " + dcg);

        double nDcg = computeNDCGRankError();
//        log.info("Evaluation result, NDCG: {}", nDcg);
//        System.out.println("Evaluation result, nDCG: " + nDcg);

        noEstimatesCount = noEstimateCounter.get();

        return mae;
    }

    protected static void execute(Collection<Callable<Void>> callables,
                                  AtomicInteger noEstimateCounter,
                                  RunningAverageAndStdDev timing) throws TasteException {

        Collection<Callable<Void>> wrappedCallables = wrapWithStatsCallables(callables, noEstimateCounter, timing);
        int numProcessors = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numProcessors);
        log.info("Starting timing of {} tasks in {} threads", wrappedCallables.size(), numProcessors);
        try {
            List<Future<Void>> futures = executor.invokeAll(wrappedCallables);
            // Go look for exceptions here, really
            for (Future<Void> future : futures) {
                future.get();
            }

        } catch (InterruptedException ie) {
            throw new TasteException(ie);
        } catch (ExecutionException ee) {
            throw new TasteException(ee.getCause());
        }

        executor.shutdown();
        try {
            executor.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            throw new TasteException(e.getCause());
        }
    }

    private static Collection<Callable<Void>> wrapWithStatsCallables(Iterable<Callable<Void>> callables,
                                                                     AtomicInteger noEstimateCounter,
                                                                     RunningAverageAndStdDev timing) {
        Collection<Callable<Void>> wrapped = new ArrayList<>();
        int count = 0;
        for (Callable<Void> callable : callables) {
            boolean logStats = count++ % 1000 == 0; // log every 1000 or so iterations
            wrapped.add(new StatsCallable(callable, logStats, timing, noEstimateCounter));
        }
        return wrapped;
    }


    protected void reset() {
        runningMAE = new FullRunningAverage();
        runningDCG = new FullRunningAverage();
        runningNDCG = new FullRunningAverage();
        noEstimatesCount = 0;
    }

    protected void processOneEstimate(float estimatedPreference, Preference realPref) {
        runningMAE.addDatum(Math.abs(realPref.getValue() - estimatedPreference));
    }

    protected void processAllEstimatesOfOneUser(Map<Long, Float> estimatedPreferences, Map<Long, Float> realPreferences, boolean alsoNDCG) {
        if(estimatedPreferences.isEmpty()) {
            return;
        }

        double dcg = 0.0;
        double nDcg = 0.0;

        List<Long> sortedItemsByEstimated = estimatedPreferences.entrySet().stream()
                .sorted((a, b) -> Float.compare(b.getValue(), a.getValue())) //descending
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        int i = 1;
        for(Long item: sortedItemsByEstimated) {
            float realRate = realPreferences.get(item);
            dcg += realRate/log2(i+1);
            i++;
        }
        runningDCG.addDatum(dcg);
        if(alsoNDCG) {
            List<Long> sortedItemsByReal = realPreferences.entrySet().stream()
                    .sorted((a, b) -> Float.compare(b.getValue(), a.getValue())) //descending
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());
            int j = 1;
            double idealDcg = 0.0;
            for(Long item: sortedItemsByReal) {
                float realRate = realPreferences.get(item);
                idealDcg += realRate/log2(j+1);
                j++;
            }

            if (idealDcg != 0.0) {
                // w razie wypadku gdy idealdcg jest 0.0 bo wszystkie preferencje wynoszą 0.0
                nDcg = dcg / idealDcg;
                runningNDCG.addDatum(nDcg);
            }
        }

    }

    private double log2(int x) {
        //log2(x)=log2(e)*loge(x)
        return 1.442695 * Math.log(x);
    }


    private double computeMAE() {
        return runningMAE.getAverage();
    }

    double computeDCGRankError() {
        return runningDCG.getAverage();
    }

    double computeNDCGRankError() { return runningNDCG.getAverage(); }

    public final class PreferenceEstimateCallable implements Callable<Void> {

        private final Recommender recommender;
        private final long testUserID;
        private final PreferenceArray prefs;
        private final AtomicInteger noEstimateCounter;

        public PreferenceEstimateCallable(Recommender recommender,
                                          long testUserID,
                                          PreferenceArray prefs,
                                          AtomicInteger noEstimateCounter) {
            this.recommender = recommender;
            this.testUserID = testUserID;
            this.prefs = prefs;
            this.noEstimateCounter = noEstimateCounter;
        }

        @Override
        public Void call() throws TasteException {
            Map<Long, Float> allEstimatedPreferencesOfCurrentUser = new HashMap<>();
            Map<Long, Float> allRealPreferencesOfCurrentUser = new HashMap<>();

            for (Preference realPref : prefs) {
                float estimatedPreference = Float.NaN;
                try {
                    estimatedPreference = recommender.estimatePreference(testUserID, realPref.getItemID());
                } catch (NoSuchUserException nsue) {
                    // It's possible that an item exists in the test data but not training data in which case
                    // NSEE will be thrown. Just ignore it and move on.
                    log.info("User exists in test data but not training data: {}", testUserID);
                } catch (NoSuchItemException nsie) {
                    log.info("Item exists in test data but not training data: {}", realPref.getItemID());
                }
                if (Float.isNaN(estimatedPreference)) {
                    noEstimateCounter.incrementAndGet();
                } else {
                    estimatedPreference = capEstimatedPreference(estimatedPreference);

                    processOneEstimate(estimatedPreference, realPref);

                    allEstimatedPreferencesOfCurrentUser.put(realPref.getItemID(), estimatedPreference);
                    allRealPreferencesOfCurrentUser.put(realPref.getItemID(), realPref.getValue());
                }
            }

            processAllEstimatesOfOneUser(allEstimatedPreferencesOfCurrentUser, allRealPreferencesOfCurrentUser, true);

            return null;
        }

    }
}
