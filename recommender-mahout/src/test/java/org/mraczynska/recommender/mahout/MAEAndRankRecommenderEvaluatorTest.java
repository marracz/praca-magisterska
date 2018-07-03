package org.mraczynska.recommender.mahout;

import com.google.common.collect.ImmutableMap;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;


/**
 * Author: Marta Raczyńska
 */
public class MAEAndRankRecommenderEvaluatorTest {

    private static final ImmutableMap<Long, Float> USER1_ESTIMATED = ImmutableMap.of(1001L, 4.2f, 1002L, 4.3f, 1003L, 4.5f, 1004L, 2.8f);
    private static final ImmutableMap<Long, Float> USER1_REAL = ImmutableMap.of(1001L, 4.0f, 1002L, 5.0f, 1003L, 3.0f, 1004L, 3.0f);
    private static final ImmutableMap<Long, Float> USER2_ESTIMATED = ImmutableMap.of(1001L, 4.2f, 1002L, 4.3f, 1003L, 4.0f, 1004L, 2.8f);
    private static final ImmutableMap<Long, Float> USER2_REAL = ImmutableMap.of(1001L, 4.0f, 1002L, 5.0f, 1003L, 3.0f, 1004L, 3.0f);
    private static final ImmutableMap<Long, Float> USER3_ESTIMATED = ImmutableMap.of(1001L, 4.2f, 1002L, 4.0f, 1003L, 4.5f, 1004L, 4.8f);
    private static final ImmutableMap<Long, Float> USER3_REAL = ImmutableMap.of(1001L, 4.0f, 1002L, 5.0f, 1003L, 3.0f, 1004L, 3.0f);

    private MAEAndRankRecommenderEvaluator sut;

    @Before
    public void setUp() {
        sut = new MAEAndRankRecommenderEvaluator();
    }

    @Test
    public void processAllEstimatesOfOneUser_calculateEachUserSeparately() {
        sut.reset();
        test(USER1_ESTIMATED, USER1_REAL, 9.4467, 0.9158);

        sut.reset();
        test(USER2_ESTIMATED, USER2_REAL, 10.3157, 1.0);

        sut.reset();
        test(USER3_ESTIMATED, USER3_REAL,9.0462, 0.8769);
    }

    private void test(ImmutableMap<Long, Float> estimatedPreferences,
                      ImmutableMap<Long, Float> realPreferences,
                      double expectedDcg, double expectedNdcg) {
        sut.processAllEstimatesOfOneUser(estimatedPreferences, realPreferences, true);
        double dcg = sut.computeDCGRankError();
        double ndcg = sut.computeNDCGRankError();
        assertThat(dcg, closeTo(expectedDcg, 0.0001));
        assertThat(ndcg, closeTo(expectedNdcg, 0.0001));
    }

    @Test
    public void processAllEstimatesOfOneUser_accumulatedAverageForAllUsers() {
        sut.reset();
        sut.processAllEstimatesOfOneUser(USER1_ESTIMATED, USER1_REAL, true);
        sut.processAllEstimatesOfOneUser(USER2_ESTIMATED, USER2_REAL, true);
        sut.processAllEstimatesOfOneUser(USER3_ESTIMATED, USER3_REAL, true);

        double dcg = sut.computeDCGRankError();
        double ndcg = sut.computeNDCGRankError();
        assertThat(dcg, closeTo(9.60287, 0.0001));
        assertThat(ndcg, closeTo(0.9309, 0.0001));
    }

    @Test
    public void processAllEstimatesOfOneUser_oneUserWithNoRatings_notTakenIntoConsideration() {
        sut.reset();
        sut.processAllEstimatesOfOneUser(USER1_ESTIMATED, USER1_REAL, true);
        sut.processAllEstimatesOfOneUser(USER2_ESTIMATED, USER2_REAL, true);
        sut.processAllEstimatesOfOneUser(USER3_ESTIMATED, USER3_REAL, true);
        sut.processAllEstimatesOfOneUser(Collections.emptyMap(), Collections.emptyMap(), true);

        double dcg = sut.computeDCGRankError();
        double ndcg = sut.computeNDCGRankError();
        assertThat(dcg, closeTo(9.60287, 0.0001));
        assertThat(ndcg, closeTo(0.9309, 0.0001));
    }
}