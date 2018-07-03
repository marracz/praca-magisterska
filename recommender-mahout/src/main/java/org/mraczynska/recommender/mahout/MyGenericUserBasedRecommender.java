package org.mraczynska.recommender.mahout;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.EstimatedPreferenceCapper;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.ArrayList;
import java.util.List;

/**
 * Author: Marta Raczyńska
 */
public class MyGenericUserBasedRecommender extends GenericUserBasedRecommender {
    private final UserSimilarity similarity;
    private EstimatedPreferenceCapper capper;
    private final List<Integer> estimateBaseOnCount;

    public MyGenericUserBasedRecommender(DataModel dataModel, UserNeighborhood neighborhood, UserSimilarity similarity) {
        super(dataModel, neighborhood, similarity);
        this.similarity = similarity;
        this.capper = buildCapper();
        this.estimateBaseOnCount = new ArrayList<>();
    }

    private EstimatedPreferenceCapper buildCapper() {
        DataModel dataModel = getDataModel();
        if (Float.isNaN(dataModel.getMinPreference()) && Float.isNaN(dataModel.getMaxPreference())) {
            return null;
        } else {
            return new EstimatedPreferenceCapper(dataModel);
        }
    }

    @Override
    protected float doEstimatePreference(long theUserID, long[] theNeighborhood, long itemID) throws TasteException {
        if (theNeighborhood.length == 0) {
            return Float.NaN;
        }
        DataModel dataModel = getDataModel();
        double preference = 0.0;
        double totalSimilarity = 0.0;
        int count = 0;
        for (long userID : theNeighborhood) {
            if (userID != theUserID) {
                // See GenericItemBasedRecommender.doEstimatePreference() too
                Float pref = dataModel.getPreferenceValue(userID, itemID);
                if (pref != null) {
                    double theSimilarity = similarity.userSimilarity(theUserID, userID);
                    if (!Double.isNaN(theSimilarity)) {
                        preference += theSimilarity * pref;
                        totalSimilarity += theSimilarity;
                        count++;
                    }
                }
            }
        }

        estimateBaseOnCount.add(count);

        // Throw out the estimate if it was based on no data points, of course, but also if based on
        // just one. This is a bit of a band-aid on the 'stock' item-based algorithm for the moment.
        // The reason is that in this case the estimate is, simply, the user's rating for one item
        // that happened to have a defined similarity. The similarity score doesn't matter, and that
        // seems like a bad situation.
        if (count <= 1) {
            return Float.NaN;
        }
        float estimate = (float) (preference / totalSimilarity);
        if (capper != null) {
            estimate = capper.capEstimate(estimate);
        }
        return estimate;
    }

    public double getAvgEstimateBaseOnCount() {
//        System.out.println(estimateBaseOnCount);
        long sum = 0;
        for (int i : estimateBaseOnCount) {
            sum += i;
        }
        return 1.0*sum/estimateBaseOnCount.size();
//        return estimateBaseOnCount.stream().collect(Collectors.averagingInt(x -> x));
//        return estimateBaseOnCount.stream().mapToInt(x -> x).average().orElse(0.0);
    }
}
