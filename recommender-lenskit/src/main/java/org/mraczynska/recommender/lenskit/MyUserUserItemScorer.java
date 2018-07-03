package org.mraczynska.recommender.lenskit;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import it.unimi.dsi.fastutil.longs.*;
import org.grouplens.lenskit.transform.threshold.Threshold;
import org.grouplens.lenskit.vectors.ImmutableSparseVector;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.data.ratings.RatingVectorPDAO;
import org.lenskit.knn.MinNeighbors;
import org.lenskit.knn.NeighborhoodSize;
import org.lenskit.knn.user.Neighbor;
import org.lenskit.knn.user.NeighborFinder;
import org.lenskit.knn.user.UserSimilarityThreshold;
import org.lenskit.results.AbstractResult;
import org.lenskit.results.Results;
import org.lenskit.transform.normalize.UserVectorNormalizer;
import org.lenskit.transform.normalize.VectorTransformation;
import org.lenskit.util.collections.LongUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.PriorityQueue;

import static java.lang.Math.abs;

public class MyUserUserItemScorer extends AbstractItemScorer {

    private static final Logger logger = LoggerFactory.getLogger(MyUserUserItemScorer.class);

    private final RatingVectorPDAO dao;
    protected final NeighborFinder neighborFinder;
    protected final UserVectorNormalizer normalizer;
    private final int neighborhoodSize;
    private final int minNeighborCount;
    private final Threshold userThreshold;

    @Inject
    public MyUserUserItemScorer(RatingVectorPDAO rvd, NeighborFinder nf,
                              UserVectorNormalizer norm,
                              @NeighborhoodSize int nnbrs,
                              @MinNeighbors int minNbrs,
                              @UserSimilarityThreshold Threshold thresh) {
        this.dao = rvd;
        neighborFinder = nf;
        normalizer = norm;
        neighborhoodSize = nnbrs;
        minNeighborCount = minNbrs;
        userThreshold = thresh;
    }

    /**
     * Normalize all neighbor rating vectors, taking care to normalize each one
     * only once.
     *
     * FIXME: MDE does not like this method.
     *
     * @param neighborhoods
     */
    protected Long2ObjectMap<SparseVector> normalizeNeighborRatings(Collection<? extends Collection<Neighbor>> neighborhoods) {
        Long2ObjectMap<SparseVector> normedVectors =
                new Long2ObjectOpenHashMap<>();
        for (Neighbor n : Iterables.concat(neighborhoods)) {
            if (!normedVectors.containsKey(n.user)) {
                normedVectors.put(n.user, normalizer.normalize(n.user, n.vector, null));
            }
        }
        return normedVectors;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        Long2DoubleMap history = dao.userRatingVector(user);

        logger.debug("Predicting for {} items for user {} with {} events",
                items.size(), user, history.size());

        LongSortedSet itemSet = LongUtils.packedSet(items);
        Long2ObjectMap<? extends Collection<Neighbor>> neighborhoods =
                findNeighbors(user, itemSet);
        Long2ObjectMap<SparseVector> normedUsers =
                normalizeNeighborRatings(neighborhoods.values());

        // Make the normalizing transform to reverse
        SparseVector urv = ImmutableSparseVector.create(history);
        VectorTransformation vo = normalizer.makeTransformation(user, urv);

        // And prepare results
        List<ResultBuilder> resultBuilders = new ArrayList<>();
        LongIterator iter = itemSet.iterator();
        while (iter.hasNext()) {
            final long item = iter.nextLong();
            double sum = 0;
            double weight = 0;
            int count = 0;
            Collection<Neighbor> nbrs = neighborhoods.get(item);
            if (nbrs != null) {
                for (Neighbor n : nbrs) {
                    SparseVector vectorEntries = normedUsers.get(n.user);
                    if(vectorEntries.containsKey(item)){
                        weight += abs(n.similarity);
                        sum += n.similarity * vectorEntries.get(item);
                        count += 1;
                    }
                }
            }

            if (count >= minNeighborCount && weight > 0) {
                if (logger.isTraceEnabled()) {
                    logger.trace("Total neighbor weight for item {} is {} from {} neighbors",
                            item, weight, count);
                }
                resultBuilders.add(UserUserResult.newBuilder()
                        .setItemId(item)
                        .setRawScore(sum / weight)
                        .setNeighborhoodSize(count)
                        .setTotalWeight(weight));
            }
        }

        // de-normalize the results
        MutableSparseVector vec = MutableSparseVector.create(itemSet);
        for (ResultBuilder rb: resultBuilders) {
            vec.set(rb.getItemId(), rb.getRawScore());
        }
        vo.unapply(vec);

        List<Result> results = new ArrayList<>(resultBuilders.size());
        for (ResultBuilder rb: resultBuilders) {
            results.add(rb.setScore(vec.get(rb.getItemId()))
                    .build());
        }

        return Results.newResultMap(results);
    }

//    /**
//     * Find the neighbors for a user with respect to a collection of items.
//     * For each item, the <var>neighborhoodSize</var> users closest to the
//     * provided user are returned.
//     *
//     * @param user  The user's rating vector.
//     * @param items The items for which neighborhoods are requested.
//     * @return A mapping of item IDs to neighborhoods.
//     */
    /*protected Long2ObjectMap<? extends Collection<Neighbor>>
    findNeighbors(long user, @Nonnull LongSet items) {
        Preconditions.checkNotNull(user, "user profile");
        Preconditions.checkNotNull(user, "item set");

        Long2ObjectMap<PriorityQueue<Neighbor>> heaps = new Long2ObjectOpenHashMap<>(items.size());
        for (LongIterator iter = items.iterator(); iter.hasNext();) {
            long item = iter.nextLong();
            heaps.put(item, new PriorityQueue<>(neighborhoodSize + 1,
                    Neighbor.SIMILARITY_COMPARATOR));
        }

        int neighborsUsed = 0;
        for (Neighbor nbr: neighborFinder.getCandidateNeighbors(user, items)) {
            for (VectorEntry e: nbr.vector) {
                final long item = e.getKey();
                PriorityQueue<Neighbor> heap = heaps.get(item);
                if (heap != null) {
                    heap.add(nbr);
                    if (heap.size() > neighborhoodSize) {
                        assert heap.size() == neighborhoodSize + 1;
                        heap.remove();
                    } else {
                        neighborsUsed += 1;
                    }
                }
            }
        }
        logger.debug("using {} neighbors across {} items",
                neighborsUsed, items.size());
        return heaps;
    }*/


//    @Override
    protected Long2ObjectMap<? extends Collection<Neighbor>> findNeighbors(long user, @Nonnull LongSet items) {
        Preconditions.checkNotNull(user, "user profile");
        Preconditions.checkNotNull(user, "item set");

        //TODO: sprawdź czy wygląda podobnie jak w mahoucie

        PriorityQueue<Neighbor> heap = new PriorityQueue<>(neighborhoodSize + 1, Neighbor.SIMILARITY_COMPARATOR);


        for (Neighbor nbr: neighborFinder.getCandidateNeighbors(user, items)) {
        //TODO weź wszystkich

            heap.add(nbr);
            if (heap.size() > neighborhoodSize) {
                assert heap.size() == neighborhoodSize + 1;
                heap.remove();
            }
        }

        Long2ObjectMap<PriorityQueue<Neighbor>> heaps = new Long2ObjectOpenHashMap<>(items.size());
        for (LongIterator iter = items.iterator(); iter.hasNext();) {
            long item = iter.nextLong();
            heaps.put(item, heap);
        }
        return heaps;
    }

    private static class ResultBuilder {
            private long itemId;
            private double rawScore;
            private double score;
            private int neighborhoodSize;
            private double totalWeight;

            public long getItemId() {
                return itemId;
            }

            public ResultBuilder setItemId(long itemId) {
                this.itemId = itemId;
                return this;
            }

            public double getRawScore() {
                return rawScore;
            }

            public ResultBuilder setRawScore(double rawScore) {
                this.rawScore = rawScore;
                return this;
            }

            public double getScore() {
                return score;
            }

            public ResultBuilder setScore(double score) {
                this.score = score;
                return this;
            }

            public int getNeighborhoodSize() {
                return neighborhoodSize;
            }

            public ResultBuilder setNeighborhoodSize(int neighborhoodSize) {
                this.neighborhoodSize = neighborhoodSize;
                return this;
            }

            public double getTotalWeight() {
                return totalWeight;
            }

            public ResultBuilder setTotalWeight(double totalWeight) {
                this.totalWeight = totalWeight;
                return this;
            }

            public UserUserResult build() {
                return new UserUserResult(itemId, score, neighborhoodSize, totalWeight);
            }
        }

    private static class UserUserResult extends AbstractResult {
        private final int neighborhoodSize;
        private final double neighborWeight;

        UserUserResult(long item, double score, int nnbrs, double weight) {
            super(item, score);
            neighborhoodSize = nnbrs;
            neighborWeight = weight;
        }

        static ResultBuilder newBuilder() {
            return new ResultBuilder();
        }

        /**
         * Get the neighborhood size for this result.
         * @return The number of neighbors used to compute the result.
         */
        public int getNeighborhoodSize() {
            return neighborhoodSize;
        }

        /**
         * Get the total neighbor weight for this result.
         * @return The total weight (similarity) of the neighbors.
         */
        public double getTotalNeighborWeight() {
            return neighborWeight;
        }

        @Override
        public int hashCode() {
            return startHashCode().append(neighborhoodSize)
                    .append(neighborWeight)
                    .toHashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof UserUserResult) {
               UserUserResult or = (UserUserResult) obj;
                return startEquality(or).append(neighborhoodSize, or.neighborhoodSize)
                        .append(neighborWeight, or.neighborWeight)
                        .isEquals();
            } else {
                return false;
            }
        }
    }

}
