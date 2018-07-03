package org.mraczynska.recommender.lenskit;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.LongIterator;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.transform.threshold.Threshold;
import org.grouplens.lenskit.vectors.ImmutableSparseVector;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonTypes;
import org.lenskit.data.ratings.RatingVectorPDAO;
import org.lenskit.knn.user.Neighbor;
import org.lenskit.knn.user.NeighborFinder;
import org.lenskit.knn.user.UserSimilarity;
import org.lenskit.knn.user.UserSimilarityThreshold;
import org.lenskit.transform.normalize.UserVectorNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import java.util.Collections;
import java.util.Iterator;

public class MyLiveNeighborFinder implements NeighborFinder {
    private static final Logger logger = LoggerFactory.getLogger(MyLiveNeighborFinder.class);

    private final UserSimilarity similarity;
    private final RatingVectorPDAO rvDAO;
    private final DataAccessObject dao;
    private final UserVectorNormalizer normalizer;
    private final Threshold threshold;

    /**
     * Construct a new user neighborhood finder.
     *
     * @param rvd    The user rating vector dAO.
     * @param dao    The data access object.
     * @param sim    The similarity function to use.
     * @param norm   The normalizer for user rating/preference vectors.
     * @param thresh The threshold for user similarities.
     */
    @Inject
    public MyLiveNeighborFinder(RatingVectorPDAO rvd, DataAccessObject dao,
                              UserSimilarity sim,
                              UserVectorNormalizer norm,
                              @UserSimilarityThreshold Threshold thresh) {
        similarity = sim;
        normalizer = norm;
        rvDAO = rvd;
        this.dao = dao;
        threshold = thresh;

        Preconditions.checkArgument(sim.isSparse(), "user similarity function is not sparse");
    }

    @Override
    public Iterable<Neighbor> getCandidateNeighbors(final long user, LongSet items) {
        Long2DoubleMap ratings = rvDAO.userRatingVector(user);
        if (ratings.isEmpty()) {
            return Collections.emptyList();
        }

        SparseVector urs = ImmutableSparseVector.create(ratings);
        final ImmutableSparseVector nratings = normalizer.normalize(user, urs, null)
                .freeze();
        final LongSet candidates = findCandidateNeighbors(user, nratings/*, items*/);
        logger.debug("found {} candidate neighbors for {}", candidates.size(), user);
        return new Iterable<Neighbor>() {
            @Override
            public Iterator<Neighbor> iterator() {
                return new MyLiveNeighborFinder.NeighborIterator(user, nratings, candidates);
            }
        };
    }

    /**
     * Get the IDs of the candidate neighbors for a user.
     * @param user The user.
     * @param uvec The user's normalized preference vector.
//     * @param itemSet The set of target items.
     * @return The set of IDs of candidate neighbors.
     */
    private LongSet findCandidateNeighbors(long user, SparseVector uvec) {
        LongSet users = new LongOpenHashSet(100);

        LongSet iusers = dao.getEntityIds(CommonTypes.USER);
            users.addAll(iusers);

        users.remove(user);

        return users;
    }

    /**
     * Check if a similarity is acceptable.
     *
     * @param sim The similarity to check.
     * @return {@code false} if the similarity is NaN, infinite, or rejected by the threshold;
     *         {@code true} otherwise.
     */
    private boolean acceptSimilarity(double sim) {
        return !Double.isNaN(sim) && !Double.isInfinite(sim) && threshold.retain(sim);
    }

    private MutableSparseVector getUserRatingVector(long user) {
        Long2DoubleMap ratings = rvDAO.userRatingVector(user);
        if (ratings.isEmpty()) {
            return null;
        } else {
            return MutableSparseVector.create(ratings);
        }
    }

    private class NeighborIterator extends AbstractIterator<Neighbor> {
        private final long user;
        private final SparseVector userVector;
        private final LongIterator neighborIter;

        public NeighborIterator(long uid, SparseVector uvec, LongSet nbrs) {
            user = uid;
            userVector = uvec;
            neighborIter = nbrs.iterator();
        }
        @Override
        protected Neighbor computeNext() {
            while (neighborIter.hasNext()) {
                final long neighbor = neighborIter.nextLong();
                MutableSparseVector nbrRatings = getUserRatingVector(neighbor);
                if (nbrRatings != null) {
                    ImmutableSparseVector rawRatings = nbrRatings.immutable();
                    normalizer.normalize(neighbor, rawRatings, nbrRatings);
                    final double sim = similarity.similarity(user, userVector, neighbor, nbrRatings);
                    if (acceptSimilarity(sim)) {
                        // we have found a neighbor
                        return new Neighbor(neighbor, rawRatings, sim);
                    }
                }
            }
            // no neighbor found, done
            return endOfData();
        }
    }
}
