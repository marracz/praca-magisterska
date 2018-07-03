package org.mraczynska.recommender.lenskit;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import it.unimi.dsi.fastutil.longs.Long2DoubleFunction;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.LongArrays;
import it.unimi.dsi.fastutil.longs.LongComparators;
import org.apache.commons.lang3.StringUtils;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.lenskit.api.Recommender;
import org.lenskit.api.ResultMap;
import org.lenskit.eval.traintest.AlgorithmInstance;
import org.lenskit.eval.traintest.DataSet;
import org.lenskit.eval.traintest.TestUser;
import org.lenskit.eval.traintest.metrics.Discount;
import org.lenskit.eval.traintest.metrics.Discounts;
import org.lenskit.eval.traintest.metrics.MetricResult;
import org.lenskit.eval.traintest.predict.PredictMetric;
import org.lenskit.util.collections.LongUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * Author: Marta Raczyńska
 */
public class DCGPredictMetric extends PredictMetric<MeanAccumulator> {
    private static final Logger logger = LoggerFactory.getLogger(DCGPredictMetric.class);
    public static final String DEFAULT_COLUMN = "Predict.DCG";
    private final String columnName;
    private final Discount discount;

    public DCGPredictMetric() {
        this(Discounts.log2(), "Predict.DCG");
    }

    public DCGPredictMetric(Discount disc) {
        this(disc, "Predict.DCG");
    }

    @JsonCreator
    public DCGPredictMetric(DCGPredictMetric.Spec spec) {
        this(spec.getParsedDiscount(), StringUtils.defaultString(spec.getColumnName(), "Predict.DCG"));
    }

    public DCGPredictMetric(Discount disc, String name) {
        super(Lists.newArrayList(new String[]{name, name + ".Raw"}), Lists.newArrayList(new String[]{name}));
        this.columnName = name;
        this.discount = disc;
    }

    @Nullable
    public MeanAccumulator createContext(AlgorithmInstance algorithm, DataSet dataSet, Recommender recommender) {
        return new MeanAccumulator();
    }

    @Nonnull
    public MetricResult getAggregateMeasurements(MeanAccumulator context) {
        return MetricResult.singleton(this.columnName, Double.valueOf(context.getMean()));
    }

    @Nonnull
    public MetricResult measureUser(TestUser user, ResultMap predictions, MeanAccumulator context) {
        if(predictions != null && !predictions.isEmpty()) {
            Long2DoubleMap ratings = user.getTestRatings();
            long[] actual = LongUtils.asLongSet(predictions.keySet()).toLongArray();
            LongArrays.quickSort(actual, LongComparators.oppositeComparator(LongUtils.keyValueComparator(LongUtils.asLong2DoubleFunction(predictions.scoreMap()))));
            double gain = this.computeDCG(actual, ratings);
            context.add(gain);
            ImmutableMap.Builder<String, Double> results = ImmutableMap.builder();
            return MetricResult.fromMap(results.put(this.columnName, Double.valueOf(gain)).put(this.columnName + ".Raw", Double.valueOf(gain)).build());
        } else {
            return MetricResult.empty();
        }
    }

    double computeDCG(long[] items, Long2DoubleFunction values) {
        double gain = 0.0D;
        int rank = 0;
        long[] var6 = items;
        int var7 = items.length;

        for(int var8 = 0; var8 < var7; ++var8) {
            long item = var6[var8];
            double v = values.get(item);
            ++rank;
            gain += v * this.discount.discount(rank);
        }

        return gain;
    }

    @JsonIgnoreProperties({"type"})
    public static class Spec {
        private String name;
        private String discount;

        public Spec() {
        }

        public String getColumnName() {
            return this.name;
        }

        public void setColumnName(String name) {
            this.name = name;
        }

        public String getDiscount() {
            return this.discount;
        }

        public void setDiscount(String discount) {
            this.discount = discount;
        }

        public Discount getParsedDiscount() {
            return (Discount)(this.discount == null?Discounts.log2():Discounts.parse(this.discount));
        }
    }
}
