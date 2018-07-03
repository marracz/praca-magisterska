package org.mraczynska.recommender.lenskit;

/**
 * Author: Marta Raczyńska
 */
public enum Task {
    EVALUATION(true, false, false),
    RECOMMENDATION(false, true, false),
    F1(false, false, true),
    ALL(true, true, true);

    private final boolean evaluation;
    private final boolean recommendation;
    private final boolean f1;

    Task(boolean evaluation, boolean recommendation, boolean f1) {
        this.evaluation = evaluation;
        this.recommendation = recommendation;
        this.f1 = f1;
    }

    public boolean isEvaluation() {
        return evaluation;
    }

    public boolean isRecommendation() {
        return recommendation;
    }

    public boolean isF1() {
        return f1;
    }
}
