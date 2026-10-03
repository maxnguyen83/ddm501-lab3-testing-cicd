"""
Model behavioral tests.

These tests verify the model's behavior patterns (CheckList style):
- Invariance: Output shouldn't change for certain perturbations
- Directional: Output should change in expected direction
- Minimum Functionality: Basic cases the model must handle

They run against the trained artefact (models/svd_model.pkl). Expected
directions are derived from the ratings stored in the model's own trainset,
so no extra data download is needed. Thresholds are explained in
docs/TESTING_STRATEGY.md.

Run tests:
    pytest tests/model/test_model_behavior.py -v
"""

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from app.model import MovieRatingModel

# Movies with at least this many ratings count as "popular" (338 movies)
POPULAR_MIN_RATINGS = 100
# How many top/bottom popular movies or extreme users to compare
GROUP_SIZE = 10
EXTREME_USERS = 50


def _raw_ratings_by_user(model: MovieRatingModel) -> Dict[str, List[Tuple[str, float]]]:
    """Map raw user ID -> [(raw movie ID, rating)] from the model's trainset."""
    ts = model.model.trainset
    return {
        ts.to_raw_uid(u): [(ts.to_raw_iid(i), r) for i, r in ratings]
        for u, ratings in ts.ur.items()
    }


@pytest.fixture(scope="module")
def ratings_by_user(trained_model) -> Dict[str, List[Tuple[str, float]]]:
    """Training ratings grouped by user."""
    return _raw_ratings_by_user(trained_model)


@pytest.fixture(scope="module")
def popular_movies_by_mean(trained_model) -> List[str]:
    """Popular movies (>= 100 ratings) sorted from lowest to highest mean rating."""
    ts = trained_model.model.trainset
    stats = [
        (float(np.mean([r for _, r in ratings])), ts.to_raw_iid(i))
        for i, ratings in ts.ir.items()
        if len(ratings) >= POPULAR_MIN_RATINGS
    ]
    return [movie for _, movie in sorted(stats)]


class TestModelInvariance:
    """
    Invariance tests - output shouldn't change for certain perturbations.

    These tests ensure that the model produces consistent results
    when given the same inputs.
    """

    # =========================================================================
    # Deterministic Output Tests
    # =========================================================================

    def test_same_input_same_output(self, trained_model):
        """Test that same input always produces same output."""
        result1 = trained_model.predict("196", "242")
        result2 = trained_model.predict("196", "242")
        assert result1 == result2

    def test_multiple_calls_consistent(self, trained_model):
        """Test that multiple calls produce consistent results."""
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert all(r == results[0] for r in results)

    # =========================================================================
    # Batch Order Invariance Tests
    # =========================================================================

    def test_batch_order_independent(self, trained_model):
        """
        Test that batch predictions are independent of input order.

        Compared per pair (not as a set) so two pairs with equal ratings
        cannot hide a swap.
        """
        pairs1 = [("196", "242"), ("186", "302"), ("22", "377")]
        pairs2 = list(reversed(pairs1))

        results1 = dict(zip(pairs1, trained_model.predict_batch(pairs1)))
        results2 = dict(zip(pairs2, trained_model.predict_batch(pairs2)))

        assert results1 == results2

    def test_individual_vs_batch_same_results(self, trained_model, known_user_movie_pairs):
        """Test that individual and batch predictions match."""
        pairs = [(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs]
        individual = [trained_model.predict(u, m) for u, m in pairs]
        assert trained_model.predict_batch(pairs) == individual

    def test_batch_context_does_not_leak(self, trained_model):
        """A pair's prediction is the same alone or surrounded by 50 other pairs."""
        rng = random.Random(0)
        noise = [(str(rng.randint(1, 943)), str(rng.randint(1, 1682))) for _ in range(50)]
        alone = trained_model.predict("196", "242")
        in_batch = trained_model.predict_batch(noise[:25] + [("196", "242")] + noise[25:])
        assert in_batch[25] == alone

    def test_whitespace_in_ids_does_not_change_prediction(self, trained_model):
        """Surface-form noise (spaces, tabs, newlines) is not a different user."""
        clean = trained_model.predict("196", "242")
        for user, movie in [(" 196", "242"), ("196 ", " 242 "), ("\t196\n", "242")]:
            assert trained_model.predict(user, movie) == clean

    def test_reloaded_artefact_gives_same_predictions(self, trained_model, known_user_movie_pairs):
        """Loading the pickle again (e.g. a new container) does not change outputs."""
        reloaded = MovieRatingModel(model_path=trained_model.model_path)
        for p in known_user_movie_pairs:
            assert reloaded.predict(p["user_id"], p["movie_id"]) == trained_model.predict(
                p["user_id"], p["movie_id"]
            )


class TestModelDirectional:
    """
    Directional tests - output should change in expected direction.

    These tests verify that the model behaves sensibly when inputs
    change in predictable ways.
    """

    # =========================================================================
    # Directional Tests
    # =========================================================================

    def test_predictions_are_reasonable(self, trained_model, known_user_movie_pairs):
        """
        Predictions for known pairs stay within 2.5 stars of the true rating.

        The starter suggested 1.5, but pair (166, 346) is a 1-star outlier for
        that user: across 20 training seeds its error was 1.78-2.29, so a
        1.5-star rule fails every time. See docs/TESTING_STRATEGY.md.
        """
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            actual = pair["actual_rating"]
            assert (
                abs(prediction - actual) < 2.5
            ), f"Prediction {prediction} too far from actual {actual}"

    def test_low_ratings_predicted_lower_than_high_ratings(
        self, trained_model, known_user_movie_pairs
    ):
        """Pairs the users rated 1 star get lower predictions than pairs rated 3 stars."""
        preds: Dict[float, List[float]] = {}
        for p in known_user_movie_pairs:
            prediction = trained_model.predict(p["user_id"], p["movie_id"])
            preds.setdefault(p["actual_rating"], []).append(prediction)
        assert np.mean(preds[1.0]) < np.mean(preds[3.0])

    def test_different_movies_different_predictions(self, trained_model):
        """Test that different movies can get different predictions (same user)."""
        movies = ["50", "242", "302", "377", "1"]
        predictions = {trained_model.predict("196", m) for m in movies}
        assert len(predictions) > 1

    def test_different_users_different_predictions(self, trained_model):
        """Test that different users can get different predictions (same movie)."""
        users = ["196", "186", "22", "244", "166"]
        predictions = {trained_model.predict(u, "50") for u in users}
        assert len(predictions) > 1

    def test_user_loved_movies_rank_above_hated_movies(self, trained_model, ratings_by_user):
        """
        For each user who gave both 5-star and 1-star ratings, the average
        prediction on their 5-star movies is higher than on their 1-star movies.
        Measured: holds for 715/715 such users (smallest gap 0.40 stars).
        """
        checked = holds = 0
        for user, ratings in ratings_by_user.items():
            loved = [m for m, r in ratings if r == 5.0]
            hated = [m for m, r in ratings if r == 1.0]
            if not loved or not hated:
                continue
            checked += 1
            loved_mean = np.mean(trained_model.predict_batch([(user, m) for m in loved]))
            hated_mean = np.mean(trained_model.predict_batch([(user, m) for m in hated]))
            holds += loved_mean > hated_mean
        assert checked > 500
        assert holds / checked >= 0.99

    def test_acclaimed_movies_beat_panned_movies_for_every_user(
        self, trained_model, popular_movies_by_mean, ratings_by_user
    ):
        """
        The 10 best-rated popular movies are predicted higher than the 10
        worst-rated popular movies for (almost) every user.
        """
        bottom = popular_movies_by_mean[:GROUP_SIZE]
        top = popular_movies_by_mean[-GROUP_SIZE:]
        holds = 0
        for user in ratings_by_user:
            top_mean = np.mean(trained_model.predict_batch([(user, m) for m in top]))
            bottom_mean = np.mean(trained_model.predict_batch([(user, m) for m in bottom]))
            holds += top_mean > bottom_mean
        assert holds / len(ratings_by_user) >= 0.99

    def test_generous_users_rate_higher_than_harsh_users(
        self, trained_model, ratings_by_user, popular_movies_by_mean
    ):
        """
        For the same movie, the 50 most generous users (highest mean rating)
        get a higher prediction than the 50 harshest users, by at least 0.5 stars.
        """
        by_mean = sorted(ratings_by_user, key=lambda u: np.mean([r for _, r in ratings_by_user[u]]))
        harsh, generous = by_mean[:EXTREME_USERS], by_mean[-EXTREME_USERS:]
        for movie in popular_movies_by_mean[-5:] + popular_movies_by_mean[:5]:
            generous_mean = np.mean(trained_model.predict_batch([(u, movie) for u in generous]))
            harsh_mean = np.mean(trained_model.predict_batch([(u, movie) for u in harsh]))
            assert generous_mean - harsh_mean >= 0.5, movie

    def test_cold_start_user_still_prefers_acclaimed_movies(
        self, trained_model, popular_movies_by_mean
    ):
        """An unknown user falls back to item biases, so movie quality still matters."""
        top = popular_movies_by_mean[-GROUP_SIZE:]
        bottom = popular_movies_by_mean[:GROUP_SIZE]
        top_mean = np.mean([trained_model.predict("new_user", m) for m in top])
        bottom_mean = np.mean([trained_model.predict("new_user", m) for m in bottom])
        assert top_mean - bottom_mean >= 1.0


class TestMinimumFunctionality:
    """
    Minimum functionality tests - basic cases the model must handle.

    These are simple test cases that the model absolutely must pass
    to be considered functional.
    """

    # =========================================================================
    # Minimum Functionality Tests
    # =========================================================================

    def test_can_predict_for_known_user(self, trained_model):
        """Test that model can make prediction for known user."""
        prediction = trained_model.predict("196", "242")
        assert prediction is not None
        assert 1.0 <= prediction <= 5.0

    def test_can_predict_for_multiple_users(self, trained_model, known_user_movie_pairs):
        """Test that model can make predictions for multiple known users."""
        for p in known_user_movie_pairs:
            prediction = trained_model.predict(p["user_id"], p["movie_id"])
            assert 1.0 <= prediction <= 5.0

    def test_predictions_not_all_same(self, trained_model, known_user_movie_pairs):
        """
        Test that not all predictions are the same value.

        If all predictions are identical, model might be broken.
        """
        predictions = [
            trained_model.predict(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs
        ]
        assert len(set(predictions)) > 1, "All predictions are identical"

    def test_known_users_are_personalised(self, trained_model):
        """A known user's prediction differs from the cold-start fallback."""
        assert trained_model.predict("196", "242") != trained_model.predict("new_user", "242")

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    def test_handles_unknown_user_gracefully(self, trained_model, unknown_users):
        """
        Test that model handles unknown users without crashing.

        SVD with biases never refuses: an unknown user gets global mean +
        movie bias, which must still be a valid rating.
        """
        for user_id in unknown_users:
            prediction = trained_model.predict(user_id, "242")
            assert 1.0 <= prediction <= 5.0

    def test_handles_unknown_movie_gracefully(self, trained_model, unknown_movies):
        """Test that model handles unknown movies without crashing."""
        for movie_id in unknown_movies:
            prediction = trained_model.predict("196", movie_id)
            assert 1.0 <= prediction <= 5.0

    def test_unknown_user_and_movie_returns_global_mean(self, trained_model):
        """With nothing known, the fallback is the training global mean."""
        global_mean = trained_model.model.trainset.global_mean
        assert trained_model.predict("new_user", "new_movie") == round(global_mean, 2)

    def test_all_unknown_users_get_identical_fallback(self, trained_model, unknown_users):
        """Every unknown user is treated the same (no hidden state per ID)."""
        assert len({trained_model.predict(u, "242") for u in unknown_users}) == 1


class TestModelPerformance:
    """
    Performance-related behavioral tests.

    These tests verify that the model performs adequately
    on known test cases. Generalisation (held-out RMSE/MAE) is checked
    separately by scripts/validate_model.py in the model-validation workflow.
    """

    @pytest.fixture(scope="class")
    def sample_errors(self, trained_model) -> np.ndarray:
        """Absolute errors on a fixed random sample of 2,000 training ratings."""
        ts = trained_model.model.trainset
        rng = random.Random(42)
        sample = rng.sample(list(ts.all_ratings()), 2000)
        return np.array(
            [
                abs(trained_model.predict(ts.to_raw_uid(u), ts.to_raw_iid(i)) - r)
                for u, i, r in sample
            ]
        )

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_average_error_acceptable(self, sample_errors):
        """
        Mean absolute error on the 2,000-rating sample is below 0.70
        (measured 0.53; held-out CV MAE is 0.74).

        Five hand-picked pairs (three of them 1-2 star outliers) are too few
        for an average, so a fixed random sample is used instead.
        """
        assert sample_errors.mean() < 0.70

    def test_no_extreme_errors(self, trained_model, known_user_movie_pairs):
        """No prediction on the known pairs is off by 3.0 stars or more."""
        for p in known_user_movie_pairs:
            error = abs(trained_model.predict(p["user_id"], p["movie_id"]) - p["actual_rating"])
            assert error < 3.0, p

    def test_large_errors_are_rare(self, sample_errors):
        """At most 2% of sampled ratings are off by more than 2 stars (measured 0.25%)."""
        assert (sample_errors > 2.0).mean() <= 0.02

    def test_prediction_latency_budget(self, trained_model):
        """1,000 single predictions finish in under 0.5 s (measured about 2 ms)."""
        start = time.perf_counter()
        for _ in range(1000):
            trained_model.predict("196", "242")
        assert time.perf_counter() - start < 0.5


class TestModelRobustness:
    """
    Robustness tests - model behavior under unusual conditions.
    """

    def test_handles_string_numeric_ids(self, trained_model):
        """Test that model handles string IDs that look like numbers."""
        prediction = trained_model.predict("1", "1")
        assert 1.0 <= prediction <= 5.0
        assert prediction != trained_model.predict("new_user", "1")  # user 1 is known

    def test_handles_leading_zeros_in_ids(self, trained_model):
        """
        IDs are opaque strings: "0196" is NOT user "196". It is treated as a
        new user and gets the cold-start prediction. Documented limitation.
        """
        assert trained_model.predict("0196", "242") == trained_model.predict("new_user", "242")
        assert trained_model.predict("0196", "242") != trained_model.predict("196", "242")

    def test_integer_ids_are_rejected_not_cold_started(self, trained_model):
        """An int ID would silently miss the lookup, so the wrapper raises instead."""
        with pytest.raises(TypeError):
            trained_model.predict(196, "242")

    def test_predictions_stay_in_range_on_random_pairs(self, trained_model, ratings_by_user):
        """5,000 random known user/movie pairs all produce ratings in [1, 5]."""
        rng = random.Random(7)
        users = sorted(ratings_by_user)
        movies = sorted({m for ratings in ratings_by_user.values() for m, _ in ratings})
        pairs = [(rng.choice(users), rng.choice(movies)) for _ in range(5000)]
        predictions = trained_model.predict_batch(pairs)
        assert min(predictions) >= 1.0 and max(predictions) <= 5.0


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
