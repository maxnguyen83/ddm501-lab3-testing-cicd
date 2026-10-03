"""
Data quality validation tests.

Three layers:
- Rule checks on the small ``sample_ratings`` fixture (fast, always run).
- The same rules plus schema, volume and distribution checks on the real
  MovieLens 100K ratings that the model is trained on.
- Negative tests: corrupted copies of the data must be flagged, which proves
  the checks can actually fail.

Run tests:
    pytest tests/data/test_data_quality.py -v
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import pytest
from surprise import Dataset

REQUIRED_FIELDS = ["user_id", "movie_id", "rating"]
MOVIELENS_STARS = {1.0, 2.0, 3.0, 4.0, 5.0}

# Facts published in the MovieLens 100K README
ML100K_RATINGS = 100_000
ML100K_USERS = 943
ML100K_MOVIES = 1_682
ML100K_MIN_RATINGS_PER_USER = 20


def find_quality_issues(
    records: List[Dict[str, Any]], allowed_ratings: Optional[Set[float]] = None
) -> List[str]:
    """
    Check rating records against the data contract and list every problem found.

    Args:
        records: Rating records with user_id, movie_id and rating keys
        allowed_ratings: If given, ratings must be one of these values

    Returns:
        Human-readable issues; an empty list means the data passed
    """
    issues: List[str] = []
    seen = set()
    for idx, record in enumerate(records):
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            issues.append(f"row {idx}: missing fields {missing}")
            continue
        for field in ("user_id", "movie_id"):
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                issues.append(f"row {idx}: invalid {field} {value!r}")
        rating = record["rating"]
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            issues.append(f"row {idx}: non-numeric rating {rating!r}")
        elif math.isnan(rating):
            issues.append(f"row {idx}: rating is NaN")
        elif not 1.0 <= rating <= 5.0:
            issues.append(f"row {idx}: rating {rating} outside [1, 5]")
        elif allowed_ratings is not None and float(rating) not in allowed_ratings:
            issues.append(f"row {idx}: rating {rating} not in {sorted(allowed_ratings)}")
        key = (record["user_id"], record["movie_id"])
        if key in seen:
            issues.append(f"row {idx}: duplicate pair {key}")
        seen.add(key)
    return issues


def distribution_issues(ratings: Sequence[float]) -> List[str]:
    """
    Check summary statistics of a batch of ratings.

    Bands: mean in [2.0, 4.5] (a 1-5 scale centred far from that is suspect),
    0 < std < 2.0 (no variation means no signal) and at least two distinct values.
    """
    issues: List[str] = []
    mean, std = float(np.mean(ratings)), float(np.std(ratings))
    if not 2.0 <= mean <= 4.5:
        issues.append(f"mean {mean:.2f} outside [2.0, 4.5]")
    if not 0.0 < std < 2.0:
        issues.append(f"std {std:.2f} outside (0, 2)")
    if len(set(ratings)) < 2:
        issues.append("only one distinct rating value")
    return issues


@pytest.fixture(scope="module")
def movielens_df() -> pd.DataFrame:
    """
    Raw MovieLens 100K ratings, parsed by the same Surprise loader the
    training script uses (downloaded on first use, cached in ~/.surprise_data).
    """
    data = Dataset.load_builtin("ml-100k", prompt=False)
    return pd.DataFrame(data.raw_ratings, columns=["user_id", "movie_id", "rating", "timestamp"])


@pytest.fixture(scope="module")
def movielens_records(movielens_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Real ratings as plain records for the rule checker."""
    return movielens_df[REQUIRED_FIELDS].to_dict("records")


class TestRatingDataQuality:
    """Tests for rating data quality."""

    # =========================================================================
    # Rating Range Tests
    # =========================================================================

    def test_all_ratings_in_valid_range(self, sample_ratings):
        """Test that all ratings are between 1.0 and 5.0."""
        for record in sample_ratings:
            assert 1.0 <= record["rating"] <= 5.0

    def test_no_negative_ratings(self, sample_ratings):
        """Test that there are no negative ratings."""
        assert all(record["rating"] >= 0 for record in sample_ratings)

    def test_no_ratings_above_maximum(self, sample_ratings):
        """Test that no ratings exceed 5.0."""
        assert max(record["rating"] for record in sample_ratings) <= 5.0

    # =========================================================================
    # ID Validation Tests
    # =========================================================================

    def test_no_missing_user_ids(self, sample_ratings):
        """Test that no user_ids are missing (None or empty)."""
        for record in sample_ratings:
            assert record["user_id"] is not None
            assert record["user_id"] != ""

    def test_no_missing_movie_ids(self, sample_ratings):
        """Test that no movie_ids are missing."""
        for record in sample_ratings:
            assert record["movie_id"] is not None
            assert record["movie_id"] != ""

    def test_user_ids_are_strings(self, sample_ratings):
        """Test that all user_ids are strings."""
        assert all(isinstance(record["user_id"], str) for record in sample_ratings)

    def test_movie_ids_are_strings(self, sample_ratings):
        """Test that all movie_ids are strings."""
        assert all(isinstance(record["movie_id"], str) for record in sample_ratings)

    # =========================================================================
    # Data Completeness Tests
    # =========================================================================

    def test_no_null_ratings(self, sample_ratings):
        """Test that no ratings are None/null."""
        assert all(record["rating"] is not None for record in sample_ratings)

    def test_all_records_have_required_fields(self, sample_ratings):
        """Test that all records have user_id, movie_id, and rating."""
        for record in sample_ratings:
            for field in REQUIRED_FIELDS:
                assert field in record

    def test_sample_passes_full_rule_check(self, sample_ratings):
        """The combined rule checker finds nothing wrong with the sample."""
        assert find_quality_issues(sample_ratings) == []


class TestRatingDistribution:
    """Tests for rating distribution statistics."""

    def test_mean_rating_reasonable(self, sample_ratings):
        """Test that mean rating is within reasonable range (2.0 - 4.5)."""
        ratings = [r["rating"] for r in sample_ratings]
        mean_rating = np.mean(ratings)
        assert 2.0 <= mean_rating <= 4.5

    def test_rating_standard_deviation(self, sample_ratings):
        """
        Test that rating standard deviation is reasonable.

        - STD > 0: some variation (constant ratings carry no signal)
        - STD < 2.0: not too much variation for a 1-5 scale
        """
        std = float(np.std([r["rating"] for r in sample_ratings]))
        assert 0.0 < std < 2.0

    def test_multiple_rating_values_exist(self, sample_ratings):
        """Test that there are multiple distinct rating values."""
        ratings = [r["rating"] for r in sample_ratings]
        unique_ratings = set(ratings)
        assert len(unique_ratings) > 1

    def test_sample_passes_distribution_check(self, sample_ratings):
        """The combined distribution checker accepts the sample."""
        assert distribution_issues([r["rating"] for r in sample_ratings]) == []


class TestDataUniqueness:
    """Tests for data uniqueness constraints."""

    def test_unique_user_movie_combinations(self, sample_ratings):
        """Test that each (user_id, movie_id) pair is unique."""
        pairs = [(r["user_id"], r["movie_id"]) for r in sample_ratings]
        assert len(pairs) == len(set(pairs))

    def test_multiple_users_exist(self, sample_ratings):
        """Test that there are multiple users in the dataset."""
        assert len({r["user_id"] for r in sample_ratings}) > 1

    def test_multiple_movies_exist(self, sample_ratings):
        """Test that there are multiple movies in the dataset."""
        assert len({r["movie_id"] for r in sample_ratings}) > 1


class TestDataTypes:
    """Tests for correct data types."""

    def test_ratings_are_numeric(self, sample_ratings):
        """Test that all ratings are numeric (int or float, not bool)."""
        for record in sample_ratings:
            rating = record["rating"]
            assert isinstance(rating, (int, float)) and not isinstance(rating, bool)

    def test_ratings_are_float_or_int(self, sample_ratings):
        """Test that ratings are float or can be converted to float."""
        for record in sample_ratings:
            assert not math.isnan(float(record["rating"]))


class TestRuleCheckerCatchesBadData:
    """Negative tests: each kind of corruption must be reported."""

    @pytest.mark.parametrize(
        "bad_record, expected_issue",
        [
            ({"user_id": "9", "movie_id": "1", "rating": 6.0}, "outside [1, 5]"),
            ({"user_id": "9", "movie_id": "1", "rating": 0.0}, "outside [1, 5]"),
            ({"user_id": "9", "movie_id": "1", "rating": -1}, "outside [1, 5]"),
            ({"user_id": "9", "movie_id": "1", "rating": None}, "non-numeric"),
            ({"user_id": "9", "movie_id": "1", "rating": "4"}, "non-numeric"),
            ({"user_id": "9", "movie_id": "1", "rating": True}, "non-numeric"),
            ({"user_id": "9", "movie_id": "1", "rating": float("nan")}, "NaN"),
            ({"user_id": "", "movie_id": "1", "rating": 3.0}, "invalid user_id"),
            ({"user_id": "  ", "movie_id": "1", "rating": 3.0}, "invalid user_id"),
            ({"user_id": 9, "movie_id": "1", "rating": 3.0}, "invalid user_id"),
            ({"user_id": "9", "movie_id": None, "rating": 3.0}, "invalid movie_id"),
            ({"user_id": "9", "rating": 3.0}, "missing fields ['movie_id']"),
        ],
    )
    def test_corrupted_record_is_flagged(self, sample_ratings, bad_record, expected_issue):
        """Appending one corrupted record makes the checker report it."""
        issues = find_quality_issues(sample_ratings + [bad_record])
        assert len(issues) == 1
        assert expected_issue in issues[0]

    def test_duplicate_pair_is_flagged(self, sample_ratings):
        """A repeated (user, movie) pair would double-count a rating in training."""
        issues = find_quality_issues(sample_ratings + [dict(sample_ratings[0])])
        assert len(issues) == 1 and "duplicate pair" in issues[0]

    def test_half_star_flagged_when_only_whole_stars_allowed(self, sample_ratings):
        """MovieLens 100K uses whole stars; 3.5 must be flagged under that contract."""
        issues = find_quality_issues(sample_ratings, allowed_ratings=MOVIELENS_STARS)
        assert len(issues) == 2  # the sample contains 3.5 and 4.5

    def test_constant_ratings_fail_distribution_check(self):
        """A pipeline bug that writes one value everywhere is caught twice over."""
        issues = distribution_issues([4.0] * 50)
        assert any("std" in i for i in issues)
        assert any("distinct" in i for i in issues)

    def test_skewed_ratings_fail_distribution_check(self):
        """A batch that is almost all 1-star (e.g. a scale bug) fails the mean band."""
        issues = distribution_issues([1.0] * 20 + [2.0])
        assert len(issues) == 1 and "mean" in issues[0]


class TestMovieLensDataset:
    """Data contract checks on the real training data (MovieLens 100K)."""

    # Schema and volume ---------------------------------------------------

    def test_schema(self, movielens_df):
        """Four columns; IDs and timestamps as strings, ratings as floats."""
        assert list(movielens_df.columns) == ["user_id", "movie_id", "rating", "timestamp"]
        for column in ("user_id", "movie_id", "timestamp"):
            assert movielens_df[column].map(type).eq(str).all(), column
        assert movielens_df["rating"].dtype == np.float64

    def test_volume_matches_published_counts(self, movielens_df):
        """A truncated or partial download shows up as wrong counts."""
        assert len(movielens_df) == ML100K_RATINGS
        assert movielens_df["user_id"].nunique() == ML100K_USERS
        assert movielens_df["movie_id"].nunique() == ML100K_MOVIES

    def test_ids_are_positive_integer_strings(self, movielens_df):
        """IDs are digit strings forming the contiguous ranges 1..943 and 1..1682."""
        for column, n in (("user_id", ML100K_USERS), ("movie_id", ML100K_MOVIES)):
            ids = movielens_df[column]
            assert ids.str.fullmatch(r"[1-9][0-9]*").all(), column
            assert set(ids.astype(int)) == set(range(1, n + 1)), column

    # Completeness and validity -------------------------------------------

    def test_no_missing_values(self, movielens_df):
        """No nulls anywhere."""
        assert not movielens_df.isna().any().any()

    def test_passes_full_rule_check_with_whole_stars(self, movielens_records):
        """Every real record passes the same rules, with whole stars only."""
        issues = find_quality_issues(movielens_records, allowed_ratings=MOVIELENS_STARS)
        assert issues == [], issues[:5]

    def test_no_duplicate_user_movie_pairs(self, movielens_df):
        """One rating per (user, movie)."""
        assert not movielens_df.duplicated(subset=["user_id", "movie_id"]).any()

    def test_every_user_has_minimum_history(self, movielens_df):
        """
        Each user rated at least 20 movies (dataset guarantee). The SVD user
        factors depend on this; very short histories would be cold-start.
        """
        assert movielens_df.groupby("user_id").size().min() >= ML100K_MIN_RATINGS_PER_USER

    def test_timestamps_inside_collection_window(self, movielens_df):
        """All ratings were collected between Sep 1997 and Apr 1998."""
        ts = movielens_df["timestamp"].astype(int)
        start = datetime(1997, 9, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(1998, 5, 1, tzinfo=timezone.utc).timestamp()
        assert ts.min() >= start and ts.max() < end

    # Distribution --------------------------------------------------------

    def test_mean_and_std_within_expected_band(self, movielens_df):
        """Measured mean 3.53 and std 1.13; bands catch shifted or squashed data."""
        assert 3.0 <= movielens_df["rating"].mean() <= 4.0
        assert 0.9 <= movielens_df["rating"].std() <= 1.3

    def test_no_star_value_dominates_or_is_missing(self, movielens_df):
        """
        Every star value appears (min share 6.1% for 1 star) and none takes
        over (max share 34.2% for 4 stars).
        """
        shares = movielens_df["rating"].value_counts(normalize=True)
        assert set(shares.index) == MOVIELENS_STARS
        assert shares.min() >= 0.03
        assert shares.max() <= 0.50

    def test_matrix_is_sparse_but_not_empty(self, movielens_df):
        """Density is about 6.3%; collaborative filtering assumes a sparse matrix."""
        density = len(movielens_df) / (ML100K_USERS * ML100K_MOVIES)
        assert 0.01 <= density <= 0.10

    # Consistency with fixtures and the trained model ---------------------

    def test_known_pairs_fixture_matches_dataset(self, movielens_df, known_user_movie_pairs):
        """The 'actual_rating' values used by model tests really are in the data."""
        lookup = movielens_df.set_index(["user_id", "movie_id"])["rating"]
        for pair in known_user_movie_pairs:
            assert lookup[(pair["user_id"], pair["movie_id"])] == pair["actual_rating"]

    def test_unknown_ids_fixtures_are_really_unknown(
        self, movielens_df, unknown_users, unknown_movies
    ):
        """Cold-start fixtures must not accidentally exist in the data."""
        assert not set(unknown_users) & set(movielens_df["user_id"])
        assert not set(unknown_movies) & set(movielens_df["movie_id"])

    def test_trained_model_saw_all_validated_data(self, movielens_df, trained_model):
        """The served model was trained on exactly this validated dataset."""
        trainset = trained_model.model.trainset
        assert trainset.n_ratings == len(movielens_df)
        assert trainset.n_users == movielens_df["user_id"].nunique()
        assert trainset.n_items == movielens_df["movie_id"].nunique()


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
