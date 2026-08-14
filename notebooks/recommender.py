import numpy as np
import pandas as pd

from scipy.stats import pearsonr


class RecommendationEngine:
    """
    Base class for movie recommendation systems.
    """

    def __init__(self, movie_ratings, movie_catalog):
        # Encapsulation:
        # keep internal copies of the datasets
        self._ratings = movie_ratings.copy()
        self._movies = movie_catalog.copy()

    def validate_user(self, user_id):
        """
        Check whether a user exists.
        """

        if user_id not in self._ratings["userId"].values:
            raise ValueError(f"User {user_id} does not exist.")

    def get_user_history(self, user_id):
        """
        Return a user's rating history.
        """

        self.validate_user(user_id)

        return self._ratings[self._ratings["userId"] == user_id].sort_values(
            "rating", ascending=False
        )

    def get_watched_movies(self, user_id):
        """
        Return movie IDs already rated by the user.
        """

        history = self.get_user_history(user_id)

        return set(history["movieId"])

    def recommend(self, user_id, n=10):
        """
        Child classes must implement their own
        recommendation method.
        """

        raise NotImplementedError("Child classes must implement recommend().")


class ContentBasedRecommender(RecommendationEngine):
    def __init__(self, movie_ratings, movie_catalog):

        super().__init__(movie_ratings, movie_catalog)

    def get_genre_preferences(self, user_id):
        """
        Calculate the user's preference for each genre
        based on both average rating and number of movies rated.
        """

        self.validate_user(user_id)

        user_data = self._ratings[self._ratings["userId"] == user_id].copy()

        user_data["genre"] = user_data["genres"].str.split("|")

        user_data = user_data.explode("genre")

        preferences = user_data.groupby("genre").agg(
            average_rating=("rating", "mean"), movies_rated=("movieId", "nunique")
        )

        return preferences

    def get_director_preferences(self, user_id):
        """
        Calculate the user's average rating
        for directors they have watched.
        """

        self.validate_user(user_id)

        user_data = self._ratings[
            (self._ratings["userId"] == user_id) & (self._ratings["director"].notna())
        ].copy()

        preferences = (
            user_data.groupby("director")
            .agg(average_rating=("rating", "mean"), movies_rated=("movieId", "nunique"))
            .sort_values(["average_rating", "movies_rated"], ascending=False)
        )

        return preferences

    def _genre_score(self, genres, genre_preferences):
        """
        Calculate a preference score for a movie's genres.
        """

        if pd.isna(genres):
            return np.nan

        movie_genres = genres.split("|")

        matching_scores = [
            genre_preferences.loc[genre, "average_rating"]
            for genre in movie_genres
            if genre in genre_preferences.index
        ]

        if not matching_scores:
            return np.nan

        return np.mean(matching_scores)

    def _director_score(self, director, director_preferences):
        """
        Return the user's historical average rating
        for a movie's director.
        """

        if pd.isna(director):
            return np.nan

        movie_directors = [name.strip() for name in director.split("|")]

        matching_scores = [
            director_preferences.loc[name, "average_rating"]
            for name in movie_directors
            if name in director_preferences.index
        ]

        if not matching_scores:
            return np.nan

        return np.mean(matching_scores)

    def recommend(self, user_id, n=10, min_ratings=20):
        """
        Recommend unwatched movies using
        genre preferences, director preferences,
        and overall movie ratings.
        """

        self.validate_user(user_id)

        genre_preferences = self.get_genre_preferences(user_id)

        director_preferences = self.get_director_preferences(user_id)

        watched_movies = self.get_watched_movies(user_id)

        # Candidate movies:
        # user has not watched them
        # and they have enough ratings
        candidates = self._movies[
            (~self._movies["movieId"].isin(watched_movies))
            & (self._movies["rating_count"] >= min_ratings)
        ].copy()

        # -----------------------------------------
        # Calculate preference scores
        # -----------------------------------------

        candidates["genre_score"] = candidates["genres"].apply(
            lambda genres: self._genre_score(genres, genre_preferences)
        )

        candidates["director_score"] = candidates["director"].apply(
            lambda director: self._director_score(director, director_preferences)
        )

        # -----------------------------------------
        # Combine the available scores
        # -----------------------------------------

        def calculate_content_score(row):

            scores = []
            weights = []

            # Genre preference
            if pd.notna(row["genre_score"]):
                scores.append(row["genre_score"] * 0.40)

                weights.append(0.40)

            # Director preference
            if pd.notna(row["director_score"]):
                scores.append(row["director_score"] * 0.25)

                weights.append(0.25)

            # General movie quality
            if pd.notna(row["average_rating"]):
                scores.append(row["average_rating"] * 0.35)

                weights.append(0.35)

            if not weights:
                return np.nan

            # Divide by the available weights so
            # missing director data does not unfairly
            # penalize a movie.
            return sum(scores) / sum(weights)

        candidates["content_score"] = candidates.apply(calculate_content_score, axis=1)

        # -----------------------------------------
        # Rank recommendations
        # -----------------------------------------

        recommendations = (
            candidates.dropna(subset=["content_score"])
            .sort_values(["content_score", "rating_count"], ascending=[False, False])
            .head(n)
        )

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "director",
                "average_rating",
                "rating_count",
                "genre_score",
                "director_score",
                "content_score",
            ]
        ]


class CollaborativeRecommender(RecommendationEngine):
    def __init__(self, movie_ratings, movie_catalog):

        super().__init__(movie_ratings, movie_catalog)

        # Create user × movie rating matrix
        self._rating_matrix = self._ratings.pivot_table(
            index="userId", columns="movieId", values="rating", aggfunc="mean"
        )

    def find_similar_users(self, user_id, min_common=5):
        """
        Calculate Pearson correlation between
        the target user and every other user.

        Only users with at least min_common
        commonly rated movies are compared.
        """

        self.validate_user(user_id)

        target_ratings = self._rating_matrix.loc[user_id]

        similarity_results = []

        for other_user_id in self._rating_matrix.index:
            # Don't compare user with themselves
            if other_user_id == user_id:
                continue

            other_ratings = self._rating_matrix.loc[other_user_id]

            # Movies rated by both users
            common_mask = target_ratings.notna() & other_ratings.notna()

            common_count = common_mask.sum()

            # Not enough overlap
            if common_count < min_common:
                continue

            target_common = target_ratings[common_mask].astype(float).values

            other_common = other_ratings[common_mask].astype(float).values

            # Pearson correlation cannot be
            # calculated if either set has
            # no variation
            if np.std(target_common) == 0 or np.std(other_common) == 0:
                continue

            correlation, p_value = pearsonr(target_common, other_common)

            if np.isnan(correlation):
                continue

            similarity_results.append(
                {
                    "userId": other_user_id,
                    "similarity": correlation,
                    "common_movies": common_count,
                    "p_value": p_value,
                }
            )

        # Convert similarity results to DataFrame
        similarities = pd.DataFrame(similarity_results)

        if similarities.empty:
            return similarities

        # Sort by strongest correlation first
        similarities = similarities.sort_values(
            ["similarity", "common_movies"], ascending=[False, False]
        ).reset_index(drop=True)

        return similarities

    def recommend(
        self,
        user_id,
        n=10,
        min_common=5,
        n_similar_users=20,
        min_similarity=0.10,
        min_neighbor_ratings=3,
    ):
        """
        Recommend movies using ratings from
        positively correlated users.
        """

        self.validate_user(user_id)

        # --------------------------------------
        # 1. Find similar users
        # --------------------------------------

        similar_users = self.find_similar_users(user_id=user_id, min_common=min_common)

        if similar_users.empty:
            return pd.DataFrame()

        # Only positively correlated users
        similar_users = (
            similar_users[similar_users["similarity"] >= min_similarity]
            .head(n_similar_users)
            .copy()
        )

        if similar_users.empty:
            return pd.DataFrame()

        # --------------------------------------
        # 2. Movies target user already watched
        # --------------------------------------

        watched_movies = self.get_watched_movies(user_id)

        # --------------------------------------
        # 3. Ratings made by similar users
        # --------------------------------------

        neighbor_ratings = self._ratings[
            self._ratings["userId"].isin(similar_users["userId"])
        ].copy()

        # Remove movies target user already rated
        neighbor_ratings = neighbor_ratings[
            ~neighbor_ratings["movieId"].isin(watched_movies)
        ]

        # --------------------------------------
        # 4. Attach Pearson similarity
        # --------------------------------------

        neighbor_ratings = neighbor_ratings.merge(
            similar_users[["userId", "similarity"]], on="userId", how="inner"
        )

        # --------------------------------------
        # 5. Weight ratings by similarity
        # --------------------------------------

        neighbor_ratings["weighted_rating"] = (
            neighbor_ratings["rating"] * neighbor_ratings["similarity"]
        )

        # --------------------------------------
        # 6. Aggregate each movie
        # --------------------------------------

        movie_scores = neighbor_ratings.groupby("movieId", as_index=False).agg(
            weighted_rating_sum=("weighted_rating", "sum"),
            similarity_sum=("similarity", "sum"),
            similar_user_ratings=("rating", "count"),
            max_similarity=("similarity", "max"),
        )

        # --------------------------------------
        # 7. Collaborative score
        # --------------------------------------

        movie_scores["collaborative_score"] = (
            movie_scores["weighted_rating_sum"] / movie_scores["similarity_sum"]
        )

        # Require support from multiple neighbors
        movie_scores = movie_scores[
            movie_scores["similar_user_ratings"] >= min_neighbor_ratings
        ]

        # --------------------------------------
        # 8. Add movie information
        # --------------------------------------

        recommendations = movie_scores.merge(self._movies, on="movieId", how="left")

        # --------------------------------------
        # 9. Rank recommendations
        # --------------------------------------

        recommendations = recommendations.sort_values(
            ["collaborative_score", "similar_user_ratings", "rating_count"],
            ascending=[False, False, False],
        ).head(n)

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "director",
                "collaborative_score",
                "similar_user_ratings",
                "max_similarity",
                "average_rating",
                "rating_count",
            ]
        ]


class ClusterRecommender(RecommendationEngine):
    def __init__(self, movie_ratings, movie_catalog, user_clusters):

        super().__init__(movie_ratings, movie_catalog)

        self._user_clusters = user_clusters.copy()

    def get_user_cluster(self, user_id):
        """
        Return the cluster assigned to a user.
        """

        self.validate_user(user_id)

        cluster = self._user_clusters.loc[
            self._user_clusters["userId"] == user_id, "cluster"
        ].iloc[0]

        return cluster

    def get_cluster_users(self, user_id):
        """
        Return users belonging to the same
        cluster as the target user.
        """

        cluster = self.get_user_cluster(user_id)

        cluster_users = self._user_clusters.loc[
            self._user_clusters["cluster"] == cluster, "userId"
        ].tolist()

        # Remove target user
        cluster_users = [
            cluster_user for cluster_user in cluster_users if cluster_user != user_id
        ]

        return cluster_users

    def recommend(self, user_id, n=10, min_cluster_ratings=5):
        """
        Recommend highly rated movies among
        users in the same cluster.
        """

        self.validate_user(user_id)

        cluster = self.get_user_cluster(user_id)

        cluster_users = self.get_cluster_users(user_id)

        watched_movies = self.get_watched_movies(user_id)

        # Ratings from users in same cluster
        cluster_ratings = self._ratings[
            self._ratings["userId"].isin(cluster_users)
        ].copy()

        # Exclude movies already rated
        # by target user
        cluster_ratings = cluster_ratings[
            ~cluster_ratings["movieId"].isin(watched_movies)
        ]

        # Calculate movie performance
        # inside the user's cluster
        cluster_scores = cluster_ratings.groupby("movieId", as_index=False).agg(
            cluster_score=("rating", "mean"), cluster_rating_count=("rating", "count")
        )

        # Require enough support from
        # users in the cluster
        cluster_scores = cluster_scores[
            cluster_scores["cluster_rating_count"] >= min_cluster_ratings
        ]

        # Add movie information
        recommendations = cluster_scores.merge(self._movies, on="movieId", how="left")

        # Rank movies
        recommendations = recommendations.sort_values(
            ["cluster_score", "cluster_rating_count", "rating_count"],
            ascending=[False, False, False],
        ).head(n)

        recommendations["user_cluster"] = cluster

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "director",
                "user_cluster",
                "cluster_score",
                "cluster_rating_count",
                "average_rating",
                "rating_count",
            ]
        ]


class HybridRecommender(RecommendationEngine):
    def __init__(
        self,
        movie_ratings,
        movie_catalog,
        content_engine,
        collaborative_engine,
        cluster_engine,
    ):

        super().__init__(movie_ratings, movie_catalog)

        self._content_engine = content_engine

        self._collaborative_engine = collaborative_engine

        self._cluster_engine = cluster_engine

    def _percentile_score(self, dataframe, score_column, new_column):
        """
        Convert model scores into percentile
        ranks between 0 and 1.
        """

        dataframe = dataframe.copy()

        dataframe[new_column] = dataframe[score_column].rank(pct=True, method="average")

        return dataframe

    def recommend(
        self,
        user_id,
        n=10,
        candidate_pool=100,
        content_weight=0.40,
        collaborative_weight=0.40,
        cluster_weight=0.20,
    ):
        """
        Combine content-based, collaborative,
        and cluster-based recommendations.
        """

        self.validate_user(user_id)

        # --------------------------------------
        # 1. Validate weights
        # --------------------------------------

        total_weight = content_weight + collaborative_weight + cluster_weight

        if not np.isclose(total_weight, 1.0):
            raise ValueError("Hybrid weights must add up to 1.")

        # --------------------------------------
        # 2. Generate candidate recommendations
        # --------------------------------------

        content_results = self._content_engine.recommend(
            user_id=user_id, n=candidate_pool, min_ratings=20
        )

        collaborative_results = self._collaborative_engine.recommend(
            user_id=user_id,
            n=candidate_pool,
            min_common=10,
            n_similar_users=30,
            min_similarity=0.10,
            min_neighbor_ratings=3,
        )

        cluster_results = self._cluster_engine.recommend(
            user_id=user_id, n=candidate_pool, min_cluster_ratings=10
        )

        # --------------------------------------
        # 3. Convert scores to percentile ranks
        # --------------------------------------

        content_results = self._percentile_score(
            content_results, "content_score", "content_normalized"
        )

        collaborative_results = self._percentile_score(
            collaborative_results, "collaborative_score", "collaborative_normalized"
        )

        cluster_results = self._percentile_score(
            cluster_results, "cluster_score", "cluster_normalized"
        )

        # --------------------------------------
        # 4. Keep only required columns
        # --------------------------------------

        content_scores = content_results[
            ["movieId", "content_score", "content_normalized"]
        ]

        collaborative_scores = collaborative_results[
            ["movieId", "collaborative_score", "collaborative_normalized"]
        ]

        cluster_scores = cluster_results[
            ["movieId", "cluster_score", "cluster_normalized"]
        ]

        # --------------------------------------
        # 5. Merge all three recommendation sets
        # --------------------------------------

        hybrid = content_scores.merge(
            collaborative_scores, on="movieId", how="outer"
        ).merge(cluster_scores, on="movieId", how="outer")

        # --------------------------------------
        # 6. Record which models recommended it
        # --------------------------------------

        hybrid["content_recommended"] = hybrid["content_score"].notna()

        hybrid["collaborative_recommended"] = hybrid["collaborative_score"].notna()

        hybrid["cluster_recommended"] = hybrid["cluster_score"].notna()

        hybrid["models_recommending"] = hybrid[
            ["content_recommended", "collaborative_recommended", "cluster_recommended"]
        ].sum(axis=1)

        # --------------------------------------
        # 7. Missing model scores become zero
        # --------------------------------------

        normalized_columns = [
            "content_normalized",
            "collaborative_normalized",
            "cluster_normalized",
        ]

        hybrid[normalized_columns] = hybrid[normalized_columns].fillna(0)

        # --------------------------------------
        # 8. Calculate hybrid score
        # --------------------------------------

        hybrid["hybrid_score"] = (
            hybrid["content_normalized"] * content_weight
            + hybrid["collaborative_normalized"] * collaborative_weight
            + hybrid["cluster_normalized"] * cluster_weight
        )

        # --------------------------------------
        # 9. Add movie information
        # --------------------------------------

        hybrid = hybrid.merge(self._movies, on="movieId", how="left")

        # --------------------------------------
        # 10. Rank final recommendations
        # --------------------------------------

        hybrid = hybrid.sort_values(
            ["hybrid_score", "models_recommending", "average_rating", "rating_count"],
            ascending=[False, False, False, False],
        ).head(n)

        return hybrid[
            [
                "movieId",
                "title",
                "genres",
                "director",
                "hybrid_score",
                "models_recommending",
                "content_score",
                "collaborative_score",
                "cluster_score",
                "average_rating",
                "rating_count",
            ]
        ]


class RecommendationExplainer:
    def __init__(self, movie_ratings):

        self._ratings = movie_ratings.copy()

    def _get_user_history(self, user_id):
        """
        Return movies rated by the user.
        """

        return self._ratings[self._ratings["userId"] == user_id].copy()

    def _find_similar_liked_movie(self, user_id, recommended_genres):
        """
        Find a highly rated movie from the
        user's history that shares a genre
        with the recommendation.
        """

        history = self._get_user_history(user_id)

        # Focus on movies the user liked
        liked_movies = history[history["rating"] >= 4.0].copy()

        if liked_movies.empty:
            return None

        recommendation_genres = set(recommended_genres.split("|"))

        def genre_similarity(genres):

            movie_genres = set(genres.split("|"))

            intersection = recommendation_genres & movie_genres

            union = recommendation_genres | movie_genres

            if not union:
                return 0

            return len(intersection) / len(union)

        liked_movies["genre_similarity"] = liked_movies["genres"].apply(
            genre_similarity
        )

        matching_movies = liked_movies[
            liked_movies["genre_similarity"] > 0
        ].sort_values(["genre_similarity", "rating"], ascending=[False, False])

        if matching_movies.empty:
            return None

        best_match = matching_movies.iloc[0]

        if best_match["genre_similarity"] < 0.25:
            return None

        return best_match

    def generate_explanation(self, user_id, recommendation):
        """
        Generate a personalized explanation
        for one recommendation.
        """

        title = recommendation["title"]

        genres = recommendation["genres"]

        models = int(recommendation["models_recommending"])

        similar_movie = self._find_similar_liked_movie(user_id, genres)

        explanation_parts = []

        # ----------------------------------
        # Personal history explanation
        # ----------------------------------

        if similar_movie is not None:
            shared_genres = set(genres.split("|")) & set(
                similar_movie["genres"].split("|")
            )

            shared_genres_text = " and ".join(sorted(shared_genres))

            explanation_parts.append(
                f"Since you rated "
                f"{similar_movie['title']} "
                f"{similar_movie['rating']:.1f}/5 "
                f"and it shares "
                f"{shared_genres_text} elements, "
                f"you might enjoy {title}."
            )

        else:
            explanation_parts.append(
                f"{title} matches patterns in your movie preferences."
            )

        # ----------------------------------
        # Recommendation-model explanation
        # ----------------------------------

        if models == 3:
            explanation_parts.append(
                "It is supported by all three "
                "recommendation models: your content "
                "preferences, similar users, and your "
                "k-means user cluster."
            )

        elif models == 2:
            explanation_parts.append(
                "It is independently supported by two of the recommendation models."
            )

        else:
            explanation_parts.append(
                "It receives a particularly strong score from one recommendation model."
            )

        return " ".join(explanation_parts)
