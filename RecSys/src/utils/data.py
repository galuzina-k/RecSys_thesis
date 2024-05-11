import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


def load_MovieLens(data_folder):
    """Load users, movies, ratings Data Frames with properly encoded userId, movieId"""

    df_movies = pd.read_csv(
        data_folder + "movies.csv",
        encoding="iso-8859-1",
        sep=";",
        names=["movieId", "name", "genre"],
    )
    df_ratings = pd.read_csv(
        data_folder + "ratings.csv",
        encoding="iso-8859-1",
        sep=";",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    df_users = pd.read_csv(
        data_folder + "users.csv",
        encoding="iso-8859-1",
        sep=";",
        names=["userId", "gender", "age", "occupation", "zip-code"],
    )

    ## Encode usedId, movieId
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df_movies["movieId"] = movie_encoder.fit_transform(df_movies["movieId"])
    df_users["userId"] = user_encoder.fit_transform(df_users["userId"])

    df_ratings["movieId"] = movie_encoder.transform(df_ratings["movieId"])
    df_ratings["userId"] = user_encoder.transform(df_ratings["userId"])

    return df_users, df_movies, df_ratings


def create_test_user(
    df_users, df_ratings, watch_list, gender="F", age=20, occupation=5, zip_code=777_777
):
    new_user_id = df_users["userId"].max() + 1
    df_users.loc[-1] = [new_user_id, gender, age, occupation, zip_code]
    df_users = df_users.reset_index(drop=True)
    df_test_user = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    df_test_user = df_test_user.assign(movieId=watch_list).assign(
        userId=df_users["userId"].max(),
        rating=5,
        timestamp=lambda x: np.arange(x.shape[0]),
    )
    df_ratings = pd.concat([df_ratings, df_test_user], ignore_index=True)
    return df_users, df_ratings, new_user_id


def add_not_watched_movies(userId, df_test, df_train, df_movies):
    df_add = pd.DataFrame()
    movie = df_test.loc[df_test.userId == userId, "movieId"]
    watched_movies = np.append(
        movie, df_train.loc[df_train.userId == userId, "movieId"].values
    )
    not_wathed_movies = np.setdiff1d(
        np.arange(df_movies["movieId"].max() + 1), watched_movies
    )
    random_500 = np.random.choice(not_wathed_movies, 500, replace=False)

    df_temp = pd.DataFrame().assign(movieId=random_500, userId=userId, action=0)
    df_add = pd.concat([df_add, df_temp], ignore_index=True)
    return df_add


def create_test_user_display_df(df_test_user, df_movies, score_column, n=100):
    return (
        df_test_user.sort_values(by=score_column, ascending=False)
        .merge(df_movies, on="movieId")
        .loc[:n, ["userId", "movieId", "name", "genre", score_column]]
    )


def train_test_val_split(df_ratings, df_movies, random_state=777):
    """
    Leave the last user action as a test,
    pre-last action as val and all previous actions as test.
    Then enrich val, test with 100 negative samples for metrics evaluation.
    """

    df_ratings["rank"] = (
        df_ratings[["userId", "timestamp"]]
        .groupby("userId", as_index=False)["timestamp"]
        .rank(method="first", ascending=False)
    )
    # leave one out
    df_train = df_ratings.loc[df_ratings["rank"] != 1].reset_index(drop=True)
    df_test = (
        df_ratings.loc[df_ratings["rank"] == 1].reset_index(drop=True).assign(action=1)
    )
    df_test, df_val = train_test_split(
        df_test, test_size=0.2, random_state=random_state
    )

    # enrich test data with 100 random movies from the ones not intercated by user
    df_add = pd.DataFrame()
    for user in tqdm(df_test.userId.unique(), desc="Enriching test"):
        movie = df_test.loc[df_test.userId == user, "movieId"]
        watched_movies = np.append(
            movie, df_train.loc[df_train.userId == user, "movieId"].values
        )
        not_wathed_movies = np.setdiff1d(
            np.arange(df_movies["movieId"].max() + 1), watched_movies
        )
        random_100 = np.random.choice(not_wathed_movies, 100, replace=False)

        df_temp = pd.DataFrame().assign(movieId=random_100, userId=user, action=0)
        df_add = pd.concat([df_add, df_temp], ignore_index=True)

    df_test = pd.concat([df_test, df_add], ignore_index=True).drop(
        columns=["timestamp", "rating", "rank"]
    )

    # enrich val data with 100 random movies from the ones not intercated by user
    df_add = pd.DataFrame()
    for user in tqdm(df_val.userId.unique(), desc="Enriching val"):
        movie = df_val.loc[df_val.userId == user, "movieId"]
        watched_movies = np.append(
            movie, df_train.loc[df_train.userId == user, "movieId"].values
        )
        not_wathed_movies = np.setdiff1d(
            np.arange(df_movies["movieId"].max() + 1), watched_movies
        )
        random_100 = np.random.choice(not_wathed_movies, 100, replace=False)

        df_temp = pd.DataFrame().assign(movieId=random_100, userId=user, action=0)
        df_add = pd.concat([df_add, df_temp], ignore_index=True)

    df_val = pd.concat([df_val, df_add], ignore_index=True).drop(
        columns=["timestamp", "rating", "rank"]
    )

    return df_train, df_test, df_val
