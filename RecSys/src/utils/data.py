import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


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
