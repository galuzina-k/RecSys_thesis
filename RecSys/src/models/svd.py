from surprise import SVDpp, Reader, Dataset

from ..utils import surprise_predict


class SVDPlusPlus:
    def __init__(self, random_state=777, vebose=True, cache_ratings=True):
        self.reader = Reader(rating_scale=(1, 5))
        self.svdpp = SVDpp(
            random_state=random_state, verbose=vebose, cache_ratings=cache_ratings
        )

    def fit(self, data):
        data = Dataset.load_from_df(
            data[["userId", "movieId", "rating"]], self.reader
        ).build_full_trainset()
        self.svdpp.fit(data)

    def predict(self, test):
        return surprise_predict(
            self.svdpp, test, usercol="userId", itemcol="movieId", predcol="rating"
        )["rating"].values
