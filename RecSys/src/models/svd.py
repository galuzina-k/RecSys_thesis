from surprise import SVDpp

from ..utils import surprise_predict


class SVDPlusPlus:
    def __init__(self, random_state=777, vebose=True, cache_ratings=True):
        self.svdpp = SVDpp(
            random_state=random_state, verbose=vebose, cache_ratings=cache_ratings
        )

    def fit(self, data):
        self.svdpp.fit(data)

    def predict(self, test):
        return surprise_predict(
            self.svdpp, test, usercol="userId", itemcol="movieId", predcol="rating"
        )["rating"].values
