import mmh3
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class trainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings["userId"], ratings["movieId"]))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return (
            torch.tensor(users),
            torch.tensor(items),
            torch.tensor(labels, dtype=torch.float),
        )


class trainDatasetWithCrossFeatures(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
        cross_features_mapping: (dict): Dict of a cross features corresponding to a movie

    """

    def __init__(
        self,
        ratings,
        all_movieIds,
        user_features_mapping,
        item_features_mapping,
        hash_bucket_size=20,
    ):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

        # constuct cross features with the restriction of maximum `hash_bucket_size` categories
        self.cross_features = []
        for i in tqdm(range(self.users.shape[0])):
            crossed_category = (
                user_features_mapping[self.users[i].item()]
                + item_features_mapping[self.items[i].item()]
            )
            category_idx = mmh3.hash(crossed_category) % hash_bucket_size

            feature = [0 for _ in range(hash_bucket_size)]
            feature[category_idx] = 1
            self.cross_features.append(feature)

        self.cross_features = torch.tensor(self.cross_features).to(torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.cross_features[idx],
            self.labels[idx],
        )

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings["userId"], ratings["movieId"]))

        num_negatives = 4
        for u, i in tqdm(user_item_set):
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return (
            torch.tensor(users),
            torch.tensor(items),
            torch.tensor(labels, dtype=torch.float),
        )


class trainDatasetWithNumCatFeatures(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
        user_features_cat, user_features_num, item_features_cat: torch.tensor with i-th element corresponding to i-th user/item

    """

    def __init__(
        self,
        ratings,
        all_movieIds,
        user_features_cat,
        user_features_num,
        item_features_cat,
    ):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

        self.cat_feats = torch.hstack(
            (user_features_cat[self.users], item_features_cat[self.items])
        ).to(torch.long)
        self.num_feats = user_features_num[self.users].clone().detach().to(torch.float)
        # # constuct cross features with the restriction of maximum `hash_bucket_size` categories
        # self.cross_features = []
        # for i in tqdm(range(self.users.shape[0])):
        #     crossed_category = (
        #         user_features_mapping[self.users[i].item()]
        #         + item_features_mapping[self.items[i].item()]
        #     )
        #     category_idx = mmh3.hash(crossed_category) % hash_bucket_size

        #     feature = [0 for _ in range(hash_bucket_size)]
        #     feature[category_idx] = 1
        #     self.cross_features.append(feature)

        # self.cross_features = torch.tensor(self.cross_features).to(torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.num_feats[idx],
            self.cat_feats[idx],
            self.labels[idx],
        )

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings["userId"], ratings["movieId"]))

        num_negatives = 4
        for u, i in tqdm(user_item_set):
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return (
            torch.tensor(users),
            torch.tensor(items),
            torch.tensor(labels, dtype=torch.float),
        )
