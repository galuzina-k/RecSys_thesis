import torch
from tqdm.auto import tqdm
import pprint
from ..utils import split_test_df
from ..metrics import reccomendation_report


def blank_generator(iterable, **kwargs):
    for item in iterable:
        yield item


def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    n_epochs,
    device="cpu",
    val_loader=None,
    df_val=None,
    cos_dist=None,
    popularity=None,
    verbose=True,
):
    """Function to train custom deep learning reccomendation model

    Args:
        model : model to trian
        optimizer : optimizer
        scheduler : scheduler
        criterion : criterion
        verbose: True
    """
    model.to(device)
    num_iterations = len(train_loader)

    for epoch in tqdm(range(n_epochs), desc="Epochs", disable=not verbose):
        # train
        total_train_loss = 0
        model.train()
        with tqdm(train_loader, unit="batch", disable=not verbose) as tepoch:
            for data in tepoch:
                # data_on_device[:-1] - all features
                # data_on_device[-1] - target
                data_on_device = [i.to(device) for i in data]

                pred_train = model(*data_on_device[:-1])
                loss_train = criterion(pred_train.flatten(), data_on_device[-1])

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                total_train_loss += loss_train.item()

                if verbose:
                    tepoch.set_postfix(
                        loss=loss_train.item(), lr=round(scheduler.get_last_lr()[0], 7)
                    )
        scheduler.step()

        if verbose:
            print("Epoch:", epoch)
            print("Train loss:", round(total_train_loss / num_iterations, 5))

        # val
        if val_loader is not None:
            model.eval()
            total_preds = torch.zeros(len(val_loader.dataset)).to(device)
            batch_size = val_loader.batch_size
            for i, data in enumerate(
                tqdm(val_loader, desc="Inference", unit="batch", disable=not verbose)
            ):
                data_on_device = [i.to(device) for i in data]
                with torch.no_grad():
                    total_preds[i * batch_size : (i + 1) * batch_size] = model(
                        *data_on_device
                    ).flatten()

            df_val["pred"] = total_preds.numpy()
            pred, target, pred_items = split_test_df(
                df_val, "userId", "movieId", "pred", "action"
            )
            if verbose:
                pprint.pp(
                    reccomendation_report(
                        pred, target, pred_items, cos_dist, popularity, k=15
                    ),
                    width=1,
                )


def predict(model, test_loader, device="cpu", verbose=True):
    """Function to perfrom predictions on a custom deep learning reccomendation model"""

    model.to(device)
    model.eval()
    total_preds = torch.zeros(len(test_loader.dataset))
    batch_size = test_loader.batch_size

    for i, data in enumerate(
        tqdm(test_loader, desc="Inference", unit="batch", disable=not verbose)
    ):
        data_on_device = [i.to(device) for i in data]
        with torch.no_grad():
            total_preds[i * batch_size : (i + 1) * batch_size] = model(
                *data_on_device
            ).flatten()
    return total_preds
