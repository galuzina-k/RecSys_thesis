from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import AutoIntConfig
from pytorch_tabular import TabularModel


class AutoInt:
    def __init__(
        self,
        task="classification",
        target=["label"],
        continuous_cols=["age"],
        categorical_cols=["userId", "movieId", "gender", "occupation", "genre"],
        learning_rate=5e-4,
        head="LinearHead",
        accelerator="cpu",
        devices=10,
        auto_lr_find=False,
        batch_size=1024,
        max_epochs=10,
        early_stopping="valid_loss",  # Monitor valid_loss for early stopping
        early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
        early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
        checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
        load_best=True,
    ):

        data_config = DataConfig(
            target=target,  # target should always be a list.
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )

        trainer_config = TrainerConfig(
            accelerator=accelerator,
            devices=devices,
            auto_lr_find=auto_lr_find,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping=early_stopping,  # Monitor valid_loss for early stopping
            early_stopping_mode=early_stopping_mode,  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=early_stopping_patience,  # No. of epochs of degradation training will wait before terminating
            checkpoints=checkpoints,  # Save best checkpoint monitoring val_loss
            load_best=load_best,  # After training, load the best checkpoint
        )
        optimizer_config = OptimizerConfig()

        model_config = AutoIntConfig(
            task=task,
            learning_rate=learning_rate,
            head=head,  # Linear Head
        )

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

    def fit(self, train):
        self.tabular_model.fit(train=train)

    def predict(self, test):
        return self.tabular_model.predict(test)["1_probability"]
