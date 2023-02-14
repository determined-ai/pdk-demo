import os
import logging
from typing import Any, Dict, Sequence, Tuple, Union, cast, List

from data import *

import torch
from torch import nn
from torch import optim
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from determined import InvalidHP

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class ChurnTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        # Initialize the trial class and wrap the models, optimizers, and LR schedulers.
        
        # Store trial context for later use.
        self.context = context
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        
        load_weights = (os.environ.get('SERVING_MODE') != 'true')
        logging.info(f"Loading weights : {load_weights}")

        if load_weights:
            
            self.files = self.download_data()
            
            if len(self.files) == 0:
                print("No data. Aborting training.")
                raise InvalidHP("No data")

        # Initialize the model and wrap it using self.context.wrap_model().
        self.model = nn.Sequential(
                                    nn.Linear(139, self.context.get_hparam("dense1")),
                                    nn.Linear(self.context.get_hparam("dense1"), 1),
                                    nn.Sigmoid()
                                )
        
        self.model = self.context.wrap_model(self.model)

        # Initialize the optimizer and wrap it using self.context.wrap_optimizer().
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.context.get_hparam("lr"))
        self.optimizer = self.context.wrap_optimizer(self.optimizer)
        
        self.loss_function = nn.BCELoss()


    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        # Run forward passes on the models and backward passes on the optimizers.
        
        X, y = batch
        
        # Define the training forward pass and calculate loss.
        output = self.model(X)
        loss = self.loss_function(output, y)
        
        # Define the training backward pass and step the optimizer.
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        
        # Compute accuracy
        output[output < 0.5] = 0.0
        output[output >= 0.5] = 1.0
        acc = torch.sum(output == y) / len(y)
        
        return {"loss": loss, "acc": acc}

    def evaluate_batch(self, batch: TorchData):
        # Define how to evaluate the model by calculating loss and other metrics
        # for a batch of validation data.
        X, y = batch
        
        output = self.model(X)
        val_loss = self.loss_function(output, y)
        
        output[output < 0.5] = 0.0
        output[output >= 0.5] = 1.0
        val_acc = torch.sum(output == y) / len(y)
        
        return {"val_loss": val_loss, "val_acc": val_acc}

    def build_training_data_loader(self):
        # Create the training data loader.
        # This should return a determined.pytorch.Dataset.
        
        train_dataset, _ = get_train_and_validation_datasets(self.files,
                                                            test_size=self.context.get_hparam("test_size"),
                                                            random_seed=self.context.get_hparam("random_seed"))
        
        return DataLoader(train_dataset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self):
        # Create the validation data loader.
        # This should return a determined.pytorch.Dataset.
        
        _, val_dataset = get_train_and_validation_datasets(self.files,
                                                            test_size=self.context.get_hparam("test_size"),
                                                            random_seed=self.context.get_hparam("random_seed"))
        
        return DataLoader(val_dataset, batch_size=self.context.get_per_slot_batch_size())
    
    def download_data(self):
        data_config = self.context.get_data_config()
        data_dir = os.path.join(self.download_directory, 'data')

        files = download_pach_repo(
            data_config['pachyderm']['host'],
            data_config['pachyderm']['port'],
            data_config["pachyderm"]["repo"],
            data_config["pachyderm"]["branch"],
            data_dir
        )
        print(f'Data dir set to : {data_dir}')

        return [des for src, des in files]
