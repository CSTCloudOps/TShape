from typing import Dict
import numpy as np
from EasyTSAD.Controller import TSADController
from EasyTSAD.Exptools import EarlyStoppingTorch
from EasyTSAD.DataFactory import TSData
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory.TorchDataSet import PredictWindow

from typing import Dict
import torchinfo
import tqdm

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F

from layyer import TShape_model 

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    datasets = ["AIOPS"]
    dataset_types = "UTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="./datasets",
        datasets=datasets,
    )
    
      
    
    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import TSData

    class TShape(BaseMethod):
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            
            self.cuda = True
            if self.cuda == True and torch.cuda.is_available():
                self.device = torch.device("cuda:3")
                print("=== Using CUDA ===")
            else:
                if self.cuda == True and not torch.cuda.is_available():
                    print("=== CUDA is unavailable ===")
                self.device = torch.device("cpu")
                print("=== Using CPU ===")
                
            self.p = params["p"]
            self.batch_size = params["batch_size"]
            self.model = TShape_model(self.p).to(self.device)
            self.epochs = params["epochs"]
            learning_rate = params["lr"]
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
            self.loss = nn.MSELoss()
            
            self.save_path = None
            self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=8)
        
        def train_valid_phase(self, tsTrain: TSData):
            
            train_loader = DataLoader(
                dataset=PredictWindow.UTSOneByOneDataset(tsTrain, "train", window_size=self.p),
                batch_size=self.batch_size,
                shuffle=True
            )
            
            valid_loader = DataLoader(
                dataset=PredictWindow.UTSOneByOneDataset(tsTrain, "valid", window_size=self.p),
                batch_size=self.batch_size,
                shuffle=False
            )
            
            for epoch in range(1, self.epochs + 1):
                self.model.train(mode=True)
                avg_loss = 0
                loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    output = self.model(x)
                    loss = self.loss(output, target)
                    loss.backward()

                    self.optimizer.step()
                    
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
                
                self.model.eval()
                avg_loss = 0
                loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
                with torch.no_grad():
                    for idx, (x, target) in loop:
                        x, target = x.to(self.device), target.to(self.device)
                        output = self.model(x)
                        loss = self.loss(output, target)
                        avg_loss += loss.cpu().item()
                        loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                        loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
                valid_loss = avg_loss/max(len(valid_loader), 1)
                self.scheduler.step()
                
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                    break
                
        def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            train_loader = DataLoader(
                dataset=PredictWindow.UTSAllInOneDataset(tsTrains, "train", window_size=self.p),
                batch_size=self.batch_size,
                shuffle=True
            )
            
            valid_loader = DataLoader(
                dataset=PredictWindow.UTSAllInOneDataset(tsTrains, "valid", window_size=self.p),
                batch_size=self.batch_size,
                shuffle=False
            )
            
            for epoch in range(1, self.epochs + 1):
                self.model.train(mode=True)
                avg_loss = 0
                loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    output = self.model(x)
                    loss = self.loss(output, target)
                    loss.backward()

                    self.optimizer.step()
                    
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
                
                self.model.eval()
                avg_loss = 0
                loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
                with torch.no_grad():
                    for idx, (x, target) in loop:
                        x, target = x.to(self.device), target.to(self.device)
                        output = self.model(x)
                        loss = self.loss(output, target)
                        avg_loss += loss.cpu().item()
                        loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                        loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
                valid_loss = avg_loss/max(len(valid_loader), 1)
                self.scheduler.step()
                
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                    break
            
        def test_phase(self, tsData: TSData):
            test_loader = DataLoader(
                dataset=PredictWindow.UTSOneByOneDataset(tsData, "test", window_size=self.p),
                batch_size=self.batch_size,
                shuffle=False
            )
            
            self.model.eval()
            scores = []
            loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    # loss = self.loss(output, target)
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    loop.set_description(f'Testing: ')

            scores = torch.cat(scores, dim=0)
            scores = scores.numpy().flatten()

            assert scores.ndim == 1
            self.__anomaly_score = scores
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            model_stats = torchinfo.summary(self.model, (self.batch_size, self.p), verbose=0)
            with open(save_file, 'w') as f:
                f.write(str(model_stats))

    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas
    
    training_schema = "naive"
    method = "TShape"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path="/home/ch/ts-dual/TSShapeFormer/Examples/run_your_algo/config.toml" # path/to/your
    )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
        ]
    )

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
