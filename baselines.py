import torch
import torch.nn as nn

import numpy as np
import time
from tqdm import tqdm

from comparisons import DLinear, Autoformer, PatchTST, TimesNet, Informer, NLinear, SegRNN, Reformer
from data_loader import load_datasets, _make_windowing_and_loader

from sklearn.metrics import mean_absolute_error, mean_squared_error
 
class Prediction(object):
    DEFAULTS = {}

    def __init__(self, opts):
        
        self.__dict__.update(Prediction.DEFAULTS, **opts)

        self.opts = opts
        self.target_dim = (self.node_feature_dim * self.num_node) + ( self.num_node * self.num_node )
        self.seq_len = self.seq_day * self.cycle
        self.pred_len = self.pred_day * self.cycle
        self.label_len = self.pred_day * self.cycle
        self.use_amp = False
        self.features = 'M'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_models()
        
        self.criterion = nn.MSELoss()

        self.patience = 5
        self.best_loss = float('inf')
        self.counter = 0
        
    def build_models(self):
        model_dict = {
            'Autoformer': Autoformer,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'Informer': Informer,
            'NLinear': NLinear,
            'SegRNN': SegRNN,
            'Reformer': Reformer,
        }
        self.model = model_dict[self.model_name].Model(self.opts)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
            self.model.to(self.device)
            if self.multi_gpu:
                self.model = nn.DataParallel(self.model)
        
    def proceed(self):
        print("======================================================")
        print("======================TRAIN MODE======================")
        print("======================================================")
        
        time_now = time.time()
        _, _, complex_3d, _ = load_datasets(self.dataset)
        train_loader, test_loader, val_loader, _ = _make_windowing_and_loader(self.dataset, self.model_name, self.batch_size, complex_3d, self.seq_day, self.pred_day, self.test_ratio, self.train_ratio, self.cycle)

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            train_loss = 0.0
            if self.model_name in ['Autoformer', 'TimesNet', 'Informer', 'Reformer']:
                for inputs, targets, input_mark, target_mark in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    input_mark, target_mark = input_mark.float().to(self.device), target_mark.float().to(self.device)
                    self.optimizer.zero_grad()

                    # decoder input
                    dec_inp = torch.zeros_like(targets[:, -self.pred_len:, :]).float()
                    dec_inp = torch.cat([targets[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    dec_inp_mark = torch.zeros_like(target_mark[:, -self.pred_len:, :]).float()
                    dec_inp_mark = torch.cat([target_mark[:, :self.label_len, :], dec_inp_mark], dim=1).float().to(self.device)
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs, input_mark, dec_inp, dec_inp_mark)
                            if self.output_attention:
                                outputs = outputs[0]
                    else:
                        outputs = self.model(inputs, input_mark, dec_inp, dec_inp_mark)
                        if self.model_name in ['TimesNet', 'Reformer']:
                            outputs = outputs
                        else:
                            outputs = outputs[0]
                    
                    f_dim = -1 if self.features == 'MS' else 0
                    outputs = outputs[:, -self.pred_len:, :]
                    targets = targets[:, -self.pred_len:, :].to(self.device)
                    
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss /= len(train_loader.dataset)

                # Validation loop
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets, val_input_mark, val_target_mark in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        val_input_mark, val_target_mark = val_input_mark.float().to(self.device), val_target_mark.float().to(self.device)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(targets[:, -self.pred_len:, :]).float()
                        dec_inp = torch.cat([targets[:, :self.pred_len, :], dec_inp], dim=1).float().to(self.device)
                        dec_inp_mark = torch.zeros_like(val_target_mark[:, -self.pred_len:, :]).float()
                        dec_inp_mark = torch.cat([val_target_mark[:, :self.label_len, :], dec_inp_mark], dim=1).float().to(self.device)            
                        
                        if self.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(inputs, val_input_mark, dec_inp, dec_inp_mark)
                                if self.output_attention:
                                    outputs = outputs[0]
                        else:
                            outputs = self.model(inputs, val_input_mark, dec_inp, dec_inp_mark)
                        if self.model_name in ['TimesNet', 'Reformer']:
                            outputs = outputs
                        else:
                            outputs = outputs[0]

                        f_dim = -1 if self.features == 'MS' else 0
                        
                        outputs = outputs[:, -self.pred_len:, f_dim:]
                        targets = targets[:, -self.pred_len:, f_dim:].to(self.device)            
                        
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)

                # Validation
                if val_loss > self.best_loss:
                    self.counter += 1
                else:
                    self.best_loss = val_loss
                    self.counter = 0
                
                if self.counter >= self.patience:
                    print("Early Stopping!")
                    break

                print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")               
            
            else:
                for inputs, targets in train_loader:
                    torch.cuda.empty_cache()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss /= len(train_loader.dataset)

                # Validation loop
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)

                # Validation
                if val_loss > self.best_loss:
                    self.counter += 1
                else:
                    self.best_loss = val_loss
                    self.counter = 0
                
                if self.counter >= self.patience:
                    print("Early Stopping!")
                    break

                print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print("Total Time: {}".format(time.time() - time_now))

        print("=======================================================")
        print("=======================TEST MODE=======================")
        print("=======================================================")

        self.model.eval()

        # Evaluation on the test set
        # Create an empty list to store predictions
        all_predictions = []

        inference_time = time.time()

        if self.model_name in ['Autoformer', 'TimesNet', 'Informer', 'Reformer']:
            with torch.no_grad():
                for inputs, targets, input_mark, target_mark in test_loader:  # No need to unpack a tuple
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    input_mark, target_mark = input_mark.float().to(self.device), target_mark.float().to(self.device)
                    
                    # decoder input
                    dec_inp = torch.zeros_like(targets[:, -self.pred_len:, :]).float()
                    dec_inp = torch.cat([targets[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    dec_inp_mark = torch.zeros_like(target_mark[:, -self.pred_len:, :]).float()
                    dec_inp_mark = torch.cat([target_mark[:, :self.label_len, :], dec_inp_mark], dim=1).float().to(self.device)
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs, input_mark, dec_inp, dec_inp_mark)
                            if self.output_attention:
                                outputs = outputs[0]
                    else:
                        outputs = self.model(inputs, input_mark, dec_inp, dec_inp_mark)
                        if self.model_name in ['TimesNet', 'Reformer']:
                            outputs = outputs
                        else:
                            outputs = outputs[0]
                    
                    f_dim = -1 if self.features == 'MS' else 0
                    outputs = outputs[:, -self.pred_len:, :]
                    targets = targets[:, -self.pred_len:, :].to(self.device)
                    
                    outputs = outputs.detach().cpu().numpy()
                    # targets = targets.detach().cpu().numpy()        

                    # Append predictions to the list
                    all_predictions.append(outputs)            
        else:
            # Iterate through the test data loader
            with torch.no_grad():
                for inputs in test_loader:  # No need to unpack a tuple
                    inputs = inputs[0].to(self.device)
                    # Forward pass
                    outputs = self.model(inputs)
                    # Append predictions to the list
                    all_predictions.append(outputs.cpu().numpy())

        # Concatenate all predictions along the batch dimension
        all_predictions = np.concatenate(all_predictions, axis=0)

        print("### Inference time: {}".format(time.time() - inference_time))

        print("=======================================================")
        print("=======================Evaluation======================")
        print("=======================================================")

        _, _, _, ts_ground_truth = _make_windowing_and_loader(self.dataset, self.model_name, self.batch_size, complex_3d, self.seq_day, self.pred_day, self.test_ratio, self.train_ratio, self.cycle)
        test_taget_reshaped = ts_ground_truth.reshape(ts_ground_truth.size(0), ts_ground_truth.size(1)*ts_ground_truth.size(2), ts_ground_truth.size(3))

        y_test = test_taget_reshaped
        # Reshape y_test and all_predictions to 2D arrays
        y_test_2d = y_test.reshape(-1, y_test.shape[-1])  # Flatten along the time series dimension
        all_predictions_2d = all_predictions.reshape(-1, all_predictions.shape[-1])

        # Compute MAE and MSE
        mae = mean_absolute_error(y_test_2d, all_predictions_2d)
        mse = mean_squared_error(y_test_2d, all_predictions_2d)

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)

        return mae, mse