import torch
import torch.nn as nn

import numpy as np
import time
import os
from tqdm import tqdm
import warnings

from models import GraphNet, TemporalConvNet, TemporalConvNet, FeatureExtractor, ResNet1D, Linear
from contrastive_losses import RankingContrastiveLoss, TemporalContrastiveLoss
from data_loader import load_datasets, _make_loader
from calculator import pairwise_cosine_sim

from sklearn.metrics import mean_absolute_error, mean_squared_error

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name        

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = float('inf')
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif np.isnan(val_loss):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True            
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


class CLIP_3D(object):
    DEFAULTS = {}

    def __init__(self, opts):

        self.__dict__.update(CLIP_3D.DEFAULTS, **opts)

        self.path = self.model_save_path
        self.od_input_dim = self.num_node * self.num_node
        self.od_output_dim = self.num_node
        self.ts_input_dim = self.node_feature_dim * self.num_node
        self.ts_num_channels = [self.ts_channel_one, self.ts_channel_two, self.num_node]
        self.target_dim = (self.node_feature_dim * self.num_node) + ( self.num_node * self.num_node )
        self.seq_len = self.seq_day * self.cycle
        self.pred_len = self.pred_day * self.cycle

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_models()
        
        self.prediction_criterion = nn.MSELoss()
        self.criterion = RankingContrastiveLoss()
        self.criterion_temporal = TemporalContrastiveLoss(temperature=self.temperature, softplus_w=self.softplus_w)
        
    def build_models(self):
        self.graph_model = GraphNet(input_dim=self.od_input_dim, hidden_dim=self.od_hidden_dim, output_dim=self.od_output_dim, num_layers=self.od_num_layers)
        self.time_model = TemporalConvNet(num_inputs=self.ts_input_dim, num_channels=self.ts_num_channels, kernel_size=self.ts_kernel_size, dropout=self.dropout)
        self.prediction_model = Linear(self.seq_day, self.pred_day, self.cycle)

        self.feature_extractor_dim = FeatureExtractor(self.num_node, self.target_dim, self.fe_kernel_size, self.output_padding)
        self.feature_extractor_time = ResNet1D(
            in_channels=self.num_node,
            target_dim = self.target_dim,
            base_filters=self.base_filters,
            kernel_size=self.kernel_size,
            stride=self.res_stride,
            groups=self.groups,
            use_bn = self.use_bn,
            n_block=self.n_block,
            n_classes=self.seq_day*self.cycle,
            downsample_gap=self.downsample_gap,
            increasefilter_gap=self.increasefilter_gap,
            use_do=True)
        
        self.optimizer = torch.optim.SGD(list(self.graph_model.parameters()) + list(self.time_model.parameters()), lr=self.lr, momentum=self.momentum)
        self.temporal_optimizer_dim = torch.optim.SGD(self.feature_extractor_dim.parameters(), lr=self.lr, momentum=self.momentum)
        self.temporal_optimizer_time = torch.optim.SGD(self.feature_extractor_time.parameters(), lr=self.lr, momentum=self.momentum)
        self.prediction_optimizer = torch.optim.Adam(self.prediction_model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
            print("Let's use", torch.cuda.device_count(), "GPUs!")

            if self.multi_gpu:
                self.graph_model.to(dtype=torch.bfloat16, device=self.device)
                self.time_model.to(self.device)
                self.feature_extractor_dim.to(dtype=torch.bfloat16, device=self.device)
                self.feature_extractor_time.to(dtype=torch.bfloat16, device=self.device)
                self.prediction_model.to(dtype=torch.bfloat16, device=self.device)

                self.time_model = nn.DataParallel(self.time_model)
                self.prediction_model = nn.DataParallel(self.prediction_model)
              
            else:
                self.graph_model.to(self.device)
                self.time_model.to(self.device)
                self.feature_extractor_dim.to(self.device)
                self.feature_extractor_time.to(self.device)
                self.prediction_model.to(self.device)

    def vali(self, valid_loader, graph_adjacency_matrix, prediction_model, graph_model, time_model, feature_extractor_dim, feature_extractor_time):
        self.graph_model.eval()
        self.time_model.eval()
        self.feature_extractor_dim.eval()
        self.feature_extractor_time.eval()
        self.prediction_model.eval()

        val_loss = []
        with torch.no_grad():
            for batch_idx, (ts_x, od_x, com_x, com_y) in enumerate(valid_loader):
                if self.multi_gpu:
                    adjacency_matrix = graph_adjacency_matrix.to(dtype=torch.bfloat16, device=self.device)
                    edge_index = adjacency_matrix.nonzero(as_tuple=False).t()
                    od_input = od_x.reshape(od_x.size(0), od_x.size(1)*od_x.size(2), -1).to(dtype=torch.bfloat16, device=self.device)
                    ts_input = ts_x.reshape(ts_x.size(0), ts_x.size(1)*ts_x.size(2), -1).permute(0, 2, 1).to(self.device)
                    # edge_index = od_input[0].nonzero(as_tuple=False).t()
                    
                    graph_latent = graph_model(od_input, edge_index) # ouput [N, window, num_node]
                    time_series_latent = time_model(ts_input).bfloat16() # ouput [N, window, num_node]
                    
                    # Reshape for efficiency
                    od_x_reshaped = graph_latent.permute(0, 2, 1).bfloat16()
                    # --> torch.Size([N, num_node, 720]) torch.Size([N, num_node, 720])                    
                
                else:
                    adjacency_matrix = graph_adjacency_matrix.to(self.device)
                    edge_index = adjacency_matrix.nonzero(as_tuple=False).t()                             
                    od_input = od_x.reshape(od_x.size(0), od_x.size(1)*od_x.size(2), -1).to(self.device)
                    ts_input = ts_x.reshape(ts_x.size(0), ts_x.size(1)*ts_x.size(2), -1).permute(0, 2, 1).to(self.device)
                    # edge_index = od_input[0].nonzero(as_tuple=False).t()

                    graph_latent = graph_model(od_input, edge_index) # ouput [N, window, num_node]
                    time_series_latent = time_model(ts_input) # ouput [N, window, num_node]
                
                    # Reshape for efficiency
                    od_x_reshaped = graph_latent.permute(0, 2, 1)
                    # --> torch.Size([N, num_node, 720]) torch.Size([N, num_node, 720])

                cosine_similarity = pairwise_cosine_sim((time_series_latent, od_x_reshaped))
                # --> cosine_similarity:torch.Size([N, num_node, num_node]), expanded_adjacency_matrix: torch.Size([N, num_node, num_node])
                
                # Forward pass: cosine_similarity = torch.Size([N, num_node, num_node])
                output_features_dim = feature_extractor_dim(cosine_similarity)
                output_features_time = feature_extractor_time(output_features_dim.permute(0, 2, 1))
                
                if self.multi_gpu:
                    ts_target = com_y.reshape(com_y.size(0), com_y.size(1)*com_y.size(2), -1).to(dtype=torch.bfloat16, device=self.device)
                    ts_input_reshaped = com_x.reshape(com_x.size(0), com_x.size(1)*com_x.size(2), -1).to(dtype=torch.bfloat16, device=self.device)                   
                else:
                    ts_target = com_y.reshape(com_y.size(0), com_y.size(1)*com_y.size(2), -1).to(self.device)
                    ts_input_reshaped = com_x.reshape(com_x.size(0), com_x.size(1)*com_x.size(2), -1).to(self.device)

                prediction_input = ts_input_reshaped + output_features_time
                outputs = prediction_model(prediction_input).float()           
                
                prediction_loss = self.prediction_criterion(outputs, ts_target)
                val_loss.append(prediction_loss.cpu().numpy())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                valid_loss = np.average(val_loss)

        return valid_loss

    def proceed(self):
        print("======================================================")
        print("======================TRAIN MODE======================")
        print("======================================================")
        
        time_now = time.time()
        ts_tensors_3d, od_tensors_3d, complex_3d, graph_adjacency_matrix = load_datasets(self.dataset, self.khop)
        train_loader, test_loader, valid_loader, _, _ = _make_loader(self.batch_size, ts_tensors_3d, od_tensors_3d, complex_3d, self.seq_day, self.pred_day, self.test_ratio, self.cycle, self.train_ratio)
        
        early_stopping = EarlyStopping(patience=5, verbose=False, dataset_name=self.dataset)
        train_steps = len(train_loader)

        for epoch in tqdm(range(self.num_epochs)):
            accumulated_loss = []
            iter_count = 0

            epoch_time = time.time()
            self.graph_model.train()
            self.time_model.train()
            self.feature_extractor_dim.train()
            self.feature_extractor_time.train()
            self.prediction_model.train()

            for batch_idx, (ts_x, od_x, com_x, com_y) in enumerate(train_loader):
                torch.cuda.empty_cache()
                # optimizers
                self.optimizer.zero_grad()
                self.temporal_optimizer_dim.zero_grad()
                self.temporal_optimizer_time.zero_grad()
                iter_count += 1
                
                if self.multi_gpu:
                    adjacency_matrix = graph_adjacency_matrix.to(dtype=torch.bfloat16, device=self.device)
                    edge_index = adjacency_matrix.nonzero(as_tuple=False).t()
                    od_input = od_x.reshape(od_x.size(0), od_x.size(1)*od_x.size(2), -1).to(dtype=torch.bfloat16, device=self.device)
                    ts_input = ts_x.reshape(ts_x.size(0), ts_x.size(1)*ts_x.size(2), -1).permute(0, 2, 1).to(self.device)
                    # edge_index = od_input[0].nonzero(as_tuple=False).t()

                    graph_latent = self.graph_model(od_input, edge_index) # ouput [N, window, num_node]
                    time_series_latent = self.time_model(ts_input).permute(0, 2, 1) # ouput [N, window, num_node]

                    # Reshape for efficiency
                    ts_x_reshaped = time_series_latent.permute(0, 2, 1).bfloat16() # --> torch.Size([N, num_node, window])
                    od_x_reshaped = graph_latent.permute(0, 2, 1).bfloat16() # --> torch.Size([N, num_node, window])

                else:
                    adjacency_matrix = graph_adjacency_matrix.to(self.device)
                    edge_index = adjacency_matrix.nonzero(as_tuple=False).t()
                    od_input = od_x.reshape(od_x.size(0), od_x.size(1)*od_x.size(2), -1).to(self.device)
                    ts_input = ts_x.reshape(ts_x.size(0), ts_x.size(1)*ts_x.size(2), -1).permute(0, 2, 1).to(self.device)
                    # edge_index = od_input[0].nonzero(as_tuple=False).t()

                    graph_latent = self.graph_model(od_input, edge_index) # ouput [N, window, num_node]
                    time_series_latent = self.time_model(ts_input).permute(0, 2, 1) # ouput [N, window, num_node]

                    # Reshape for efficiency
                    ts_x_reshaped = time_series_latent.permute(0, 2, 1) # --> torch.Size([N, num_node, window])
                    od_x_reshaped = graph_latent.permute(0, 2, 1) # --> torch.Size([N, num_node, window])

                cosine_similarity = pairwise_cosine_sim((ts_x_reshaped, od_x_reshaped))
                current_batch_size, _, _ = cosine_similarity.size()
                expanded_adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(current_batch_size, 1, 1)
                # --> cosine_similarity:torch.Size([N, num_node, num_node]), expanded_adjacency_matrix: torch.Size([N, num_node, num_node])

                # Compute ranking contrastive loss
                spatial_loss = self.criterion(cosine_similarity, expanded_adjacency_matrix)

                # Feature_extractor_dim: cosine_similarity = torch.Size([N, num_node, num_node]) --> output: torch.Size([N, num_node, target_dim])
                # Feature_extractor_time: cosine_similarity = torch.Size([N, num_node, target_dim]) --> output: torch.Size([N, window, target_dim])
                output_features_dim = self.feature_extractor_dim(cosine_similarity)
                output_features_time = self.feature_extractor_time(output_features_dim.permute(0, 2, 1)) # torch.Size([8, 720, 206])
                temporal_loss = self.criterion_temporal(output_features_time)

                # Forecasting training loop
                if self.multi_gpu:
                    ts_target = com_y.reshape(com_y.size(0), com_y.size(1)*com_y.size(2), -1).to(dtype=torch.bfloat16, device=self.device)
                    ts_input_reshaped = com_x.reshape(com_x.size(0), com_x.size(1)*com_x.size(2), -1).to(dtype=torch.bfloat16, device=self.device)                   
                else:
                    ts_target = com_y.reshape(com_y.size(0), com_y.size(1)*com_y.size(2), -1).to(self.device)
                    ts_input_reshaped = com_x.reshape(com_x.size(0), com_x.size(1)*com_x.size(2), -1).to(self.device)
                
                prediction_input = ts_input_reshaped + output_features_time
                self.prediction_optimizer.zero_grad()
                outputs = self.prediction_model(prediction_input)
                prediction_loss = self.prediction_criterion(outputs, ts_target)

                loss = 0.25*spatial_loss + 0.25*temporal_loss + 0.5*prediction_loss

                # Caculate computation cost
                if (batch_idx + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - batch_idx)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Backward pass and optimization step
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.temporal_optimizer_dim.step()
                self.temporal_optimizer_time.step()                 
                self.prediction_optimizer.step()   

                # print(f'[{batch_idx}th batch]')
                # print(f'Spatial Loss: {spatial_loss.item()}, Temporal Loss: {temporal_loss.item()}')
                # print(f'Forecasting Loss: {prediction_loss.item()}, Loss Sum: {loss.item()}')
                accumulated_loss.append(loss.item())

            total_loss = np.average(accumulated_loss)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            valid_loss = self.vali(valid_loader, graph_adjacency_matrix, self.prediction_model, self.graph_model, self.time_model, self.feature_extractor_dim, self.feature_extractor_time)

            print("Epoch: [{0}/{1}], Steps: {2} | Train Loss: {3:.7f} Vali Loss: {3:.7f}".format( epoch+1, self.num_epochs, train_steps, total_loss, valid_loss))
            early_stopping(valid_loss, self.prediction_model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.prediction_optimizer, epoch + 1, self.lr)

        print("Total Time: {}".format(time.time() - time_now))

        print("=======================================================")
        print("=======================TEST MODE=======================")
        print("=======================================================")

        self.prediction_model.eval()

        # Evaluation on the test set
        # Create an empty list to store predictions
        all_predictions = []

        inference_time = time.time()
        with torch.no_grad():
            for batch_idx, (ts_x, od_x, com_x, com_y) in tqdm(enumerate(test_loader)):
                if self.multi_gpu:
                    od_input = od_x.reshape(od_x.size(0), od_x.size(1)*od_x.size(2), -1).to(dtype=torch.bfloat16, device=self.device)
                    ts_input = ts_x.reshape(ts_x.size(0), ts_x.size(1)*ts_x.size(2), -1).permute(0, 2, 1).to(self.device)
                    # edge_index = od_input[0].nonzero(as_tuple=False).t()
                    
                    graph_latent = self.graph_model(od_input, edge_index) # ouput [N, window, num_node]
                    time_series_latent = self.time_model(ts_input).bfloat16() # ouput [N, window, num_node]
                    
                    # Reshape for efficiency
                    od_x_reshaped = graph_latent.permute(0, 2, 1).bfloat16()
                    # --> torch.Size([N, num_node, 720]) torch.Size([N, num_node, 720])                    
                
                else:
                    od_input = od_x.reshape(od_x.size(0), od_x.size(1)*od_x.size(2), -1).to(self.device)
                    ts_input = ts_x.reshape(ts_x.size(0), ts_x.size(1)*ts_x.size(2), -1).permute(0, 2, 1).to(self.device)
                    edge_index = od_input[0].nonzero(as_tuple=False).t()

                    graph_latent = self.graph_model(od_input, edge_index) # ouput [N, window, num_node]
                    time_series_latent = self.time_model(ts_input) # ouput [N, window, num_node]
                
                    # Reshape for efficiency
                    od_x_reshaped = graph_latent.permute(0, 2, 1)
                    # --> torch.Size([N, num_node, 720]) torch.Size([N, num_node, 720])

                cosine_similarity = pairwise_cosine_sim((time_series_latent, od_x_reshaped))
                current_batch_size, _, _ = cosine_similarity.size()
                expanded_adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(current_batch_size, 1, 1)
                # --> cosine_similarity:torch.Size([N, num_node, num_node]), expanded_adjacency_matrix: torch.Size([N, num_node, num_node])
                
                # Forward pass: cosine_similarity = torch.Size([N, num_node, num_node])
                output_features_dim = self.feature_extractor_dim(cosine_similarity)
                output_features_time = self.feature_extractor_time(output_features_dim.permute(0, 2, 1))
                if self.multi_gpu:
                    ts_input_reshaped = com_x.reshape(com_x.size(0), com_x.size(1)*com_x.size(2), -1).to(self.device)
                else:
                    ts_input_reshaped = com_x.reshape(com_x.size(0), com_x.size(1)*com_x.size(2), -1).to(dtype=torch.bfloat16, device=self.device)
                
                prediction_input = ts_input_reshaped + output_features_time
                outputs = self.prediction_model(prediction_input).float()

                # Append predictions to the list
                all_predictions.append(outputs.cpu().numpy())

        # Concatenate all predictions along the batch dimension
        all_predictions = np.concatenate(all_predictions, axis=0)

        print("### Inference time: {}".format(time.time() - inference_time))

        print("=======================================================")
        print("=======================Evaluation======================")
        print("=======================================================")

        _, _, _, _, ts_ground_truth = _make_loader(self.batch_size, ts_tensors_3d, od_tensors_3d, complex_3d, self.seq_day, self.pred_day, self.test_ratio, self.cycle, self.train_ratio)
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
