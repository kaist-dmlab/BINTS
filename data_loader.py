import numpy as np
import pandas as pd
import os

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def linear_regression_fit(n, x):
    y = - 1 / (n+1) * (x-1) + 1
    return y

def exponential_regression_fit(n, x):
    y = np.exp(-(x-1))
    return y

def k_hop_matrix(adjacency_matrix, max_hops):
    # Initialize the k-hop matrix with the original adjacency matrix
    k_hop_matrix = np.array(adjacency_matrix, dtype=np.float32)
    
    # Raise the adjacency matrix to the power of 2 up to max_hops
    for _ in range(2, max_hops + 1):
        k_hop_matrix += np.linalg.matrix_power(adjacency_matrix, _) # Reachability matrix
    
    # Set values greater than max_hops to 0
    k_hop_matrix[k_hop_matrix > max_hops] = 0
    
    # Replace the diagonal with 1s from the identity matrix
    np.fill_diagonal(k_hop_matrix, 1)
    
    return k_hop_matrix

# Generated training sequences for use in the model.
def _create_sequences(values, len_lookback, len_forecast, stride=1):
    tensor_x = []
    tensor_y = []
    values = torch.from_numpy(values).float()
    for i in range(0, len(values) - len_lookback - len_forecast + 1, stride):
        tensor_x.append(values[i : i + len_lookback])
        tensor_y.append(values[i + len_lookback : i + len_lookback + len_forecast])
   
    return torch.stack(tensor_x), torch.stack(tensor_y)

def _reshape_for_normalization(value):
    days, timestamp, dim = value.size()
    value_2d = value.reshape(days*timestamp, dim)
    return value_2d

def _reshape_for_restore(value, cycle):
    hours_per_day = cycle
    days_time, dim = value.shape[0], value.shape[1]
    days = days_time // hours_per_day
    value = value.reshape(days, hours_per_day, dim)
    return value

def _get_time_features(start_date, end_date):
    start_date = start_date
    end_date = end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Make dataframe for time features
    df_stamp = pd.DataFrame()
    df_stamp['date'] = pd.to_datetime(date_range.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
    df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
    df_stamp = df_stamp.drop(columns=['date']).values
    return df_stamp

def _make_windowing_and_loader(dataset, model, batch_size, tensor, seq_day, pred_day, test_ratio, train_ratio, cycle):
    # Define the lookback window and forecasting window
    lookback_window = seq_day  # Number of days to look back
    forecasting_window = pred_day  # Number of days to forecast
    test_start_index = len(tensor) - int(len(tensor) * test_ratio) # Assume you want to use the last 90 days for testing

    # train, test seperate
    ts_train = tensor[:test_start_index]
    ts_test = tensor[test_start_index:]
    
    ts_train = _reshape_for_normalization(ts_train)
    ts_test = _reshape_for_normalization(ts_test)

    scaler = StandardScaler()
    
    ts_train = scaler.fit_transform(ts_train)
    ts_test = scaler.transform(ts_test)

    scaled_ts_train = _reshape_for_restore(ts_train, cycle)
    scaled_ts_test = _reshape_for_restore(ts_test, cycle)
    
    ts_train_x, ts_train_y = _create_sequences(scaled_ts_train, lookback_window, forecasting_window)
    # print("TS TrainX: ", ts_train_x.shape, "TS TrainY: ", ts_train_y.shape)
    ts_test_x, ts_test_y = _create_sequences(scaled_ts_test, lookback_window, forecasting_window)
    # print("TS TestX: ", ts_test_x.shape, "TS TestY: ", ts_test_y.shape)

    ts_train_x_reshaped = ts_train_x.reshape(ts_train_x.size(0), ts_train_x.size(1)*ts_train_x.size(2), ts_train_x.size(3))
    ts_train_y_reshaped = ts_train_y.reshape(ts_train_y.size(0), ts_train_y.size(1)*ts_train_y.size(2), ts_train_y.size(3))
    ts_test_x_reshaped = ts_test_x.reshape(ts_test_x.size(0), ts_test_x.size(1)*ts_test_x.size(2), ts_test_x.size(3))
    ts_test_y_reshaped = ts_test_y.reshape(ts_test_y.size(0), ts_test_y.size(1)*ts_test_y.size(2), ts_test_y.size(3))

    # Split the data into training and validation sets
    train_size = int(train_ratio * len(ts_train_x_reshaped))
    train_input = ts_train_x_reshaped[:train_size]
    train_target = ts_train_y_reshaped[:train_size]
    
    val_input = ts_train_x_reshaped[train_size:]
    val_target = ts_train_y_reshaped[train_size:]

    if model in ['Autoformer', 'TimesNet', 'Informer', 'Reformer']:
        if dataset == 'nyc':
            start_date = '2021-01-01'
            end_date = '2023-12-31'
        elif dataset == 'covid':
            start_date = '2020-01-20'
            end_date = '2023-08-31'
        elif dataset == 'nyc_covid':
            start_date = '2020-03-01'
            end_date = '2023-12-31'
        elif dataset == 'busan_new':
            start_date = '2021-01-01'
            end_date = '2023-12-31'
        elif dataset == 'daegu_new':
            start_date = '2021-01-01'
            end_date = '2023-12-31'              
        elif dataset == 'seoul_new':
            start_date = '2022-01-01'
            end_date = '2023-12-31'
        else:
            start_date = '2022-01-01'
            end_date = '2022-12-31'
        data_stamp = _get_time_features(start_date, end_date)

        seq_train_mark = data_stamp[:test_start_index]
        seq_test_mark = data_stamp[test_start_index:]    
        seq_train_mark = scaler.fit_transform(seq_train_mark)
        seq_test_mark = scaler.transform(seq_test_mark)        
        train_x_mark, train_y_mark = _create_sequences(seq_train_mark, lookback_window, forecasting_window)
        test_x_mark, test_y_mark = _create_sequences(seq_test_mark, lookback_window, forecasting_window)

        train_x_mark = train_x_mark.repeat(1, cycle, 1)
        train_y_mark = train_y_mark.repeat(1, cycle, 1)
        test_x_mark = test_x_mark.repeat(1, cycle, 1)
        test_y_mark = test_y_mark.repeat(1, cycle, 1)

        train_input_mark = train_x_mark[:train_size]
        train_target_mark = train_y_mark[:train_size]
        val_input_mark = train_x_mark[train_size:]
        val_target_mark = train_y_mark[train_size:]

        train_dataset = TensorDataset(train_input, train_target, train_input_mark, train_target_mark)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_input, val_target, val_input_mark, val_target_mark)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(ts_test_x_reshaped, ts_test_y_reshaped, test_x_mark, test_y_mark)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)        
    
    else:
        train_dataset = TensorDataset(train_input, train_target)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_input, val_target)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(ts_test_x_reshaped)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, ts_test_y


def _make_loader(batch_size, ts_latent_tensor, od_latent_tensor, complex_tensor, seq_day, pred_day, test_ratio, cycle, train_ratio):
    # Define the lookback window and forecasting window
    lookback_window = seq_day  # Number of days to look back
    forecasting_window = pred_day  # Number of days to forecast
    test_start_index = len(ts_latent_tensor) - int(len(ts_latent_tensor) * test_ratio)
    
    # train, test seperate
    ts_train = ts_latent_tensor[:test_start_index]
    ts_test = ts_latent_tensor[test_start_index:]
    od_train = od_latent_tensor[:test_start_index]
    od_test = od_latent_tensor[test_start_index:]
    complex_train = complex_tensor[:test_start_index]
    complex_test = complex_tensor[test_start_index:]
    
    ts_train = _reshape_for_normalization(ts_train)
    ts_test = _reshape_for_normalization(ts_test)
    od_train = _reshape_for_normalization(od_train)
    od_test = _reshape_for_normalization(od_test)
    complex_train = _reshape_for_normalization(complex_train)
    complex_test = _reshape_for_normalization(complex_test)

    scaler = StandardScaler()
    
    ts_train = scaler.fit_transform(ts_train)
    ts_test = scaler.transform(ts_test)
    od_train = scaler.fit_transform(od_train)
    od_test = scaler.transform(od_test)
    complex_train = scaler.fit_transform(complex_train)
    complex_test = scaler.transform(complex_test)    

    scaled_ts_train = _reshape_for_restore(ts_train, cycle)
    scaled_ts_test = _reshape_for_restore(ts_test, cycle)
    scaled_od_train = _reshape_for_restore(od_train, cycle)
    scaled_od_test = _reshape_for_restore(od_test, cycle)
    scaled_complex_train = _reshape_for_restore(complex_train, cycle)
    scaled_complex_test = _reshape_for_restore(complex_test, cycle)
    
    ts_train_x, ts_train_y = _create_sequences(scaled_ts_train, lookback_window, forecasting_window)
    od_train_x, od_train_y = _create_sequences(scaled_od_train, lookback_window, forecasting_window)
    complex_train_x, complex_train_y = _create_sequences(scaled_complex_train, lookback_window, forecasting_window)
    
    ts_test_x, ts_test_y = _create_sequences(scaled_ts_test, lookback_window, forecasting_window)
    od_test_x, od_test_y = _create_sequences(scaled_od_test, lookback_window, forecasting_window)
    complex_test_x, complex_test_y = _create_sequences(scaled_complex_test, lookback_window, forecasting_window)


    # Create Validation Dataset
    train_size = int(train_ratio * len(complex_train_x))
    timeseries_train_x, timeseries_train_y = ts_train_x[:train_size], ts_train_y[:train_size]
    origindestin_train_x, origindestin_train_y = od_train_x[:train_size], od_train_y[:train_size]
    com_train_x, com_train_y = complex_train_x[:train_size], complex_train_y[:train_size]
    
    ts_val_x, ts_val_y = ts_train_x[train_size:], ts_train_y[train_size:]
    od_val_x, od_val_y = od_train_x[train_size:], od_train_y[train_size:]
    com_valid_x, com_valid_y = complex_train_x[train_size:], complex_train_y[train_size:]

    # Create a PyTorch Dataset
    train_dataset = TensorDataset(timeseries_train_x, origindestin_train_x, com_train_x, com_train_y)

    # Create a PyTorch Dataset
    test_dataset = TensorDataset(ts_test_x, od_test_x, complex_test_x, complex_test_y)
    
    # Create a PyTorch Dataset
    valid_dataset = TensorDataset(ts_val_x, od_val_x, com_valid_x, com_valid_y)

    # Create test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader, complex_train_y, complex_test_y


def load_datasets(dataset, khop=0):
    # Dataset directory
    ADJ_DIR = "/data2/maradonam/adj_matrix/"
    if dataset == "seoul":
        ADJ_FILE = "seoul_adj_matrix_with_diag_1.npy"
        TS_DIR = "/data2/maradonam/seoul_node_npy_dataset/"
        OD_DIR = "/data2/maradonam/seoul_npy_dataset_233_by_233/"

        # Origin-Destination Data
        od_datasets  = sorted([f for f in os.listdir(f'{OD_DIR}') if os.path.isdir(os.path.join(f'{OD_DIR}', f))])

        od_stacks = []
        for month in od_datasets:
            od_files = sorted([f for f in os.listdir(f'{OD_DIR}/{month}') if os.path.isfile(os.path.join(f'{OD_DIR}/{month}', f))])
            for npy in od_files:
                node_feature_data = torch.tensor(np.load(f'{OD_DIR}{month}/' + npy), dtype=torch.float32)
                node_feature_data = np.transpose(np.array(node_feature_data), (2, 0, 1))
                od_stacks.append(node_feature_data)
        od_tensors = torch.Tensor(np.stack(od_stacks))
    
    elif dataset == "seoul_new":
        ADJ_FILE = "seoul_adj_matrix_with_diag_1_128.npy"
        TS_DIR = "/data2/maradonam/seoul_node_npy_dataset_new/"
        OD_DIR = "/data2/maradonam/seoul_npy_dataset_new/"

        # Origin-Destination Data
        od_datasets  = sorted([f for f in os.listdir(f'{OD_DIR}') if os.path.isdir(os.path.join(f'{OD_DIR}', f))])

        od_stacks = []
        for month in od_datasets:
            od_files = sorted([f for f in os.listdir(f'{OD_DIR}/{month}') if os.path.isfile(os.path.join(f'{OD_DIR}/{month}', f))])
            for npy in od_files:
                node_feature_data = torch.tensor(np.load(f'{OD_DIR}{month}/' + npy), dtype=torch.float32)
                node_feature_data = np.transpose(np.array(node_feature_data), (2, 0, 1))
                od_stacks.append(node_feature_data)
        od_tensors = torch.Tensor(np.stack(od_stacks))
         
    else:
        if dataset == "busan":
            ADJ_FILE = "busan_adj_matrix_with_diag_1.npy"
            TS_DIR = "/data2/maradonam/2022_busan_node_npy_dataset/"
            OD_DIR = "/data2/maradonam/2022_busan_npy_dataset_103_by_103/"
        elif dataset == "busan_new":
            ADJ_FILE = "busan_adj_matrix_with_diag_1_60.npy"
            TS_DIR = "/data2/maradonam/busan_node_npy_dataset_new/"
            OD_DIR = "/data2/maradonam/busan_od_npy_dataset_new/"
        elif dataset == "daegu":
            ADJ_FILE = "daegu_adj_matrix_with_diag_1.npy"
            TS_DIR = "/data2/maradonam/2022_daegu_node_npy_dataset/"
            OD_DIR = "/data2/maradonam/2022_daegu_npy_dataset_85_by_85/"
        elif dataset == "daegu_new":
            ADJ_FILE = "daegu_adj_matrix_with_diag_1_61.npy"
            TS_DIR = "/data2/maradonam/daegu_node_npy_dataset_new/"
            OD_DIR = "/data2/maradonam/daegu_od_npy_dataset_new/"            
        elif dataset == "covid":
            ADJ_FILE = "nationwide_adj_matrix_with_diag_1.npy"
            TS_DIR = "/data2/maradonam/nationwide_node_npy_dataset/"
            OD_DIR = "/data2/maradonam/nationwide_npy_dataset/"
        elif dataset == "nyc":
            ADJ_FILE = "nyc_taxi_matrix_with_diag_1.npy"
            TS_DIR = "/data2/maradonam/nyc_node_npy_dataset/"
            OD_DIR = "/data2/maradonam/nyc_od_npy_dataset/"
        elif dataset == "nyc_covid":
            ADJ_FILE = "nyc_covid_matrix_with_diag_1.npy"
            TS_DIR = "/data2/maradonam/nyc_covid_node_npy_dataset/"
            OD_DIR = "/data2/maradonam/nyc_covid_od_npy_dataset/"
        
        # Origin-Destination Data
        od_datasets  = sorted([f for f in os.listdir(f'{OD_DIR}') if os.path.isfile(os.path.join(f'{OD_DIR}', f))])

        od_stacks = []
        for npy in od_datasets:
            node_feature_data = torch.tensor(np.load(OD_DIR + npy), dtype=torch.float32)
            node_feature_data = np.transpose(np.array(node_feature_data), (2, 0, 1))
            od_stacks.append(node_feature_data)
        od_tensors = torch.Tensor(np.stack(od_stacks))
        
    # Topology Data
    graph_adjacency_matrix = torch.tensor(np.load(ADJ_DIR + ADJ_FILE), dtype=torch.float32)
    if khop > 0:
        original_matrix = graph_adjacency_matrix
        for hop in range(1, khop+1):
            k_hop_matrix_result = k_hop_matrix(original_matrix, hop)
            # continuous_value = linear_regression_fit(khop, hop)
            continuous_value = exponential_regression_fit(khop, hop)
            R_per_hop = continuous_value*(k_hop_matrix_result>0)
            
            # print(f'{hop}th hops: ')
            # print(R_per_hop)
            
            graph_adjacency_matrix += continuous_value*((original_matrix==0) & (R_per_hop>0))        

    # Node Feature TS Data
    ts_datasets  = sorted([f for f in os.listdir(f'{TS_DIR}') if os.path.isfile(os.path.join(f'{TS_DIR}', f))])

    ts_stacks = []
    for npy in ts_datasets:
        time_series_data = torch.tensor(np.load(TS_DIR + npy), dtype=torch.float32)
        time_series_data = np.transpose(np.array(time_series_data), (1, 0, 2))
        ts_stacks.append(time_series_data)
    ts_tensors = torch.tensor(np.stack(ts_stacks))

    ts_tensors_3d = ts_tensors.reshape(ts_tensors.size(0), ts_tensors.size(1), ts_tensors.size(2)*ts_tensors.size(3))
    od_tensors_3d = od_tensors.reshape(od_tensors.size(0), od_tensors.size(1), od_tensors.size(2)*od_tensors.size(3))
    
    # Multi-Modal Concat (High-dim)
    complex_3d = torch.cat((ts_tensors_3d, od_tensors_3d), dim=-1)

    return ts_tensors_3d, od_tensors_3d, complex_3d, graph_adjacency_matrix