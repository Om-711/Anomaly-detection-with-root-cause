import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from queue import Queue
import json

n = 1000  
seq_length = 10  

# timestamps
timestamps = pd.date_range(start="2025-09-26", periods=n, freq='min')

services = ['frontend', 'auth-service', 'database-service', 'cache-service']
event_types = ['SERVICE_RESTART', 'CONFIG_CHANGE', 'USER_LOGIN', 'DEPLOYMENT']
log_levels = ['INFO', 'WARN', 'ERROR']

def generate_metrics(n=n):

    print("\nGeneating metrics data ..........")
    cpu_base = 18
    mem_base = 1507

    cpu_usage = cpu_base + np.random.randint(-5, 5, size=n)
    mem_usage = mem_base + np.random.randint(-250, 250, size=n)
    response_time = np.random.randint(100, 110, size=n)
    services_arr = np.random.choice(services, size=n)

    # Random spiles
    is_anomaly = np.random.rand(n) < 0.05
    cpu_usage[is_anomaly] += np.random.randint(50, 70, size=is_anomaly.sum())
    response_time[is_anomaly] += np.random.randint(300, 500, size=is_anomaly.sum())
    is_anomaly = is_anomaly.astype(np.int8)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage.astype(np.int16),
        'mem_base': mem_usage.astype(np.int32),
        'response_time': response_time.astype(np.int16),
        'service': services_arr,
        'is_anomaly': is_anomaly
    })
    return df

def generate_logs(n=n):

    print("\nGeneating Logs data ..........")
    levels = np.random.choice(log_levels, size=n, p=[0.7, 0.2, 0.1])
    services_arr = np.random.choice(services, size=n)

    # services = ['frontend', 'auth-service', 'database-service', 'cache-service']
    # log_levels = ['INFO', 'WARN', 'ERROR']
    
    messages = np.array([
        f"{s} operation successful" if l=="INFO" else 
        f"{s} high memory usage" if l=="WARN" else 
        f"{s} error occurred"
        for s, l in zip(services_arr, levels)
    ])

    is_anomaly = (levels == 'ERROR').astype(np.int8)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'level': levels,
        'message': messages,
        'service': services_arr,
        'is_anomaly': is_anomaly
    })

    return df

def generate_events(n=n):

    print("\nGeneating events data ..........")
    event_arr = np.random.choice(event_types, size=n)
    services_arr = np.random.choice(services, size=n)

    # services = ['frontend', 'auth-service', 'database-service', 'cache-service']
    # event_types = ['SERVICE_RESTART', 'CONFIG_CHANGE', 'USER_LOGIN', 'DEPLOYMENT']

    is_anomaly = (event_arr == 'SERVICE_RESTART').astype(np.int8)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'event_type': event_arr,
        'service': services_arr,
        'is_anomaly': is_anomaly
    })
    return df

def generate_traces(n=n):

    print("\nGeneating traces data ..........")
    trace_id = np.arange(1, n+1)
    duration = np.random.randint(20, 50, size=(n, len(services)))

    data = []
    for idx, t in enumerate(timestamps):
        parent_id = None
        for s_idx, service in enumerate(services):
            # 20%  span is missing  consider missing span as anomaly
            missing_span = np.random.rand() < 0.2
            span_id = f"{trace_id[idx]}_{service}"
            if missing_span:
                data.append([t, trace_id[idx], span_id, parent_id, service, duration[idx, s_idx], 1])
                continue
            data.append([t, trace_id[idx], span_id, parent_id, service, duration[idx, s_idx], 0])
            parent_id = span_id

    df = pd.DataFrame(data, columns=['start_time','trace_id','span_id','parent_id','service','duration_ms','is_anomaly'])
    return df