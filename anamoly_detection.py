from sklearn.svm import OneClassSVM
from scipy.stats import zscore
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.ensemble import  IsolationForest

def make_dataloader(X_seq, seq_length=10, batch_size=16, shuffle=True):
    loader = DataLoader(X_seq, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader


def create_sequences(arr, seq_len=10):
    n = arr.shape[0] - seq_len
    sequences = np.array([arr[i:i+seq_len] for i in range(n)], dtype=np.float32)
    return torch.from_numpy(sequences)

# print(X_seq.shape)
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1),1,1).permute(1,0,2)
        out, _ = self.decoder(h)
        return out


def detect_anomalies_melt(metrics_df, log_df, event_df, trace_df):
    print("\n Metrics Anomaly Detection...")
    X = metrics_df[['cpu_usage', 'mem_base', 'response_time']]

    # Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    metrics_df['metrics_iso'] = iso.fit_predict(X)
    metrics_df['metrics_iso'] = metrics_df['metrics_iso'].map({1: 0, -1: 1})

    # print("\nAnamoly detection using SVM......") 
    oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto') 
    metrics_df['metrics_svm'] = oc_svm.fit_predict(X)

    # Statistical z-score
    metrics_df['metrics_stat'] = (np.abs(zscore(metrics_df['cpu_usage'])) > 3).astype(int)

    # LSTM Autoencoder
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = create_sequences(X_scaled, seq_len=10)

    model = LSTMAutoEncoder(input_dim=X_seq.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    dataloader = DataLoader(X_seq, batch_size=512, shuffle=False)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f"[Metrics] Epoch {epoch+1}/3 loss={total_loss/len(dataloader.dataset):.6f}")

    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(mse)
    errors = np.concatenate(errors)
    threshold = errors.mean() + 3 * errors.std()
    metrics_df['metrics_lstm'] = 0
    for idx in np.where(errors > threshold)[0]:
        metrics_df.loc[idx:idx+9, 'metrics_lstm'] = 1

    # ---------------- LOGS ----------------
    print("\n Log Anomaly Detection...")
    X_logs = (log_df['level'] == 'ERROR').astype(int).values.reshape(-1, 1)
    log_df['log_iso'] = IsolationForest(contamination=0.05, random_state=42).fit_predict(X_logs)
    log_df['log_iso'] = log_df['log_iso'].map({1: 0, -1: 1})
    log_df['log_stat'] = (X_logs > 0).astype(int)
    log_df['log_svm'] = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto').fit_predict(X_logs)
    log_df['log_svm'] = log_df['log_svm'].map({1: 0, -1: 1})


    X_logs_scaled = MinMaxScaler().fit_transform(X_logs)
    X_seq_logs = create_sequences(X_logs_scaled, seq_len=10)
    model_logs = LSTMAutoEncoder(input_dim=X_seq_logs.shape[2])
    opt_logs = torch.optim.Adam(model_logs.parameters(), lr=0.01)
    dataloader_logs = DataLoader(X_seq_logs, batch_size=512, shuffle=False)

    for epoch in range(3):
        for batch in dataloader_logs:
            opt_logs.zero_grad()
            recon = model_logs(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt_logs.step()

    model_logs.eval()
    errors_logs = []
    with torch.no_grad():
        for batch in dataloader_logs:
            recon = model_logs(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors_logs.append(mse)
    errors_logs = np.concatenate(errors_logs)
    threshold_logs = errors_logs.mean() + 3 * errors_logs.std()
    log_df['log_lstm'] = 0
    for idx in np.where(errors_logs > threshold_logs)[0]:
        log_df.loc[idx:idx+9, 'log_lstm'] = 1

    # ---------------- EVENTS ----------------
    print("\n Event Anomaly Detection...")
    X_events = (event_df['event_type'] == 'SERVICE_RESTART').astype(int).values.reshape(-1, 1)
    event_df['event_iso'] = IsolationForest(contamination=0.05, random_state=42).fit_predict(X_events)
    event_df['event_iso'] = event_df['event_iso'].map({1: 0, -1: 1})
    event_df['event_stat'] = X_events
    event_df['event_svm'] = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto').fit_predict(X_events)
    event_df['event_svm'] = event_df['event_svm'].map({1: 0, -1: 1})


    X_events_scaled = MinMaxScaler().fit_transform(X_events)
    X_seq_events = create_sequences(X_events_scaled, seq_len=10)
    model_events = LSTMAutoEncoder(input_dim=X_seq_events.shape[2])
    opt_events = torch.optim.Adam(model_events.parameters(), lr=0.01)
    dataloader_events = DataLoader(X_seq_events, batch_size=512, shuffle=False)

    for epoch in range(3):
        for batch in dataloader_events:
            opt_events.zero_grad()
            recon = model_events(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt_events.step()

    model_events.eval()
    errors_events = []
    with torch.no_grad():
        for batch in dataloader_events:
            recon = model_events(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors_events.append(mse)
    errors_events = np.concatenate(errors_events)
    threshold_events = errors_events.mean() + 3 * errors_events.std()
    event_df['event_lstm'] = 0
    for idx in np.where(errors_events > threshold_events)[0]:
        event_df.loc[idx:idx+9, 'event_lstm'] = 1

    # ---------------- TRACES ----------------
    print("\n Trace Anomaly Detection...")
    X_traces = trace_df[['duration_ms']].values
    trace_df['trace_iso'] = IsolationForest(contamination=0.05, random_state=42).fit_predict(X_traces)
    trace_df['trace_iso'] = trace_df['trace_iso'].map({1: 0, -1: 1})
    trace_df['trace_stat'] = (np.abs(zscore(trace_df['duration_ms'])) > 3).astype(int)
    trace_df['trace_svm'] = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto').fit_predict(X_traces)
    trace_df['trace_svm'] = trace_df['trace_svm'].map({1: 0, -1: 1})

    X_traces_scaled = MinMaxScaler().fit_transform(X_traces)
    X_seq_traces = create_sequences(X_traces_scaled, seq_len=10)
    model_traces = LSTMAutoEncoder(input_dim=X_seq_traces.shape[2])
    opt_traces = torch.optim.Adam(model_traces.parameters(), lr=0.01)
    dataloader_traces = DataLoader(X_seq_traces, batch_size=512, shuffle=False)

    for epoch in range(3):
        for batch in dataloader_traces:
            opt_traces.zero_grad()
            recon = model_traces(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt_traces.step()

    model_traces.eval()
    errors_traces = []
    with torch.no_grad():
        for batch in dataloader_traces:
            recon = model_traces(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors_traces.append(mse)
    errors_traces = np.concatenate(errors_traces)

    threshold_traces = errors_traces.mean() + 3 * errors_traces.std()
    trace_df['trace_lstm'] = 0

    for idx in np.where(errors_traces > threshold_traces)[0]:
        trace_df.loc[idx:idx+9, 'trace_lstm'] = 1

    return metrics_df, log_df, event_df, trace_df
