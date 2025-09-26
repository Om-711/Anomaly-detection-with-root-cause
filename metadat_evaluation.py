import pandas as pd
import numpy as np

def making_metadata(metrics_df, log_df, event_df, trace_df):

    print("\nPreparing the metadata..........")
    # metric_anamoly = metrics_df[metrics_df[['anomaly_iso','anomaly_svm','cpu_anomaly_stat','anomaly_lstm']].any(axis=1)]
    metric_anamoly = metrics_df[metrics_df[['metrics_iso','metrics_stat','metrics_lstm', 'metrics_svm']].any(axis=1)]

    metric_metadata = []
    for _, row in metric_anamoly.iterrows():
        metadata = {
            "timestamp": row['timestamp'],
            "service": row.get('service','unknown'),  
            "component": "metric",
            "feature": "cpu_usage/mem_usage/response_time",  
            "value": {
                "cpu": row['cpu_usage'],
                "memory": row['mem_base'],
                "response_time": row['response_time']
            },
            "anomaly_type": "spike"
        }
        metric_metadata.append(metadata)

    log_summary = log_df.groupby('service')['level'].apply(lambda x: (x=='ERROR').sum())
    threshold = log_summary.mean() + 3*log_summary.std()
    top_logs = log_summary[log_summary > threshold]
    for service, count in top_logs.items():
        metadata = {
            "timestamp": pd.Timestamp.now(),
            "service": service,
            "component": "log",
            "feature": {"error_count": int(count)},
            "anomaly_type": "error_spike"
        }
        metric_metadata.append(metadata)

    # Traces: long durations
    trace_anom = trace_df[trace_df['duration_ms'] > 45]
    for _, row in trace_anom.iterrows():
        metadata = {
            "timestamp": row['start_time'],
            "service": row['service'],
            "component": "trace",
            "feature": {"duration_ms": row['duration_ms']},
            "anomaly_type": "long_span"
        }
        metric_metadata.append(metadata)


    # Events: unusual frequency
    event_counts = event_df.groupby(['service','event_type']).size()
    threshold = event_counts.mean() + 3*event_counts.std()
    for (service, e_type), count in event_counts.items():
        if count > threshold:
            metadata = {
                "timestamp": pd.Timestamp.now(),
                "service": service,
                "component": "event",
                "feature": {"event_count": int(count), "event_type": e_type},
                "anomaly_type": "event_spike"
            }

            metric_metadata.append(metadata)

    return metric_metadata


def evaluate_all_methods_melt(metrics_df, log_df, event_df, traces_df):
    """
    Evaluate anomaly detection performance for all MELT components.
    Each df should have:
        - 'is_anomaly' column (ground truth)
        - prediction columns: *_iso, *_stats, *_lstm, *_svm
    """
    print("Started Evaluation1.......")
    results = {}

    methods = ['iso', 'stat', 'lstm', 'svm']

    # ---- METRICS ----
    for method in methods:
        col_name = f"metrics_{method}"
        if col_name in metrics_df.columns:
            TP = ((metrics_df['is_anomaly'] == 1) & (metrics_df[col_name] == 1)).sum()
            FP = ((metrics_df['is_anomaly'] == 0) & (metrics_df[col_name] == 1)).sum()
            FN = ((metrics_df['is_anomaly'] == 1) & (metrics_df[col_name] == 0)).sum()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[col_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }

    print("Started Evaluation2.......")
    # ---- LOGS ----
    for method in methods:
        col_name = f"log_{method}"
        if col_name in log_df.columns:
            TP = ((log_df['is_anomaly'] == 1) & (log_df[col_name] == 1)).sum()
            FP = ((log_df['is_anomaly'] == 0) & (log_df[col_name] == 1)).sum()
            FN = ((log_df['is_anomaly'] == 1) & (log_df[col_name] == 0)).sum()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[col_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }

    print("Started Evaluation3.......")
    # ---- TRACES ----
    for method in methods:
        col_name = f"trace_{method}"
        if col_name in traces_df.columns:
            TP = ((traces_df['is_anomaly'] == 1) & (traces_df[col_name] == 1)).sum()
            FP = ((traces_df['is_anomaly'] == 0) & (traces_df[col_name] == 1)).sum()
            FN = ((traces_df['is_anomaly'] == 1) & (traces_df[col_name] == 0)).sum()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[col_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }

    print("Started Evaluation4.......")
    # ---- EVENTS ----
    for method in methods:
        col_name = f"event_{method}"
        if col_name in event_df.columns:
            TP = ((event_df['is_anomaly'] == 1) & (event_df[col_name] == 1)).sum()
            FP = ((event_df['is_anomaly'] == 0) & (event_df[col_name] == 1)).sum()
            FN = ((event_df['is_anomaly'] == 1) & (event_df[col_name] == 0)).sum()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[col_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }

    return results


