from queue import Queue
from llm_explanation import get_cause_llm
from anamoly_detection import detect_anomalies_melt
from metadat_evaluation import making_metadata, evaluate_all_methods_melt
import json
from generate_data import generate_events, generate_logs, generate_metrics, generate_traces


event_queue = Queue()
candidate_results = []

metric_df = generate_metrics()
log_df = generate_logs()
event_df = generate_events()
trace_df = generate_traces()

metric_df, log_df, event_df, trace_df = detect_anomalies_melt(metric_df, log_df, event_df, trace_df)

metric_metadata = making_metadata(metric_df, log_df, event_df, trace_df)

results = evaluate_all_methods_melt(metric_df, log_df, event_df, trace_df)

i = 0
for method, metrics in results.items():
    if i % 4 ==0:
        print("\n")
    i += 1
    print(f"{method} -> {metrics}")


for anomaly in metric_metadata[:5]:
    event_queue.put(anomaly)

# mem_bytes = metric_df.memory_usage(deep=True).sum()
# mem_kb = mem_bytes / 1024
# print(f"DataFrame size: {mem_kb/1024:.2f} MB")


while not event_queue.empty():
    anomaly = event_queue.get()
    print(event_queue.qsize())
    candidate_json = get_cause_llm(anomaly)   
    candidate_results.append(candidate_json)

print(json.dumps(candidate_results, indent=2))


