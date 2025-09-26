# Anomaly Detection with Root Cause Analysis

## Overview

An end-to-end solution for detecting anomalies in microservices architectures and identifying their root causes. This system integrates MELT data (Metrics, Events, Logs, Traces) using advanced machine learning techniques to provide proactive system monitoring and intelligent anomaly explanation.

## Key Features

### Multi-Model Anomaly Detection
- Isolation Forest: Detects outliers in high-dimensional data
- Z-Score Analysis: Identifies statistical anomalies  
- LSTM Autoencoders: Captures temporal patterns and sequences
- Ensemble Methods: Combines multiple models for improved accuracy

### Comprehensive Root Cause Analysis
- Cross-Data Correlation: Links anomalies across metrics, logs, events, and traces
- Dependency Mapping: Understands service interdependencies
- Temporal Analysis: Identifies causation chains through time-based correlation

### Intelligent Metadata Generation
- Structured Output: JSON-formatted anomaly summaries
- Context-Aware Descriptions: Meaningful explanations for each anomaly
- Severity Classification: Risk assessment and priority scoring

### Enterprise-Ready Scalability
- High-Volume Processing: Handles massive MELT data streams
- Distributed Architecture: Scales horizontally across multiple nodes
- Real-Time Processing: Low-latency anomaly detection

## Directory Structure

```
Anomaly-detection-with-root-cause/
├── anomaly_detection.py       # Core anomaly detection algorithms
├── event_queue.py            # Event queue processing and management
├── generate_data.py          # Synthetic MELT data generation
├── llm_explanation.py        # LLM integration for anomaly explanation
├── metadata_evaluation.py    # Metadata quality assessment
├── requirements.txt          # Python package dependencies
├── Dockerfile               # Container configuration
└── README.md               # Project documentation
```


## Installation

### Method 1: Local Installation

1. Clone the repository
```bash
git clone https://github.com/Om-711/Anomaly-detection-with-root-cause.git
cd Anomaly-detection-with-root-cause
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Method 2: Docker Installation

1. Clone the repository
```bash
git clone https://github.com/Om-711/Anomaly-detection-with-root-cause.git
cd Anomaly-detection-with-root-cause
```

2. Build the Docker image
```bash
docker build -t anomaly-detection .
```

3. Run the container
```bash
docker run -p 5000:5000 anomaly-detection
```

## Usage

### Quick Start

Generate synthetic test data and run anomaly detection:

```bash
# Generate sample MELT data
python generate_data.py

# Run anomaly detection
python anomaly_detection.py

# Evaluate results
python metadata_evaluation.py
python event_queue.py
```

### Detailed Usage

#### 1. Data Generation
Create synthetic MELT data for testing and development:

```bash
python generate_data.py 
```
#### 2. Anomaly Detection
Process data and detect anomalies:

```bash
python anomaly_detection.py 
```

#### 3. Metadata Evaluation
Assess the quality of generated metadata:

```bash
python metadata_evaluation.py 
```
#### 4. Root-Cause LLM explanation

```bash
python event_queue.py
```
## Example Output

The system generates structured metadata for detected anomalies:

```json
{
  "timestamp": "2025-09-26 00:10:00",
  "service": "auth-service",
  "component": "metric",
  "feature": "cpu_usage/mem_usage/response_time",
  "value": 69,
  "anomaly_type": "spike"
}

```

The LLM will then generates the root cause for the metadata:
```json
{'root_cause': 'Widespread resource contention and performance bottlenecks across core services (auth, cache, database, frontend), likely triggered by increased workload, inefficient code/queries, or underlying infrastructure limitations.',
 'severity': 'High',
 'suggested_action': 'Immediately review traffic patterns and recent deployments. Deep-dive into database and cache performance metrics, and consider scaling resources or optimizing application code.'}
```
## Troubleshooting

### Common Issues
**Low Detection Accuracy**
- Retrain models with more data: 

**LLM Integration Failures**
- Check API key configuration
- Verify network connectivity
- Review API rate limits

### Debug Mode

Run in debug mode for detailed troubleshooting:

```bash
python anomaly_detection.py --debug --verbose
```



