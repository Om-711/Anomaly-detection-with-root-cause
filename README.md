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

## Prerequisites

Before installing and running this project, ensure you have:

- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)
- Minimum 4GB RAM recommended
- Network access for LLM integration

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
```

### Detailed Usage

#### 1. Data Generation
Create synthetic MELT data for testing and development:

```bash
python generate_data.py --services 5 --duration 24 --interval 1
```

Options:
- `--services`: Number of microservices to simulate (default: 3)
- `--duration`: Duration in hours (default: 12)
- `--interval`: Data point interval in minutes (default: 1)

#### 2. Anomaly Detection
Process data and detect anomalies:

```bash
python anomaly_detection.py --input data/ --output results/ --models all
```

Options:
- `--input`: Input data directory
- `--output`: Output directory for results
- `--models`: Models to use (isolation, zscore, lstm, all)
- `--threshold`: Anomaly threshold (default: 0.95)

#### 3. Metadata Evaluation
Assess the quality of generated metadata:

```bash
python metadata_evaluation.py --results results/ --metrics accuracy,precision,recall
```

## Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
export ANOMALY_THRESHOLD=0.95
export LLM_API_KEY=your_api_key_here
export LOG_LEVEL=INFO
export BATCH_SIZE=1000
```

### Configuration File

Create a `config.yaml` file for advanced configuration:

```yaml
detection:
  models:
    isolation_forest:
      contamination: 0.1
      n_estimators: 100
    zscore:
      threshold: 3.0
    lstm:
      sequence_length: 50
      epochs: 100

processing:
  batch_size: 1000
  parallel_workers: 4
  
output:
  format: json
  include_explanations: true
```

## Example Output

The system generates structured metadata for detected anomalies:

```json
{
  "timestamp": "2024-09-26T10:30:00Z",
  "service": "auth-service",
  "component": "log",
  "feature": "ERROR_count",
  "anomaly_details": {
    "value": 50,
    "expected_range": "5-15",
    "anomaly_type": "unusual_frequency",
    "severity": "high",
    "confidence": 0.87
  },
  "root_cause_analysis": {
    "correlated_anomalies": [
      {
        "service": "database-service",
        "metric": "connection_timeout",
        "correlation_score": 0.94
      }
    ],
    "suggested_causes": [
      "Database connection pool exhaustion",
      "Network latency spike",
      "Service overload"
    ]
  },
  "explanation": "Authentication service showing 3x higher error rate than normal, strongly correlated with database timeout issues."
}
```
## Troubleshooting

### Common Issues

**Memory Errors with Large Datasets**
- Reduce batch size: `--batch-size 500`


**Low Detection Accuracy**
- Adjust thresholds:
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

## Contributing

We welcome contributions to improve this project. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black anomaly_detection.py
flake8 anomaly_detection.py
```

