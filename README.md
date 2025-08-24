# AI-Driven Low-Rate DDoS Detection in SDN

A comprehensive research prototype for detecting low-rate DDoS attacks in Software-Defined Networks using deep learning techniques with PyTorch and Ryu SDN controller.

## 🎯 Project Overview

This project implements an intelligent DDoS detection system that:

- Monitors network traffic in real-time using SDN capabilities
- Employs deep learning models to identify low-rate DDoS attacks
- Provides automated attack mitigation strategies
- Offers comprehensive visualization and analysis tools

## 🏗️ Architecture

```
├── src/
│   ├── sdn_controller/     # Ryu controller applications
│   ├── ml_models/          # PyTorch deep learning models
│   ├── detection/          # Attack detection algorithms
│   ├── utils/              # Utility functions
│   └── visualization/      # Data visualization tools
├── data/                   # Datasets and preprocessed data
├── models/                 # Trained model checkpoints
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── scripts/                # Setup and utility scripts
```

## 🚀 Features

- **Real-time Traffic Monitoring**: SDN-based network flow collection
- **Deep Learning Detection**: Advanced neural networks for attack identification
- **Low-rate DDoS Focus**: Specialized detection for stealthy attacks
- **Attack Mitigation**: Automated response mechanisms
- **Comprehensive Logging**: Detailed attack analysis and reporting
- **Visualization Dashboard**: Real-time network status and attack visualization
- **Research-oriented**: Extensive documentation and experimental capabilities

## 📋 Requirements

- Python 3.8+
- Ryu SDN Controller
- PyTorch
- Mininet (for network simulation)
- Linux environment (recommended)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd AI-DDoS-Detection-SDN
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Setup Mininet (Linux):

```bash
sudo apt-get install mininet
```

## 🎮 Quick Start

1. **Start the SDN controller**:

```bash
python src/sdn_controller/ddos_controller.py
```

2. **Run network simulation**:

```bash
sudo python scripts/start_simulation.py
```

3. **Train the detection model**:

```bash
python src/ml_models/train_model.py
```

4. **Start real-time detection**:

```bash
python src/detection/real_time_detector.py
```

## 📊 Dataset

The project supports multiple datasets:

- CICIDS2017/2018
- NSL-KDD
- Custom generated traffic data
- Real-time SDN flow data

## 🧠 Machine Learning Models

Implemented models include:

- **LSTM Networks**: For temporal pattern recognition
- **CNN-LSTM Hybrid**: Combined spatial-temporal analysis
- **Autoencoders**: For anomaly detection
- **Transformer Models**: For sequence analysis

## 🔧 Configuration

Main configuration files:

- `config/model_config.yaml`: ML model parameters
- `config/sdn_config.yaml`: SDN controller settings
- `config/detection_config.yaml`: Detection thresholds and parameters

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📈 Performance Metrics

The system evaluates performance using:

- Detection Accuracy
- False Positive Rate
- Detection Latency
- Throughput Impact
- Resource Utilization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Research Papers

This implementation is based on research in:

- Low-rate DDoS attack detection
- SDN-based network monitoring
- Deep learning for cybersecurity
- Network traffic analysis

## 🆘 Support

For questions and support:

- Create an issue in the repository
- Check the documentation in `docs/`
- Review example configurations

## 🔮 Future Work

- Integration with additional SDN controllers
- Real-world deployment capabilities
- Advanced visualization features
- Integration with network security frameworks
