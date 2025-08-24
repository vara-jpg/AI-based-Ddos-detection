#!/usr/bin/env python3
"""
Model Training Script for DDoS Detection
Comprehensive training pipeline with validation and performance metrics
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.lstm_model import LSTMDDoSDetector, CNNLSTMModel, AutoEncoder
from utils.data_loader import DDoSDataLoader
from utils.logger_config import setup_logger
from utils.performance_metrics import ModelPerformanceTracker


class DDoSModelTrainer:
    """
    Comprehensive trainer for DDoS detection models
    """

    def __init__(self, config_path="config/model_config.yaml"):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging
        self.logger = setup_logger("ModelTrainer", self.config.get("logging", {}))
        self.logger.info(f"Using device: {self.device}")

        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scaler = None

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Performance tracking
        self.performance_tracker = ModelPerformanceTracker()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.logger.info("DDoS Model Trainer initialized")

    def _load_config(self, config_path):
        """Load training configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self):
        """Default configuration"""
        return {
            "model": {
                "name": "LSTM_DDoS_Detector",
                "type": "lstm",
                "input_size": 20,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": 2,
                "dropout": 0.2,
                "bidirectional": True,
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 100,
                "weight_decay": 0.00001,
                "sequence_length": 10,
            },
            "checkpoint": {
                "save_path": "models/",
                "save_frequency": 10,
                "best_model_path": "models/best_model.pth",
            },
        }

    def load_data(self, data_path="data/", dataset_name="synthetic"):
        """Load and preprocess training data"""
        self.logger.info("Loading training data...")

        # Create data loader
        data_loader = DDoSDataLoader(self.config, self.logger)

        if dataset_name == "synthetic":
            # Generate synthetic data for demonstration
            X, y = self._generate_synthetic_data()
        else:
            # Load real dataset (implement based on your dataset)
            X, y = data_loader.load_dataset(data_path, dataset_name)

        # Preprocess data
        X_processed, y_processed = self._preprocess_data(X, y)

        # Split data
        train_data, val_data, test_data = self._split_data(X_processed, y_processed)

        # Create data loaders
        self._create_data_loaders(train_data, val_data, test_data)

        self.logger.info(f"Data loaded successfully:")
        self.logger.info(f"  Training samples: {len(train_data[0])}")
        self.logger.info(f"  Validation samples: {len(val_data[0])}")
        self.logger.info(f"  Test samples: {len(test_data[0])}")

    def _generate_synthetic_data(self, num_samples=10000):
        """Generate synthetic network traffic data"""
        self.logger.info("Generating synthetic data...")

        np.random.seed(42)
        input_size = self.config["model"]["input_size"]
        sequence_length = self.config["model"]["sequence_length"]

        # Generate normal traffic patterns
        normal_samples = num_samples // 2
        normal_data = []

        for _ in range(normal_samples):
            # Normal traffic: steady flow with small variations
            base_values = np.random.normal(0.5, 0.1, input_size)
            sequence = []
            for t in range(sequence_length):
                # Add temporal variation
                noise = np.random.normal(0, 0.05, input_size)
                values = np.clip(base_values + noise, 0, 1)
                sequence.append(values)
            normal_data.append(sequence)

        # Generate attack traffic patterns
        attack_samples = num_samples - normal_samples
        attack_data = []

        for _ in range(attack_samples):
            # Attack traffic: irregular patterns, higher variance
            base_values = np.random.normal(0.7, 0.2, input_size)
            sequence = []
            for t in range(sequence_length):
                # Add attack-like patterns
                if t > sequence_length // 2:  # Attack intensifies
                    noise = np.random.normal(0.1, 0.15, input_size)
                else:
                    noise = np.random.normal(0, 0.1, input_size)
                values = np.clip(base_values + noise, 0, 1)
                sequence.append(values)
            attack_data.append(sequence)

        # Combine data
        X = np.array(normal_data + attack_data)
        y = np.array([0] * normal_samples + [1] * attack_samples)

        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        self.logger.info(f"Generated {len(X)} synthetic samples")
        return X, y

    def _preprocess_data(self, X, y):
        """Preprocess the data"""
        self.logger.info("Preprocessing data...")

        # Normalize features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)

        X_processed = X_normalized.reshape(original_shape)

        return X_processed, y

    def _split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        train_split = self.config["model"].get("train_split", 0.7)
        val_split = self.config["model"].get("val_split", 0.2)

        n_total = len(X)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        # Split data
        X_train = X[:n_train]
        y_train = y[:n_train]

        X_val = X[n_train : n_train + n_val]
        y_val = y[n_train : n_train + n_val]

        X_test = X[n_train + n_val :]
        y_test = y[n_train + n_val :]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _create_data_loaders(self, train_data, val_data, test_data):
        """Create PyTorch data loaders"""
        batch_size = self.config["model"]["batch_size"]

        # Convert to tensors
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    def build_model(self):
        """Build the model based on configuration"""
        model_type = self.config["model"]["type"]
        model_params = self.config["model"]

        if model_type == "lstm":
            self.model = LSTMDDoSDetector(
                input_size=model_params["input_size"],
                hidden_size=model_params["hidden_size"],
                num_layers=model_params["num_layers"],
                output_size=model_params["output_size"],
                dropout=model_params["dropout"],
                bidirectional=model_params["bidirectional"],
            )
        elif model_type == "cnn_lstm":
            self.model = CNNLSTMModel(
                input_size=model_params["input_size"],
                lstm_hidden_size=model_params["hidden_size"],
                num_layers=model_params["num_layers"],
                output_size=model_params["output_size"],
                dropout=model_params["dropout"],
            )
        elif model_type == "autoencoder":
            self.model = AutoEncoder(
                input_size=model_params["input_size"], dropout=model_params["dropout"]
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)

        # Setup optimizer
        print("DEBUG weight_decay:", model_params["weight_decay"])
        assert isinstance(
            model_params["weight_decay"], (float, int)
        ), "weight_decay must be a number"
        assert model_params["weight_decay"] >= 0.0, "weight_decay must be >= 0.0"

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_params["learning_rate"],
            weight_decay=model_params["weight_decay"],
        )

        # Setup criterion
        self.criterion = nn.CrossEntropyLoss()

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.logger.info(f"Model built: {model_params['name']}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        # Calculate additional metrics
        precision = precision_score(all_targets, all_preds, average="weighted")
        recall = recall_score(all_targets, all_preds, average="weighted")
        f1 = f1_score(all_targets, all_preds, average="weighted")

        return avg_loss, accuracy, precision, recall, f1

    def train(self):
        """Main training loop"""
        epochs = self.config["model"]["epochs"]
        best_val_acc = 0.0

        self.logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Track performance
            self.performance_tracker.update_metrics(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            epoch_time = time.time() - start_time

            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Val F1: {val_f1:.4f} | Time: {epoch_time:.2f}s"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model("best")
                self.logger.info(
                    f"New best model saved with validation accuracy: {val_acc:.2f}%"
                )

            # Save checkpoint
            if (epoch + 1) % self.config["checkpoint"]["save_frequency"] == 0:
                self._save_model(f"checkpoint_epoch_{epoch+1}")

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    def test(self):
        """Test the model"""
        self.logger.info("Testing model...")

        self.model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = 100.0 * correct / total
        precision = precision_score(all_targets, all_preds, average="weighted")
        recall = recall_score(all_targets, all_preds, average="weighted")
        f1 = f1_score(all_targets, all_preds, average="weighted")

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Log results
        self.logger.info("Test Results:")
        self.logger.info(f"  Accuracy: {accuracy:.2f}%")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1-Score: {f1:.4f}")

        # Save results
        test_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds,
            "targets": all_targets,
            "probabilities": all_probs,
        }

        self._save_test_results(test_results)
        self._plot_confusion_matrix(cm)

        return test_results

    def _save_model(self, name):
        """Save model checkpoint"""
        os.makedirs(self.config["checkpoint"]["save_path"], exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "scaler": self.scaler,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

        if name == "best":
            path = self.config["checkpoint"]["best_model_path"]
        else:
            path = os.path.join(self.config["checkpoint"]["save_path"], f"{name}.pth")

        torch.save(checkpoint, path)

    def _save_test_results(self, results):
        """Save test results"""
        import json
        import numpy as np

        # Helper to convert numpy/tensor types to Python types
        def make_serializable(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            return val

        # Recursively convert all values
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return make_serializable(obj)

        results_serializable = convert(results)

        os.makedirs("logs", exist_ok=True)
        with open("logs/test_results.json", "w") as f:
            json.dump(results_serializable, f, indent=2)


def main():
    """Main training function"""
    trainer = DDoSModelTrainer()

    # Load data
    trainer.load_data()

    # Build model
    trainer.build_model()

    # Train model
    trainer.train()

    # Test model
    test_results = trainer.test()

    # Plot training history
    trainer.plot_training_history()

    print("Training completed successfully!")
    print(f"Final test accuracy: {test_results['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
