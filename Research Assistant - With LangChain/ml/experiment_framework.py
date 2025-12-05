"""
PyTorch experiment framework for training and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Optional, Callable


class ExperimentFramework:
    """
    Base framework for running PyTorch experiments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        metrics_dir: str = "./results"
    ):
        """
        Initialize experiment framework.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            device: Device to use ('cuda' or 'cpu')
            checkpoint_dir: Directory for saving checkpoints
            metrics_dir: Directory for saving metrics
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters())
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": 100. * correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": 100. * correct / total
        }
    
    def test(self) -> Dict[str, float]:
        """Test the model."""
        if self.test_loader is None:
            return {"error": "No test loader provided"}
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            "loss": total_loss / len(self.test_loader),
            "accuracy": 100. * correct / total
        }
    
    def train(self, num_epochs: int, save_best: bool = True) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            
        Returns:
            Training metrics
        """
        best_val_acc = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.metrics["train_loss"].append(train_metrics["loss"])
            self.metrics["train_acc"].append(train_metrics["accuracy"])
            
            # Validate
            val_metrics = self.validate()
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["val_acc"].append(val_metrics["accuracy"])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if save_best and val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(f"best_model.pt", epoch, val_metrics["accuracy"])
                print(f"✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, val_metrics["accuracy"])
        
        # Save final metrics
        self.save_metrics()
        
        return self.metrics
    
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        with open(self.metrics_dir / filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved to {self.metrics_dir / filename}")
