"""
Experiment tools for designing and running ML experiments.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from ml.models import SimpleLSTM, SimpleGRU, SimpleCNN, SimpleMLP, SimpleTransformer
from ml.experiment_framework import ExperimentFramework
from ml.data_pipeline import get_data_loaders
from config import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def design_experiment(description: str) -> str:
    """
    Design an ML experiment based on a description.
    
    Args:
        description: Natural language description of the experiment
        
    Returns:
        JSON string with experiment configuration
    """
    logger.info(f"Designing experiment: {description}")
    
    # Parse description to extract key components
    description_lower = description.lower()
    
    # Determine model type
    if "lstm" in description_lower:
        model_type = "lstm"
    elif "gru" in description_lower:
        model_type = "gru"
    elif "cnn" in description_lower:
        model_type = "cnn"
    elif "transformer" in description_lower:
        model_type = "transformer"
    else:
        model_type = "mlp"
    
    # Determine dataset
    if "mnist" in description_lower:
        dataset = "mnist"
    elif "cifar" in description_lower:
        dataset = "cifar10"
    else:
        dataset = "sequence"
    
    # Determine dataset size
    if "small" in description_lower:
        train_size = 1000
    elif "medium" in description_lower:
        train_size = 5000
    else:
        train_size = None  # Use full dataset
    
    # Create experiment config
    experiment_config = {
        "name": f"{model_type}_{dataset}_experiment",
        "description": description,
        "model": {
            "type": model_type,
            "params": {
                "input_size": 28 * 28 if dataset == "mnist" else 32 * 32 * 3 if dataset == "cifar10" else 1,
                "hidden_size": 128,
                "num_layers": 2,
                "num_classes": 10,
                "dropout": 0.2
            }
        },
        "dataset": {
            "name": dataset,
            "batch_size": 32,
            "train_size": train_size
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adam"
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    logger.info(f"Experiment designed: {experiment_config['name']}")
    return json.dumps(experiment_config, indent=2)


def run_experiment(config_json: str) -> str:
    """
    Run an ML experiment based on configuration.
    
    Args:
        config_json: JSON string with experiment configuration
        
    Returns:
        JSON string with experiment results
    """
    try:
        config_dict = json.loads(config_json)
        logger.info(f"Running experiment: {config_dict['name']}")
        
        # Get data loaders
        dataset_name = config_dict['dataset']['name']
        batch_size = config_dict['dataset']['batch_size']
        
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name,
            batch_size=batch_size
        )
        
        # Limit dataset size if specified
        if config_dict['dataset'].get('train_size'):
            # This is simplified - in practice you'd subsample the dataset
            logger.info(f"Using subset of {config_dict['dataset']['train_size']} samples")
        
        # Create model
        model_type = config_dict['model']['type']
        model_params = config_dict['model']['params']
        
        if model_type == "lstm":
            model = SimpleLSTM(**model_params)
        elif model_type == "gru":
            model = SimpleGRU(**model_params)
        elif model_type == "cnn":
            model = SimpleCNN(
                num_classes=model_params['num_classes'],
                input_channels=1 if dataset_name == "mnist" else 3
            )
        elif model_type == "mlp":
            model = SimpleMLP(
                input_size=model_params['input_size'],
                hidden_sizes=[model_params['hidden_size']] * model_params['num_layers'],
                num_classes=model_params['num_classes'],
                dropout=model_params['dropout']
            )
        elif model_type == "transformer":
            model = SimpleTransformer(
                input_size=model_params['input_size'],
                num_classes=model_params['num_classes']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create experiment framework
        framework = ExperimentFramework(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=config_dict['device']
        )
        
        # Train model
        epochs = config_dict['training']['epochs']
        metrics = framework.train(num_epochs=epochs, save_best=True)
        
        # Test model
        test_results = framework.test()
        
        # Prepare results
        results = {
            "experiment_name": config_dict['name'],
            "model_type": model_type,
            "dataset": dataset_name,
            "final_train_acc": metrics['train_acc'][-1] if metrics['train_acc'] else 0,
            "final_val_acc": metrics['val_acc'][-1] if metrics['val_acc'] else 0,
            "test_acc": test_results['accuracy'],
            "test_loss": test_results['loss'],
            "epochs_trained": epochs,
            "best_val_acc": max(metrics['val_acc']) if metrics['val_acc'] else 0
        }
        
        logger.info(f"Experiment completed: Test Acc = {results['test_acc']:.2f}%")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        return json.dumps({"error": str(e)})


def compare_experiments(experiment_ids: str) -> str:
    """
    Compare multiple experiments.
    
    Args:
        experiment_ids: Comma-separated list of experiment IDs
        
    Returns:
        Comparison summary
    """
    ids = [id.strip() for id in experiment_ids.split(',')]
    logger.info(f"Comparing experiments: {ids}")
    
    # This is a placeholder - in practice, you'd retrieve from memory
    comparison = f"Comparison of {len(ids)} experiments:\n"
    comparison += f"Experiment IDs: {', '.join(ids)}\n"
    comparison += "\nNote: Implement experiment retrieval from memory for full comparison."
    
    return comparison


def evaluate_model(model_path: str, dataset: str) -> str:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model_path: Path to saved model checkpoint
        dataset: Dataset name to evaluate on
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating model: {model_path} on {dataset}")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get data loaders
        _, _, test_loader = get_data_loaders(dataset_name=dataset, batch_size=32)
        
        # Evaluation would happen here
        # This is simplified for demonstration
        
        results = {
            "model_path": model_path,
            "dataset": dataset,
            "status": "Evaluation placeholder - implement full evaluation logic"
        }
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return json.dumps({"error": str(e)})
