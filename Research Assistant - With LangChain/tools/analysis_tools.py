"""
Analysis tools for analyzing experiment results and generating insights.
"""
import json
import logging
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_results(results_json: str) -> str:
    """
    Perform statistical analysis on experiment results.
    
    Args:
        results_json: JSON string with experiment results
        
    Returns:
        Analysis summary
    """
    try:
        results = json.loads(results_json)
        logger.info(f"Analyzing results for: {results.get('experiment_name', 'unknown')}")
        
        analysis = {
            "experiment": results.get('experiment_name', 'unknown'),
            "performance": {
                "test_accuracy": results.get('test_acc', 0),
                "test_loss": results.get('test_loss', 0),
                "best_val_accuracy": results.get('best_val_acc', 0)
            },
            "training_quality": {
                "final_train_acc": results.get('final_train_acc', 0),
                "final_val_acc": results.get('final_val_acc', 0),
                "overfitting": results.get('final_train_acc', 0) - results.get('final_val_acc', 0)
            },
            "insights": []
        }
        
        # Generate insights
        if analysis['training_quality']['overfitting'] > 10:
            analysis['insights'].append("Model shows signs of overfitting (train-val gap > 10%)")
        
        if analysis['performance']['test_accuracy'] > 90:
            analysis['insights'].append("Excellent performance (>90% test accuracy)")
        elif analysis['performance']['test_accuracy'] > 80:
            analysis['insights'].append("Good performance (>80% test accuracy)")
        else:
            analysis['insights'].append("Performance could be improved (<80% test accuracy)")
        
        logger.info(f"Analysis complete: {len(analysis['insights'])} insights generated")
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        return json.dumps({"error": str(e)})


def generate_plots(data_json: str, plot_type: str = "training_curves") -> str:
    """
    Generate visualization plots for experiment data.
    
    Args:
        data_json: JSON string with data to plot
        plot_type: Type of plot (training_curves, comparison, confusion_matrix)
        
    Returns:
        Path to saved plot
    """
    try:
        data = json.loads(data_json)
        logger.info(f"Generating {plot_type} plot")
        
        # Create output directory
        output_dir = Path("./outputs/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == "training_curves":
            # Plot training and validation curves
            if 'train_acc' in data and 'val_acc' in data:
                epochs = range(1, len(data['train_acc']) + 1)
                plt.plot(epochs, data['train_acc'], 'b-', label='Training Accuracy')
                plt.plot(epochs, data['val_acc'], 'r-', label='Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Training and Validation Accuracy')
                plt.legend()
                plt.grid(True)
            else:
                # Placeholder plot
                plt.text(0.5, 0.5, 'Training curves plot\n(Provide train_acc and val_acc data)',
                        ha='center', va='center', fontsize=12)
        
        elif plot_type == "comparison":
            # Bar plot for comparing experiments
            if 'experiments' in data:
                names = [exp['name'] for exp in data['experiments']]
                accuracies = [exp['accuracy'] for exp in data['experiments']]
                plt.bar(names, accuracies)
                plt.xlabel('Experiment')
                plt.ylabel('Accuracy (%)')
                plt.title('Experiment Comparison')
                plt.xticks(rotation=45, ha='right')
            else:
                plt.text(0.5, 0.5, 'Comparison plot\n(Provide experiments data)',
                        ha='center', va='center', fontsize=12)
        
        else:
            plt.text(0.5, 0.5, f'Plot type: {plot_type}\n(Not yet implemented)',
                    ha='center', va='center', fontsize=12)
        
        # Save plot
        plot_path = output_dir / f"{plot_type}_{np.random.randint(1000, 9999)}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to: {plot_path}")
        return str(plot_path)
        
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        return f"Error: {str(e)}"


def identify_trends(experiments_json: str) -> str:
    """
    Identify trends across multiple experiments.
    
    Args:
        experiments_json: JSON string with list of experiments
        
    Returns:
        Trends summary
    """
    try:
        experiments = json.loads(experiments_json)
        logger.info(f"Identifying trends across {len(experiments)} experiments")
        
        if not experiments:
            return "No experiments provided"
        
        # Extract metrics
        accuracies = [exp.get('test_acc', 0) for exp in experiments]
        model_types = [exp.get('model_type', 'unknown') for exp in experiments]
        
        # Identify trends
        trends = {
            "total_experiments": len(experiments),
            "average_accuracy": np.mean(accuracies),
            "best_accuracy": max(accuracies),
            "worst_accuracy": min(accuracies),
            "model_performance": {}
        }
        
        # Performance by model type
        for model_type in set(model_types):
            model_accs = [exp['test_acc'] for exp in experiments 
                         if exp.get('model_type') == model_type]
            if model_accs:
                trends['model_performance'][model_type] = {
                    "average": np.mean(model_accs),
                    "count": len(model_accs)
                }
        
        # Find best model
        if trends['model_performance']:
            best_model = max(trends['model_performance'].items(), 
                           key=lambda x: x[1]['average'])
            trends['best_model_type'] = best_model[0]
            trends['best_model_avg_acc'] = best_model[1]['average']
        
        logger.info(f"Trends identified: Best model = {trends.get('best_model_type', 'N/A')}")
        return json.dumps(trends, indent=2)
        
    except Exception as e:
        logger.error(f"Error identifying trends: {str(e)}")
        return json.dumps({"error": str(e)})


def generate_insights(analysis_data: str) -> str:
    """
    Generate high-level insights from analysis data.
    
    Args:
        analysis_data: JSON string with analysis results
        
    Returns:
        Insights summary
    """
    try:
        data = json.loads(analysis_data)
        logger.info("Generating insights from analysis data")
        
        insights = {
            "summary": "Analysis Insights",
            "key_findings": [],
            "recommendations": []
        }
        
        # Extract key findings
        if 'average_accuracy' in data:
            avg_acc = data['average_accuracy']
            insights['key_findings'].append(
                f"Average accuracy across experiments: {avg_acc:.2f}%"
            )
            
            if avg_acc > 85:
                insights['recommendations'].append(
                    "Models are performing well. Consider fine-tuning hyperparameters for marginal gains."
                )
            else:
                insights['recommendations'].append(
                    "Performance could be improved. Consider trying different architectures or data augmentation."
                )
        
        if 'best_model_type' in data:
            insights['key_findings'].append(
                f"Best performing model type: {data['best_model_type']} "
                f"({data.get('best_model_avg_acc', 0):.2f}% avg accuracy)"
            )
        
        if not insights['key_findings']:
            insights['key_findings'].append("Insufficient data for detailed insights")
        
        logger.info(f"Generated {len(insights['key_findings'])} findings and "
                   f"{len(insights['recommendations'])} recommendations")
        return json.dumps(insights, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return json.dumps({"error": str(e)})


def statistical_comparison(exp1_json: str, exp2_json: str) -> str:
    """
    Perform statistical comparison between two experiments.
    
    Args:
        exp1_json: JSON string with first experiment results
        exp2_json: JSON string with second experiment results
        
    Returns:
        Comparison summary
    """
    try:
        exp1 = json.loads(exp1_json)
        exp2 = json.loads(exp2_json)
        
        logger.info(f"Comparing {exp1.get('experiment_name', 'exp1')} vs "
                   f"{exp2.get('experiment_name', 'exp2')}")
        
        comparison = {
            "experiment_1": exp1.get('experiment_name', 'exp1'),
            "experiment_2": exp2.get('experiment_name', 'exp2'),
            "metrics": {
                "test_accuracy": {
                    "exp1": exp1.get('test_acc', 0),
                    "exp2": exp2.get('test_acc', 0),
                    "difference": exp1.get('test_acc', 0) - exp2.get('test_acc', 0)
                },
                "test_loss": {
                    "exp1": exp1.get('test_loss', 0),
                    "exp2": exp2.get('test_loss', 0),
                    "difference": exp1.get('test_loss', 0) - exp2.get('test_loss', 0)
                }
            }
        }
        
        # Determine winner
        if comparison['metrics']['test_accuracy']['difference'] > 1:
            comparison['winner'] = comparison['experiment_1']
            comparison['conclusion'] = f"{comparison['experiment_1']} performs better"
        elif comparison['metrics']['test_accuracy']['difference'] < -1:
            comparison['winner'] = comparison['experiment_2']
            comparison['conclusion'] = f"{comparison['experiment_2']} performs better"
        else:
            comparison['winner'] = "tie"
            comparison['conclusion'] = "Performance is similar"
        
        logger.info(f"Comparison complete: {comparison['conclusion']}")
        return json.dumps(comparison, indent=2)
        
    except Exception as e:
        logger.error(f"Error in statistical comparison: {str(e)}")
        return json.dumps({"error": str(e)})
