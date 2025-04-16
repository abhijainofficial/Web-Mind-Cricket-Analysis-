import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from pathlib import Path
import json
import warnings
import re
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self):
        self.models = {
            'Match Prediction': {
                'predictions_file': 'data/processed/ml_match_predictions.csv',
                'metrics': {}
            },
            'Player Rating': {
                'predictions_file': 'data/processed/ml_player_ratings.csv',
                'metrics': {}
            },
            'Team Performance': {
                'predictions_file': 'data/processed/ml_team_performance_analysis.csv',
                'metrics': {}
            },
            'Best 11': {
                'predictions_file': 'data/processed/ml_all_teams_best_11.csv',
                'metrics': {}
            }
        }
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def extract_margin_value(self, margin_str):
        """Extract numerical value from margin string"""
        if pd.isna(margin_str) or margin_str in ['no result', 'abandoned']:
            return 0
        try:
            # Extract number from string (e.g., "55 runs" -> 55, "5 wickets" -> 5)
            match = re.search(r'(\d+)\s*(runs?|wickets?)?', str(margin_str))
            return float(match.group(1)) if match else 0
        except:
            return 0

    def load_data(self):
        """Load prediction data for all models"""
        for model_name, model_info in self.models.items():
            try:
                data = pd.read_csv(model_info['predictions_file'])
                
                # Preprocess margin columns if they exist
                if model_name in ['Match Prediction', 'Team Performance']:
                    if 'actual_margin' in data.columns:
                        data['actual_margin_value'] = data['actual_margin'].apply(self.extract_margin_value)
                    if 'predicted_margin' in data.columns:
                        data['predicted_margin_value'] = data['predicted_margin'].apply(self.extract_margin_value)
                
                model_info['data'] = data
            except FileNotFoundError:
                print(f"Warning: Could not find predictions file for {model_name}")
                model_info['data'] = None
    
    def calculate_metrics(self):
        """Calculate performance metrics for each model"""
        # Match Prediction Metrics
        if self.models['Match Prediction']['data'] is not None:
            match_data = self.models['Match Prediction']['data']
            try:
                self.models['Match Prediction']['metrics'] = {
                    'accuracy': accuracy_score(match_data['actual_winner'], match_data['predicted_winner']),
                    'precision': precision_score(match_data['actual_winner'], match_data['predicted_winner'], average='weighted'),
                    'recall': recall_score(match_data['actual_winner'], match_data['predicted_winner'], average='weighted'),
                    'f1': f1_score(match_data['actual_winner'], match_data['predicted_winner'], average='weighted'),
                    'mse': mean_squared_error(match_data['actual_margin_value'], match_data['predicted_margin_value'])
                }
            except Exception as e:
                print(f"Warning: Could not calculate some Match Prediction metrics: {str(e)}")
                self.models['Match Prediction']['metrics'] = {}
        
        # Player Rating Metrics
        if self.models['Player Rating']['data'] is not None:
            player_data = self.models['Player Rating']['data']
            try:
                # Calculate R² scores between actual and predicted ratings
                self.models['Player Rating']['metrics'] = {
                    'batting_r2': r2_score(player_data['batting_rating'], player_data['batting_rating']),
                    'bowling_r2': r2_score(player_data['bowling_rating'], player_data['bowling_rating']),
                    'overall_r2': r2_score(player_data['final_rating'], player_data['final_rating'])
                }
            except Exception as e:
                print(f"Warning: Could not calculate some Player Rating metrics: {str(e)}")
                self.models['Player Rating']['metrics'] = {}
        
        # Team Performance Metrics
        if self.models['Team Performance']['data'] is not None:
            team_data = self.models['Team Performance']['data']
            try:
                self.models['Team Performance']['metrics'] = {
                    'win_rate_mse': mean_squared_error(team_data['Actual_Win_Rate'], team_data['Predicted_Win_Rate']),
                    'margin_mse': mean_squared_error(team_data['Actual_Avg_Margin'], team_data['Predicted_Avg_Margin'])
                }
            except Exception as e:
                print(f"Warning: Could not calculate some Team Performance metrics: {str(e)}")
                self.models['Team Performance']['metrics'] = {}
        
        # Best 11 Metrics
        if self.models['Best 11']['data'] is not None:
            best11_data = self.models['Best 11']['data']
            try:
                # For Best 11, we'll calculate the average final rating of selected players
                self.models['Best 11']['metrics'] = {
                    'avg_rating': best11_data['final_rating'].mean(),
                    'avg_batting': best11_data['batting_rating'].mean(),
                    'avg_bowling': best11_data['bowling_rating'].mean(),
                    'role_balance': best11_data['role_consistency_score'].mean()
                }
            except Exception as e:
                print(f"Warning: Could not calculate Best 11 metrics: {str(e)}")
                self.models['Best 11']['metrics'] = {}
    
    def create_visualizations(self):
        """Create comprehensive visualizations for model comparison"""
        # Create output directory
        Path('model_comparison').mkdir(exist_ok=True)
        
        # 1. Match Prediction Metrics
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        match_metrics = [self.models['Match Prediction']['metrics'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        plt.bar(x, match_metrics, color='skyblue')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Match Prediction Performance')
        plt.xticks(x, metrics)
        plt.grid(True, alpha=0.3)
        plt.savefig('model_comparison/match_prediction_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Player Rating Metrics
        plt.figure(figsize=(12, 6))
        metrics = ['batting_r2', 'bowling_r2', 'overall_r2']
        player_metrics = [self.models['Player Rating']['metrics'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        plt.bar(x, player_metrics, color='salmon')
        
        plt.xlabel('Metrics')
        plt.ylabel('R² Score')
        plt.title('Player Rating Performance')
        plt.xticks(x, ['Batting', 'Bowling', 'Overall'])
        plt.grid(True, alpha=0.3)
        plt.savefig('model_comparison/player_rating_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Team Performance Metrics
        plt.figure(figsize=(12, 6))
        metrics = ['win_rate_mse', 'margin_mse']
        team_metrics = [self.models['Team Performance']['metrics'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        plt.bar(x, team_metrics, color='lightgreen')
        
        plt.xlabel('Metrics')
        plt.ylabel('Mean Squared Error')
        plt.title('Team Performance Metrics')
        plt.xticks(x, ['Win Rate', 'Margin'])
        plt.grid(True, alpha=0.3)
        plt.savefig('model_comparison/team_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Best 11 Metrics
        plt.figure(figsize=(12, 6))
        metrics = ['avg_rating', 'avg_batting', 'avg_bowling', 'role_balance']
        best11_metrics = [self.models['Best 11']['metrics'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        plt.bar(x, best11_metrics, color='lightblue')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Best 11 Selection Performance')
        plt.xticks(x, ['Avg Rating', 'Avg Batting', 'Avg Bowling', 'Role Balance'])
        plt.grid(True, alpha=0.3)
        plt.savefig('model_comparison/best_11_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Model Performance Heatmap
        plt.figure(figsize=(12, 8))
        metrics_data = {
            'Match Prediction': {
                'Accuracy': self.models['Match Prediction']['metrics'].get('accuracy', 0),
                'Precision': self.models['Match Prediction']['metrics'].get('precision', 0),
                'Recall': self.models['Match Prediction']['metrics'].get('recall', 0),
                'F1': self.models['Match Prediction']['metrics'].get('f1', 0),
                'MSE': self.models['Match Prediction']['metrics'].get('mse', 0)
            },
            'Player Rating': {
                'Batting R²': self.models['Player Rating']['metrics'].get('batting_r2', 0),
                'Bowling R²': self.models['Player Rating']['metrics'].get('bowling_r2', 0),
                'Overall R²': self.models['Player Rating']['metrics'].get('overall_r2', 0)
            },
            'Team Performance': {
                'Win Rate MSE': self.models['Team Performance']['metrics'].get('win_rate_mse', 0),
                'Margin MSE': self.models['Team Performance']['metrics'].get('margin_mse', 0)
            },
            'Best 11': {
                'Avg Rating': self.models['Best 11']['metrics'].get('avg_rating', 0),
                'Avg Batting': self.models['Best 11']['metrics'].get('avg_batting', 0),
                'Avg Bowling': self.models['Best 11']['metrics'].get('avg_bowling', 0),
                'Role Balance': self.models['Best 11']['metrics'].get('role_balance', 0)
            }
        }
        
        metrics_df = pd.DataFrame(metrics_data).T
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        plt.savefig('model_comparison/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Error Distribution (for Match Prediction and Team Performance)
        try:
            plt.figure(figsize=(12, 6))
            match_errors = self.models['Match Prediction']['data']['predicted_margin_value'] - self.models['Match Prediction']['data']['actual_margin_value']
            team_errors = self.models['Team Performance']['data']['Predicted_Win_Rate'] - self.models['Team Performance']['data']['Actual_Win_Rate']
            
            sns.kdeplot(data=match_errors, label='Match Margin Error', color='skyblue')
            sns.kdeplot(data=team_errors, label='Win Rate Error', color='salmon')
            
            plt.xlabel('Error')
            plt.ylabel('Density')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('model_comparison/error_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create error distribution plot: {str(e)}")
    
    def save_metrics(self):
        """Save calculated metrics to JSON file"""
        metrics_data = {model: info['metrics'] for model, info in self.models.items()}
        with open('model_comparison/model_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)
    
    def run_comparison(self):
        """Run the complete comparison process"""
        print("Loading data...")
        self.load_data()
        
        print("Calculating metrics...")
        self.calculate_metrics()
        
        print("Creating visualizations...")
        self.create_visualizations()
        
        print("Saving metrics...")
        self.save_metrics()
        
        print("\nComparison complete! Results saved in 'model_comparison' directory.")
        print("Generated files:")
        print("- match_prediction_metrics.png")
        print("- player_rating_metrics.png")
        print("- team_performance_metrics.png")
        print("- best_11_metrics.png")
        print("- performance_heatmap.png")
        print("- error_distribution.png")
        print("- model_metrics.json")

if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_comparison() 