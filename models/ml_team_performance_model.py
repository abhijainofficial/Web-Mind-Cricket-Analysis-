import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import pathlib

class MLTeamPerformanceModel:
    def __init__(self):
        self.data = None
        self.win_rate_model = None
        self.margin_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load match data and filter out no-result/abandoned matches"""
        data_path = pathlib.Path('data/processed/matches.csv')
        self.data = pd.read_csv(data_path)
        self.data = self.data[~self.data['winner'].isin(['no result', 'abandoned'])]
        print("Data loaded successfully")
        
    def prepare_features(self):
        """Prepare features for model training"""
        teams = pd.concat([self.data['team1'], self.data['team2']]).unique()
        features = []
        labels_win_rate = []
        labels_margin = []
        
        for team in teams:
            team_stats = self.get_team_stats(team)
            features.append([
                team_stats['matches_played'],
                team_stats['matches_won'],
                team_stats['matches_lost'],
                team_stats['win_rate'],
                team_stats['avg_margin']
            ])
            labels_win_rate.append(team_stats['win_rate'])
            labels_margin.append(team_stats['avg_margin'])
            
        features = np.array(features)
        features = self.scaler.fit_transform(features)
        return features, np.array(labels_win_rate), np.array(labels_margin)
        
    def get_team_stats(self, team):
        """Calculate various statistics for a team"""
        team_matches = self.data[(self.data['team1'] == team) | (self.data['team2'] == team)]
        matches_played = len(team_matches)
        matches_won = len(team_matches[team_matches['winner'] == team])
        matches_lost = matches_played - matches_won
        win_rate = matches_won / matches_played if matches_played > 0 else 0
        
        margins = []
        for _, match in team_matches.iterrows():
            try:
                margin = float(match['margin'])
                if match['winner'] == team:
                    margins.append(margin)
                else:
                    margins.append(-margin)
            except (ValueError, TypeError):
                continue
                
        avg_margin = np.mean(margins) if margins else 0
        
        return {
            'matches_played': matches_played,
            'matches_won': matches_won,
            'matches_lost': matches_lost,
            'win_rate': win_rate,
            'avg_margin': avg_margin
        }
        
    def train_models(self):
        """Train XGBoost models for win rate and margin prediction"""
        features, labels_win_rate, labels_margin = self.prepare_features()
        
        # Split data
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            features, labels_win_rate, test_size=0.2, random_state=42
        )
        _, _, y_margin_train, y_margin_test = train_test_split(
            features, labels_margin, test_size=0.2, random_state=42
        )
        
        # Train win rate model
        self.win_rate_model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        self.win_rate_model.fit(X_train, y_win_train)
        
        # Train margin model
        self.margin_model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        self.margin_model.fit(X_train, y_margin_train)
        
        # Evaluate models
        win_rate_pred = self.win_rate_model.predict(X_test)
        margin_pred = self.margin_model.predict(X_test)
        
        win_rate_mse = mean_squared_error(y_win_test, win_rate_pred)
        margin_mse = mean_squared_error(y_margin_test, margin_pred)
        
        print(f"Win Rate Model MSE: {win_rate_mse:.4f}")
        print(f"Margin Model MSE: {margin_mse:.4f}")
        
    def calculate_team_performance(self):
        """Calculate team performance metrics using trained models"""
        teams = pd.concat([self.data['team1'], self.data['team2']]).unique()
        results = {}
        
        for team in teams:
            team_stats = self.get_team_stats(team)
            features = np.array([[
                team_stats['matches_played'],
                team_stats['matches_won'],
                team_stats['matches_lost'],
                team_stats['win_rate'],
                team_stats['avg_margin']
            ]])
            
            features_scaled = self.scaler.transform(features)
            predicted_win_rate = self.win_rate_model.predict(features_scaled)[0]
            predicted_margin = self.margin_model.predict(features_scaled)[0]
            
            results[team] = {
                'matches_played': team_stats['matches_played'],
                'matches_won': team_stats['matches_won'],
                'matches_lost': team_stats['matches_lost'],
                'win_rate': predicted_win_rate,
                'average_margin': predicted_margin,
                'total_margin': predicted_margin * team_stats['matches_played']
            }
            
        return pd.DataFrame.from_dict(results, orient='index').round(3)
        
    def save_results(self, results):
        """Save team performance analysis to CSV"""
        output_path = pathlib.Path('data/processed/team_performance_analysis_v2.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        results.to_csv(output_path)
        print(f"Results saved to {output_path}")
        
        # Display summary
        print("\nTeam Performance Summary:")
        print(results.sort_values('win_rate', ascending=False))

def main():
    model = MLTeamPerformanceModel()
    model.load_data()
    model.train_models()
    results = model.calculate_team_performance()
    model.save_results(results)
    
    # Display results
    print("\nTeam Performance Analysis:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main() 