import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
from pathlib import Path

class MLMatchPredictionModel:
    def __init__(self):
        self.player_ratings = None
        self.matches = None
        self.winner_model = None
        self.margin_model = None
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        
    def load_data(self):
        """Load player ratings and match data"""
        data_dir = Path('data/processed')
        
        # Load player ratings
        self.player_ratings = pd.read_csv(data_dir / 'player_ratings.csv')
        
        # Load match data
        self.matches = pd.read_csv(data_dir / 'matches.csv')
        
        print("Data loaded successfully!")
        
    def _convert_margin_to_numeric(self, margin_str):
        """Convert margin string to numeric value"""
        try:
            # Extract the numeric value
            value = int(margin_str.split()[0])
            
            # Convert wickets to equivalent runs (assuming 1 wicket = 10 runs)
            if 'wickets' in margin_str:
                value = value * 10
                
            return value
        except:
            return 0
            
    def prepare_features(self):
        """Prepare features for match prediction"""
        # Calculate team statistics
        team_stats = self._calculate_team_stats()
        
        # Prepare features for each match
        features = []
        labels = []
        margins = []
        
        for _, match in self.matches.iterrows():
            team1 = match['team1']
            team2 = match['team2']
            
            # Get team features
            team1_features = self._get_team_features(team1, team_stats)
            team2_features = self._get_team_features(team2, team_stats)
            
            # Combine features
            match_features = np.concatenate([team1_features, team2_features])
            features.append(match_features)
            
            # Get labels
            if match['winner'] == team1:
                labels.append(1)
            else:
                labels.append(0)
                
            # Get margin as numeric value
            margin_value = self._convert_margin_to_numeric(match['margin'])
            margins.append(margin_value)
            
        return np.array(features), np.array(labels), np.array(margins)
    
    def _calculate_team_stats(self):
        """Calculate team statistics from player ratings"""
        team_stats = {}
        
        for team in self.player_ratings['team'].unique():
            team_players = self.player_ratings[self.player_ratings['team'] == team]
            
            # Calculate team batting strength
            batting_strength = team_players['batting_rating'].mean()
            
            # Calculate team bowling strength
            bowling_strength = team_players['bowling_rating'].mean()
            
            # Calculate team overall strength
            overall_strength = team_players['final_rating'].mean()
            
            team_stats[team] = {
                'batting_strength': batting_strength,
                'bowling_strength': bowling_strength,
                'overall_strength': overall_strength
            }
            
        return team_stats
    
    def _get_team_features(self, team, team_stats):
        """Get features for a team"""
        stats = team_stats[team]
        
        # Basic team features
        features = [
            stats['batting_strength'],
            stats['bowling_strength'],
            stats['overall_strength']
        ]
        
        # Add recent form (last 5 matches)
        recent_matches = self.matches[
            (self.matches['team1'] == team) | (self.matches['team2'] == team)
        ].tail(5)
        
        wins = len(recent_matches[recent_matches['winner'] == team])
        features.append(wins / 5)  # Win rate in last 5 matches
        
        # Add head-to-head advantage
        h2h_matches = self.matches[
            ((self.matches['team1'] == team) & (self.matches['team2'] != team)) |
            ((self.matches['team2'] == team) & (self.matches['team1'] != team))
        ]
        
        h2h_wins = len(h2h_matches[h2h_matches['winner'] == team])
        features.append(h2h_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0.5)
        
        return np.array(features)
    
    def train_models(self):
        """Train ML models for winner and margin prediction"""
        # Prepare features and labels
        X, y, margins = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test, margins_train, margins_test = train_test_split(
            X, y, margins, test_size=0.2, random_state=42
        )
        
        # Train winner prediction model
        self.winner_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.winner_model.fit(X_train, y_train)
        
        # Train margin prediction model
        self.margin_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.margin_model.fit(X_train, margins_train)
        
        # Evaluate models
        y_pred = self.winner_model.predict(X_test)
        margin_pred = self.margin_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(margins_test, margin_pred)
        
        print(f"\nModel Performance:")
        print(f"Winner Prediction Accuracy: {accuracy:.2%}")
        print(f"Margin Prediction MSE: {mse:.2f}")
    
    def predict_match(self, team1, team2):
        """Predict the outcome of a match between two teams"""
        # Calculate team statistics
        team_stats = self._calculate_team_stats()
        
        # Get team features
        team1_features = self._get_team_features(team1, team_stats)
        team2_features = self._get_team_features(team2, team_stats)
        
        # Combine features
        match_features = np.concatenate([team1_features, team2_features])
        
        # Predict winner
        winner_prob = self.winner_model.predict_proba([match_features])[0]
        predicted_winner = team1 if winner_prob[1] > 0.5 else team2
        
        # Predict margin
        predicted_margin = self.margin_model.predict([match_features])[0]
        
        return {
            'team1': team1,
            'team2': team2,
            'predicted_winner': predicted_winner,
            'team1_probability': winner_prob[1],
            'team2_probability': winner_prob[0],
            'predicted_margin': predicted_margin
        }
    
    def generate_predictions(self):
        """Generate predictions for all matches"""
        predictions = []
        
        for _, match in self.matches.iterrows():
            prediction = self.predict_match(match['team1'], match['team2'])
            prediction['actual_winner'] = match['winner']
            prediction['actual_margin'] = match['margin']
            prediction['prediction_status'] = 'Correct' if prediction['predicted_winner'] == match['winner'] else 'Incorrect'
            predictions.append(prediction)
            
        return pd.DataFrame(predictions)
    
    def save_predictions(self, predictions):
        """Save match predictions to CSV"""
        output_dir = Path('data/processed')
        predictions.to_csv(output_dir / 'ml_match_predictions.csv', index=False)
        print("\nPredictions saved to ml_match_predictions.csv")
        
    def display_prediction_accuracy(self, predictions):
        """Display prediction accuracy statistics"""
        total_matches = len(predictions)
        correct_predictions = len(predictions[predictions['prediction_status'] == 'Correct'])
        accuracy = correct_predictions / total_matches
        
        print(f"\nPrediction Accuracy: {accuracy:.2%}")
        print(f"Correct Predictions: {correct_predictions}/{total_matches}")

def main():
    # Initialize model
    model = MLMatchPredictionModel()
    
    # Load data
    print("Loading data...")
    model.load_data()
    
    # Train models
    print("\nTraining ML models...")
    model.train_models()
    
    # Generate predictions
    print("\nGenerating match predictions...")
    predictions = model.generate_predictions()
    
    # Save predictions
    model.save_predictions(predictions)
    
    # Display accuracy
    model.display_prediction_accuracy(predictions)

if __name__ == "__main__":
    main() 