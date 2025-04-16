import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path
import os

class MLBest11Model:
    def __init__(self):
        self.player_ratings = None
        self.player_roles = None
        self.best_11s = {}
        self.player_selector = None
        self.scaler = StandardScaler()
        self.role_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        
    def load_data(self):
        """Load player ratings data"""
        try:
            data_dir = Path('/Users/abhi/Desktop/Web Mind/data/processed')
            
            # Load player ratings
            ratings_file = data_dir / 'ml_player_ratings.csv'
            if not os.path.exists(ratings_file):
                print(f"Error: File not found: {ratings_file}")
                return False
                
            self.player_ratings = pd.read_csv(ratings_file)
            print(f"Loaded {len(self.player_ratings)} player ratings")
            
            # Load player roles
            roles_file = data_dir / 'player_roles.csv'
            if os.path.exists(roles_file):
                self.player_roles = pd.read_csv(roles_file)
                print(f"Loaded {len(self.player_roles)} player roles")
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
        
    def prepare_features(self):
        """Prepare features for ML model"""
        if self.player_ratings is None:
            print("Error: Player ratings not loaded")
            return None
            
        try:
            # Create features for player selection
            features = [
                'batting_rating', 'bowling_rating', 'overall_rating', 
                'role_consistency_score', 'final_rating'
            ]
            
            # Check if required columns exist
            missing_cols = [col for col in features if col not in self.player_ratings.columns]
            if missing_cols:
                print(f"Error: Missing columns: {missing_cols}")
                return None
            
            # Encode playing role
            self.player_ratings['role_encoded'] = self.role_encoder.fit_transform(
                self.player_ratings['playing_role']
            )
            
            # Encode team
            self.player_ratings['team_encoded'] = self.team_encoder.fit_transform(
                self.player_ratings['team']
            )
            
            # Prepare feature matrix
            X = self.player_ratings[features + ['role_encoded', 'team_encoded']].copy()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return None
    
    def train_model(self):
        """Train ML model for player selection"""
        if self.player_ratings is None:
            print("Error: Player ratings not loaded")
            return False
            
        try:
            # Prepare features
            X = self.prepare_features()
            if X is None:
                return False
            
            # Create target variable (1 for selected players, 0 for not selected)
            y = np.zeros(len(self.player_ratings))
            
            # For each team, mark top players as selected
            for team in self.player_ratings['team'].unique():
                team_players = self.player_ratings[self.player_ratings['team'] == team]
                
                # Select top 11 players by final_rating
                top_11_indices = team_players.nlargest(11, 'final_rating').index
                y[top_11_indices] = 1
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train player selector model
            self.player_selector = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.player_selector.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.player_selector.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            
            print(f"\nModel Performance:")
            print(f"Player Selection Accuracy: {accuracy:.2%}")
            
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def select_best_11(self, team_name):
        """Select best 11 players for a given team using ML model"""
        if self.player_ratings is None or self.player_selector is None:
            print("Error: Model not properly initialized")
            return None
            
        try:
            # Get team's players
            team_players = self.player_ratings[self.player_ratings['team'] == team_name].copy()
            
            if team_players.empty:
                print(f"No players found for team: {team_name}")
                return None
            
            # Prepare features for prediction
            X = self.prepare_features()
            if X is None:
                return None
            
            # Get team indices
            team_indices = team_players.index
            
            # Predict selection probability
            selection_probs = self.player_selector.predict_proba(X[team_indices])[:, 1]
            
            # Add probabilities to team players
            team_players['selection_probability'] = selection_probs
            
            # Select top 11 players by probability
            best_11 = team_players.nlargest(11, 'selection_probability')
            
            # Format output
            result = {
                'team': team_name,
                'players': best_11[['name', 'playing_role', 'batting_rating', 'bowling_rating', 'overall_rating']].to_dict('records')
            }
            
            return result
        except Exception as e:
            print(f"Error selecting best 11: {str(e)}")
            return None
    
    def generate_all_teams_best_11(self):
        """Generate best 11 for all teams"""
        teams = self.player_ratings['team'].unique()
        
        for team in teams:
            print(f"\nGenerating Best 11 for {team}...")
            best_11 = self.select_best_11(team)
            if best_11 is not None:
                self.best_11s[team] = best_11
                print(f"Best 11 for {team} generated successfully!")
    
    def save_best_11s(self):
        """Save all teams' best 11 to CSV files"""
        if not self.best_11s:
            print("No best 11s to save. Please generate them first.")
            return
            
        output_dir = Path('data/processed')
        
        # Save individual team files
        for team, best_11 in self.best_11s.items():
            # Sort by position
            best_11 = best_11.sort_values('position')
            
            # Remove encoded columns
            best_11 = best_11.drop(['role_encoded', 'team_encoded'], axis=1)
            
            # Keep all original columns from player_ratings
            filename = f'ml_best_11_{team.lower().replace(" ", "_")}.csv'
            best_11.to_csv(output_dir / filename, index=False)
            print(f"Saved best 11 for {team} to {filename}")
        
        # Save combined file with all columns
        combined_dfs = []
        for team, df in self.best_11s.items():
            df = df.sort_values('position').drop(['role_encoded', 'team_encoded'], axis=1)
            df['team'] = team  # Add team column
            combined_dfs.append(df)
        
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_df = combined_df[['team', 'position', 'name', 'playing_role', 'batting_rating', 
                                 'bowling_rating', 'overall_rating', 'role_consistency_score', 
                                 'final_rating']]  # Reorder columns
        combined_df.to_csv(output_dir / 'ml_all_teams_best_11.csv', index=False)
        print("Saved combined best 11s to ml_all_teams_best_11.csv")
        
    def display_best_11(self, team_name):
        """Display the best 11 for a given team"""
        if team_name not in self.best_11s:
            print(f"No best 11 found for team: {team_name}")
            return
            
        best_11 = self.best_11s[team_name]
        print(f"\nBest 11 for {team_name}:")
        print("-" * 50)
        for _, player in best_11.iterrows():
            print(f"{player['position']}. {player['name']} ({player['playing_role']}) - Rating: {player['final_rating']:.2f}")
        print("-" * 50)

def main():
    # Initialize model
    model = MLBest11Model()
    
    # Load data
    print("Loading player data...")
    model.load_data()
    
    # Train model
    print("\nTraining ML model...")
    model.train_model()
    
    # Generate best 11s for all teams
    print("\nGenerating best 11s for all teams...")
    model.generate_all_teams_best_11()
    
    # Save results
    print("\nSaving best 11s...")
    model.save_best_11s()
    
    # Display best 11s for each team
    print("\nDisplaying best 11s for each team:")
    for team in model.best_11s.keys():
        model.display_best_11(team)

if __name__ == "__main__":
    main() 