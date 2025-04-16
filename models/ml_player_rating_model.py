import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path

class MLPlayerRatingModel:
    def __init__(self):
        self.player_data = None
        self.batting_model = None
        self.bowling_model = None
        self.scaler = StandardScaler()
        self.role_encoder = LabelEncoder()
        self.ratings_df = None
        
    def load_data(self):
        """Load player performance data"""
        self.player_data = pd.read_csv('data/processed/player_performance_clean.csv')
        print("Data loaded successfully!")
        
    def prepare_features(self):
        """Prepare features for ML models"""
        # Create batting features
        batting_features = [
            'batting_runs', 'total_balls', 'total_fours', 'total_sixes',
            'batting_average', 'strike_rate'
        ]
        
        # Create bowling features
        bowling_features = [
            'total_overs', 'total_wickets', 'bowling_runs', 'total_maidens',
            'total_wides', 'total_noballs', 'bowling_average', 'economy_rate',
            'bowling_strike_rate'
        ]
        
        # Encode playing role
        self.player_data['role_encoded'] = self.role_encoder.fit_transform(
            self.player_data['playing_role']
        )
        
        # Prepare feature matrices
        X_batting = self.player_data[batting_features].copy()
        X_bowling = self.player_data[bowling_features].copy()
        
        # Add role encoding
        X_batting['role'] = self.player_data['role_encoded']
        X_bowling['role'] = self.player_data['role_encoded']
        
        # Scale features
        X_batting_scaled = self.scaler.fit_transform(X_batting)
        X_bowling_scaled = self.scaler.fit_transform(X_bowling)
        
        return X_batting_scaled, X_bowling_scaled
    
    def train_models(self):
        """Train ML models for batting and bowling ratings"""
        # Prepare features
        X_batting, X_bowling = self.prepare_features()
        
        # Prepare target variables
        y_batting = self.player_data['batting_rating']
        y_bowling = self.player_data['bowling_rating']
        
        # Split data
        X_bat_train, X_bat_test, y_bat_train, y_bat_test = train_test_split(
            X_batting, y_batting, test_size=0.2, random_state=42
        )
        
        X_bowl_train, X_bowl_test, y_bowl_train, y_bowl_test = train_test_split(
            X_bowling, y_bowling, test_size=0.2, random_state=42
        )
        
        # Train batting rating model
        self.batting_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.batting_model.fit(X_bat_train, y_bat_train)
        
        # Train bowling rating model
        self.bowling_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.bowling_model.fit(X_bowl_train, y_bowl_train)
        
        # Evaluate models
        bat_pred = self.batting_model.predict(X_bat_test)
        bowl_pred = self.bowling_model.predict(X_bowl_test)
        
        bat_mse = mean_squared_error(y_bat_test, bat_pred)
        bowl_mse = mean_squared_error(y_bowl_test, bowl_pred)
        
        bat_r2 = r2_score(y_bat_test, bat_pred)
        bowl_r2 = r2_score(y_bowl_test, bowl_pred)
        
        print("\nModel Performance:")
        print(f"Batting Rating - MSE: {bat_mse:.2f}, R²: {bat_r2:.2f}")
        print(f"Bowling Rating - MSE: {bowl_mse:.2f}, R²: {bowl_r2:.2f}")
    
    def calculate_ratings(self):
        """Calculate player ratings using ML models"""
        # Prepare features
        X_batting, X_bowling = self.prepare_features()
        
        # Get predictions
        batting_ratings = self.batting_model.predict(X_batting)
        bowling_ratings = self.bowling_model.predict(X_bowling)
        
        # Create ratings DataFrame
        self.ratings_df = self.player_data[['player_id', 'name', 'team', 'playing_role']].copy()
        self.ratings_df['batting_rating'] = batting_ratings
        self.ratings_df['bowling_rating'] = bowling_ratings
        
        # Calculate overall rating
        self.ratings_df['overall_rating'] = (
            self.ratings_df['batting_rating'] * 0.5 + 
            self.ratings_df['bowling_rating'] * 0.5
        )
        
        # Calculate role consistency score
        role_means = self.ratings_df.groupby('playing_role')['overall_rating'].mean()
        role_stds = self.ratings_df.groupby('playing_role')['overall_rating'].std()
        
        self.ratings_df['role_consistency_score'] = self.ratings_df.apply(
            lambda row: (row['overall_rating'] - role_means[row['playing_role']]) / 
                       role_stds[row['playing_role']] if role_stds[row['playing_role']] > 0 else 0,
            axis=1
        )
        
        # Calculate final rating
        self.ratings_df['final_rating'] = (
            self.ratings_df['overall_rating'] * 0.7 + 
            self.ratings_df['role_consistency_score'] * 0.3
        )
        
        # Round ratings
        rating_columns = ['batting_rating', 'bowling_rating', 'overall_rating', 
                         'role_consistency_score', 'final_rating']
        self.ratings_df[rating_columns] = self.ratings_df[rating_columns].round(2)
        
        return self.ratings_df
    
    def get_player_rating(self, player_id):
        """Get ratings for a specific player"""
        if self.ratings_df is None:
            self.calculate_ratings()
            
        player_data = self.ratings_df[self.ratings_df['player_id'] == player_id]
        if player_data.empty:
            return None
            
        return {
            'player_id': player_id,
            'name': player_data['name'].iloc[0],
            'team': player_data['team'].iloc[0],
            'playing_role': player_data['playing_role'].iloc[0],
            'batting_rating': player_data['batting_rating'].iloc[0],
            'bowling_rating': player_data['bowling_rating'].iloc[0],
            'overall_rating': player_data['overall_rating'].iloc[0],
            'role_consistency_score': player_data['role_consistency_score'].iloc[0],
            'final_rating': player_data['final_rating'].iloc[0]
        }
    
    def save_ratings(self, ratings_df):
        """Save player ratings to CSV"""
        output_path = Path('data/processed/ml_player_ratings.csv')
        ratings_df.to_csv(output_path, index=False)
        print(f"\nRatings saved to: {output_path}")

    def calculate_initial_ratings(self):
        """Calculate initial batting and bowling ratings based on performance metrics"""
        # Calculate batting rating with role-specific weights
        self.player_data['batting_rating'] = self.player_data.apply(
            lambda row: (
                row['batting_runs'] * 0.4 +
                row['strike_rate'] * 0.3 +
                row['batting_average'] * 0.3
            ) if pd.notna(row['batting_runs']) else 0,
            axis=1
        )
        
        # Calculate bowling rating with role-specific weights
        self.player_data['bowling_rating'] = self.player_data.apply(
            lambda row: (
                row['total_wickets'] * 0.4 +
                (100 - row['economy_rate']) * 0.3 +
                (100 - row['bowling_average']) * 0.3
            ) if pd.notna(row['total_wickets']) else 0,
            axis=1
        )
        
        # Adjust ratings based on playing role
        self.player_data['batting_rating'] = self.player_data.apply(
            lambda row: row['batting_rating'] * 1.2 if row['playing_role'] == 'Batsman' else
                       row['batting_rating'] * 1.0 if row['playing_role'] == 'All-rounder' else
                       row['batting_rating'] * 0.8,
            axis=1
        )
        
        self.player_data['bowling_rating'] = self.player_data.apply(
            lambda row: row['bowling_rating'] * 1.2 if row['playing_role'] == 'Bowler' else
                       row['bowling_rating'] * 1.0 if row['playing_role'] == 'All-rounder' else
                       row['bowling_rating'] * 0.8,
            axis=1
        )
        
        # Normalize ratings to 0-100 scale
        self.player_data['batting_rating'] = (
            (self.player_data['batting_rating'] - self.player_data['batting_rating'].min()) /
            (self.player_data['batting_rating'].max() - self.player_data['batting_rating'].min()) * 100
        )
        
        self.player_data['bowling_rating'] = (
            (self.player_data['bowling_rating'] - self.player_data['bowling_rating'].min()) /
            (self.player_data['bowling_rating'].max() - self.player_data['bowling_rating'].min()) * 100
        )
        
        # Fill NaN values with 0
        self.player_data['batting_rating'] = self.player_data['batting_rating'].fillna(0)
        self.player_data['bowling_rating'] = self.player_data['bowling_rating'].fillna(0)
        
    def get_player_ratings_by_name(self, player_name):
        """Get ratings for players matching the given name"""
        if self.ratings_df is None:
            self.calculate_ratings()
            
        # Convert both search term and player names to lowercase for case-insensitive search
        search_term = player_name.lower().strip()
        
        # Find players with names containing the search term
        player_data = self.ratings_df[
            self.ratings_df['name'].str.lower().str.contains(search_term, na=False)
        ]
        
        if player_data.empty:
            return None
            
        # Convert to list of dictionaries
        players = []
        for _, row in player_data.iterrows():
            players.append({
                'player_id': int(row['player_id']),
                'name': row['name'],
                'team': row['team'],
                'playing_role': row['playing_role'],
                'batting_rating': float(row['batting_rating']),
                'bowling_rating': float(row['bowling_rating']),
                'overall_rating': float(row['overall_rating']),
                'role_consistency_score': float(row['role_consistency_score']),
                'final_rating': float(row['final_rating'])
            })
            
        return players

def main():
    # Initialize model
    model = MLPlayerRatingModel()
    
    # Load data
    print("Loading player data...")
    model.load_data()
    
    # Calculate initial ratings
    print("\nCalculating initial ratings...")
    model.calculate_initial_ratings()
    
    # Train models
    print("\nTraining ML models...")
    model.train_models()
    
    # Calculate ratings
    print("\nCalculating player ratings...")
    ratings_df = model.calculate_ratings()
    
    # Save ratings
    model.save_ratings(ratings_df)
    
    # Display top players by role
    print("\nTop Players by Role:")
    print("-" * 50)
    
    roles = ratings_df['playing_role'].unique()
    for role in roles:
        role_players = ratings_df[ratings_df['playing_role'] == role].nlargest(3, 'final_rating')
        print(f"\n{role}:")
        for _, player in role_players.iterrows():
            print(f"{player['name']} ({player['team']}) - Rating: {player['final_rating']:.2f}")

if __name__ == "__main__":
    main() 