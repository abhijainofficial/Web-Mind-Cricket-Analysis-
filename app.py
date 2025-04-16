from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from models.ml_team_performance_model import MLTeamPerformanceModel
from models.ml_player_rating_model import MLPlayerRatingModel
from models.ml_match_prediction_model import MLMatchPredictionModel
from models.ml_best_11_model import MLBest11Model
import os
import traceback

# Create Flask app
app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)

# Initialize models as None
team_model = None
player_model = None
match_model = None
best11_model = None

def initialize_models():
    global team_model, player_model, match_model, best11_model
    try:
        print("Loading Team Performance Model...")
        team_model = MLTeamPerformanceModel()
        team_model.load_data()
        team_model.train_models()
        print("Team Performance Model loaded successfully!")
        
        print("Loading Player Rating Model...")
        player_model = MLPlayerRatingModel()
        player_model.load_data()
        player_model.calculate_initial_ratings()
        player_model.train_models()
        player_model.calculate_ratings()
        print("Player Rating Model loaded successfully!")
        
        print("Loading Match Prediction Model...")
        match_model = MLMatchPredictionModel()
        match_model.load_data()
        match_model.train_models()  # Train the model after loading data
        print("Match Prediction Model loaded successfully!")
        
        print("Loading Best 11 Model...")
        best11_model = MLBest11Model()
        best11_model.load_data()
        best11_model.train_model()
        print("Best 11 Model loaded successfully!")
        
        print("All models initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

# Initialize models when the app starts
initialize_models()

@app.route('/')
def home():
    print("Rendering home page...")
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/api/predict/match', methods=['POST'])
def predict_match():
    if match_model is None:
        return jsonify({'error': 'Match prediction model not initialized'}), 500
    try:
        data = request.get_json()
        team1 = data.get('team1')
        team2 = data.get('team2')
        
        if not all([team1, team2]):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        prediction = match_model.predict_match(team1, team2)
        
        # Convert numpy float32 values to Python native float
        prediction['team1_probability'] = float(prediction['team1_probability'])
        prediction['team2_probability'] = float(prediction['team2_probability'])
        prediction['predicted_margin'] = float(prediction['predicted_margin'])
        
        return jsonify(prediction)
    except Exception as e:
        print(f"Error in predict_match: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/player/ratings/<player_id>', methods=['GET'])
def get_player_ratings(player_id):
    if player_model is None:
        return jsonify({'error': 'Player rating model not initialized'}), 500
    try:
        ratings = player_model.get_player_rating(player_id)
        if ratings is None:
            return jsonify({'error': f'Player with ID {player_id} not found'}), 404
        return jsonify(ratings)
    except Exception as e:
        print(f"Error in get_player_ratings: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/player/ratings/name/<player_name>', methods=['GET'])
def get_player_ratings_by_name(player_name):
    """Get player ratings by name"""
    if not player_model:
        return jsonify({'error': 'Player rating model not initialized'}), 500
        
    try:
        # Get player ratings using the new name-based search
        players = player_model.get_player_ratings_by_name(player_name)
        
        if not players:
            return jsonify({'error': f'No players found matching "{player_name}"'}), 404
            
        return jsonify(players)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/team/performance/<team_name>', methods=['GET'])
def get_team_performance(team_name):
    if team_model is None:
        return jsonify({'error': 'Team performance model not initialized'}), 500
    try:
        # Calculate performance for all teams
        performance_df = team_model.calculate_team_performance()
        
        # Check if team exists
        if team_name not in performance_df.index:
            return jsonify({'error': f'Team {team_name} not found'}), 404
            
        # Get team performance
        team_performance = performance_df.loc[team_name].to_dict()
        return jsonify(team_performance)
    except Exception as e:
        print(f"Error in get_team_performance: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/team/best11/<team_name>', methods=['GET'])
def get_best11(team_name):
    if best11_model is None:
        return jsonify({'error': 'Best 11 model not initialized'}), 500
    try:
        best11 = best11_model.select_best_11(team_name)
        if best11 is None:
            return jsonify({'error': f'No players found for team: {team_name}'}), 404
        return jsonify(best11)
    except Exception as e:
        print(f"Error in get_best11: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/players/list', methods=['GET'])
def get_all_players():
    """Get list of all players"""
    if not player_model:
        return jsonify({'error': 'Player rating model not initialized'}), 500
        
    try:
        if player_model.ratings_df is None:
            player_model.calculate_ratings()
            
        # Get all players sorted by name
        players = player_model.ratings_df[['player_id', 'name', 'team']].sort_values('name').to_dict('records')
        return jsonify(players)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams/list', methods=['GET'])
def get_all_teams():
    """Get list of all teams"""
    if not team_model:
        return jsonify({'error': 'Team performance model not initialized'}), 500
        
    try:
        # Get unique teams from the match data
        teams = pd.concat([team_model.data['team1'], team_model.data['team2']]).unique()
        teams = sorted([team for team in teams if team != 'no result' and team != 'abandoned'])
        # Return as a list of team objects with id and name
        teams_list = [{'id': i, 'name': team} for i, team in enumerate(teams)]
        return jsonify(teams_list)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Template folder:", app.template_folder)
    print("Static folder:", app.static_folder)
    app.run(host='0.0.0.0', port=5001, debug=True) 