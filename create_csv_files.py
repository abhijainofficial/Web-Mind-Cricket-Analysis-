import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_player_info(player_data):
    players = []
    for player in player_data:
        players.append({
            'player_id': len(players) + 1,
            'name': player['name'],
            'team': player['team'],
            'batting_style': player['battingStyle'],
            'bowling_style': player['bowlingStyle'],
            'playing_role': player['playingRole']
        })
    return pd.DataFrame(players)

def process_batting_data(batting_data):
    batting_stats = {}
    for match in batting_data:
        for innings in match['battingSummary']:
            player_name = innings['batsmanName']
            if player_name not in batting_stats:
                batting_stats[player_name] = {
                    'batting_runs': 0,
                    'total_balls': 0,
                    'total_fours': 0,
                    'total_sixes': 0,
                    'matches_played': 0
                }
            
            stats = batting_stats[player_name]
            stats['batting_runs'] += int(innings['runs'])
            stats['total_balls'] += int(innings['balls'])
            stats['total_fours'] += int(innings['4s'])
            stats['total_sixes'] += int(innings['6s'])
            stats['matches_played'] += 1
    
    return pd.DataFrame.from_dict(batting_stats, orient='index').reset_index().rename(columns={'index': 'name'})

def process_bowling_data(bowling_data):
    bowling_stats = {}
    for match in bowling_data:
        for innings in match['bowlingSummary']:
            player_name = innings['bowlerName']
            if player_name not in bowling_stats:
                bowling_stats[player_name] = {
                    'total_overs': 0,
                    'total_wickets': 0,
                    'bowling_runs': 0,
                    'total_maidens': 0,
                    'total_wides': 0,
                    'total_noballs': 0
                }
            
            stats = bowling_stats[player_name]
            stats['total_overs'] += float(innings['overs'])
            stats['total_wickets'] += int(innings['wickets'])
            stats['bowling_runs'] += int(innings['runs'])
            stats['total_maidens'] += int(innings['maiden'])
            stats['total_wides'] += int(innings['wides'])
            stats['total_noballs'] += int(innings['noBalls'])
    
    return pd.DataFrame.from_dict(bowling_stats, orient='index').reset_index().rename(columns={'index': 'name'})

def process_match_data(match_data):
    matches = []
    for match in match_data:
        for summary in match['matchSummary']:
            matches.append({
                'match_id': len(matches) + 1,
                'team1': summary['team1'],
                'team2': summary['team2'],
                'winner': summary['winner'],
                'margin': summary['margin'],
                'ground': summary['ground'],
                'match_date': summary['matchDate']
            })
    return pd.DataFrame(matches)

def calculate_player_ratings(players_df, batting_df, bowling_df):
    # Merge all player statistics
    player_stats = players_df.merge(batting_df, on='name', how='left')
    player_stats = player_stats.merge(bowling_df, on='name', how='left')
    
    # Fill NaN values with 0
    numeric_columns = ['batting_runs', 'total_balls', 'total_fours', 'total_sixes', 
                      'total_overs', 'total_wickets', 'bowling_runs', 'total_maidens',
                      'total_wides', 'total_noballs']
    player_stats[numeric_columns] = player_stats[numeric_columns].fillna(0)
    
    # Calculate batting strike rate
    player_stats['batting_strike_rate'] = np.where(
        player_stats['total_balls'] > 0,
        (player_stats['batting_runs'] / player_stats['total_balls']) * 100,
        0
    )
    
    # Calculate bowling economy
    player_stats['bowling_economy'] = np.where(
        player_stats['total_overs'] > 0,
        player_stats['bowling_runs'] / player_stats['total_overs'],
        0
    )
    
    # Calculate player rating (simple version)
    player_stats['player_rating'] = (
        player_stats['batting_strike_rate'] * 0.4 +
        (100 - player_stats['bowling_economy']) * 0.4 +
        player_stats['total_wickets'] * 2 +
        player_stats['total_fours'] * 0.5 +
        player_stats['total_sixes'] * 1
    )
    
    return player_stats

def main():
    # Load JSON files from raw data directory
    player_info = load_json_file('/Users/abhi/Desktop/Web Mind/data/raw/t20_wc_player_info.json')
    batting_data = load_json_file('/Users/abhi/Desktop/Web Mind/data/raw/t20_wc_batting_summary.json')
    bowling_data = load_json_file('/Users/abhi/Desktop/Web Mind/data/raw/t20_wc_bowling_summary.json')
    match_data = load_json_file('/Users/abhi/Desktop/Web Mind/data/raw/t20_wc_match_results.json')
    
    # Process data
    players_df = process_player_info(player_info)
    batting_df = process_batting_data(batting_data)
    bowling_df = process_bowling_data(bowling_data)
    matches_df = process_match_data(match_data)
    
    # Calculate player ratings
    players_df = calculate_player_ratings(players_df, batting_df, bowling_df)
    
    # Save to CSV files
    players_df.to_csv('data/processed/players.csv', index=False)
    matches_df.to_csv('data/processed/matches.csv', index=False)
    
    # Create player performance data
    player_performance = pd.merge(
        players_df[['player_id', 'name', 'team', 'playing_role']],
        batting_df,
        on='name',
        how='left'
    )
    player_performance = pd.merge(
        player_performance,
        bowling_df,
        on='name',
        how='left'
    )
    player_performance.to_csv('data/processed/player_performance.csv', index=False)
    
    # Create team performance data
    team_performance = matches_df.copy()
    team_performance['team_strength_index'] = np.random.uniform(0, 1, size=len(team_performance))
    team_performance.to_csv('data/processed/team_performance.csv', index=False)
    
    # Create player roles data
    player_roles = players_df[['player_id', 'name', 'team', 'batting_style', 'bowling_style', 'playing_role']].copy()
    player_roles['role_consistency_score'] = np.random.uniform(0, 1, size=len(player_roles))
    player_roles.to_csv('data/processed/player_roles.csv', index=False)
    
    print("CSV files have been created successfully in the data/processed directory!")

if __name__ == "__main__":
    main() 