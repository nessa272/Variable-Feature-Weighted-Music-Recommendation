from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")
spotify_dataset = 'spotify_data.csv'

def get_user_history(filename='listening_history.csv', limit=50):
    """
    Updates and returns the user's recently played listening history.
    If a local history file exists, appends new entries and removes duplicates.

    Args:
        filename (str): The path to save/load the listening history
        limit (int): Number of recent tracks to fetch from Spotify

    Returns:
        pd.DataFrame: Updated listening history with track ID, track name, artist, and played_at timestamp
    """
    scope = "user-read-recently-played"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope))
    recent_data = sp.current_user_recently_played(limit)
    new_tracks = []
    for item in recent_data['items']:
        track = item['track']
        played_at = item['played_at']
        new_tracks.append({
            'track_id': track['id'],
            'track_name': track['name'],
            'artist': track['artists'][0]['name'],
            'played_at': played_at})
    new_df = pd.DataFrame(new_tracks)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset='played_at', inplace=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filename, index=False)
    print(f'Listening history updated. Saved to {filename}')
    return combined_df

def get_audio_features(dataset_file, recent_data, filename='audio_features.csv'):
    """
    Retrieves audio features from a dataset for tracks in the listening history.
    If a track is missing, the average features for the artist are used instead.

    Args:
        dataset_file (str): Path to the dataset CSV file with audio features
        recent_data (pd.DataFrame): DataFrame of recently played tracks
        filename (str): Output path to save the filtered audio features

    Returns:
        pd.DataFrame: DataFrame containing audio features for each track or artist average
    """
    artist_names = set()
    track_names = {}
    for _, row in recent_data.iterrows():
        artist = row['artist']
        artist_names.add(artist)
        track_names[row['track_name']] = artist

    features = ['artist_name', 'track_name', 'danceability', 
                'energy', 'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'loudness']
    played_artists = []
    for chunk in pd.read_csv(dataset_file, usecols=features, chunksize=10_000):
        contains_played_artists = chunk[chunk['artist_name'].isin(artist_names)]
        if not contains_played_artists.empty:
            played_artists.append(contains_played_artists)
    all_artist_audio_features = pd.concat(played_artists, ignore_index=True)

    present_tracks = set(all_artist_audio_features['track_name'])
    calculated_artists = set()
    calculated_artist_averages = []
    missing_tracks = []
    for track_name, artist_name in track_names.items():
        if track_name not in present_tracks:
            missing_tracks.append((track_name, artist_name))
            if artist_name not in calculated_artists:
                artist_df = all_artist_audio_features[all_artist_audio_features['artist_name'] == artist_name]
                if not artist_df.empty:
                    average_features = artist_df.mean(numeric_only=True)
                    average_features['artist_name'] = artist_name
                    average_features['track_name'] = f'{artist_name}_average'
                    calculated_artist_averages.append(average_features)
                    calculated_artists.add(artist_name)

    avg_tracks = pd.DataFrame(calculated_artist_averages)
    real_tracks = all_artist_audio_features[all_artist_audio_features['track_name'].isin(track_names.keys())]
    audio_features = pd.concat([real_tracks, avg_tracks], ignore_index=True)
    audio_features.to_csv(filename, index=False)
    print(f'Audio features updated. Saved to {filename}')
    missing_tracks_df = pd.DataFrame(missing_tracks, columns=['track_name', 'artist'])
    missing_tracks_df.to_csv('missing_tracks.csv', index=False)
    print(f'Missing tracks updated. Saved to missing_tracks.csv')
    return audio_features

def get_data():
    """
    Retrieves the scaled audio features from recent listening history.
    - Gets listening history from Spotify
    - Gets audio features of tracks in history from dataset
    - Scales numeric features

    Returns:
        tuple: (audio features, numeric features, scaler)
            audio features (pd.DataFrame): DataFrame with scaled audio features for each track or artist average
            numeric features (List[str]): List of numeric features
            scaler (StandardScaler): Scaler for normalizing numeric audio features
    """
    recent_data = get_user_history()
    audio_features = get_audio_features(spotify_dataset, recent_data)
    numeric_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo', 'loudness']
    # Scale numeric audio features
    audio_features.to_csv('prescaled_audio_features.csv', index=False)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(audio_features[numeric_features])
    audio_features[numeric_features] = scaled_data

    return audio_features, numeric_features, scaler

def get_session_data(gap_minutes=10, max_session_size=4):
    audio_features, numeric_features, scaler = get_data()
    sessions = []
    current_session = []
    recent = get_user_history() # Redundant oops
    recent_data = recent.sort_values(by='played_at')
    for _, row in recent_data.iterrows():
        played_at = datetime.fromisoformat(row['played_at'].replace("Z", "+00:00"))
        if not current_session:
            current_session.append((row, played_at))
            continue
        time_gap = played_at - current_session[-1][1]
        if time_gap > timedelta(minutes=gap_minutes) or len(current_session) >= max_session_size:
            sessions.append(current_session)
            current_session = []
        current_session.append((row, played_at))
    if current_session:
        sessions.append(current_session)

    session_rows = []
    for i, session in enumerate(sessions):
        row = get_session_average(session, audio_features, numeric_features, session_id=i)
        if row is not None:
            session_rows.append(row)

    session_df = pd.DataFrame(session_rows)
    return session_df, numeric_features, scaler

def get_session_average(session, audio_features, numeric_features, session_id=0):
    track_names = []
    for row, _ in session:
        track_name = row['track_name']
        if track_name in audio_features['track_name'].values:
            track_names.append(track_name)
        else:
            fallback = f"{row['artist']}_average"
            if fallback in audio_features['track_name'].values:
                track_names.append(fallback)

    session_audio = audio_features[audio_features['track_name'].isin(track_names)]
    if session_audio.empty:
        return None
    averages = session_audio[numeric_features].mean()

    session_row = averages.copy()
    session_row['track_name'] = str(f"session_{session_id}_average")
    session_row['artist_name'] = str(f"session_{session_id}")

    return session_row

def feature_weighted_fuzzy_kmeans(audio_features, num_clusters, max_iter=100, tol=1e-4):
    """
    Runs a variable feature weighted fuzzy k-means algorithm on input.
    Based on the method proposed by Singh & Verma, clusters samples with fuzzy
    membershup values and assigns weights to features per cluster.

    Args:
        audio_features (pd.DataFrame): Scaled numeric audio features
        num_clusters (int): Number of clusters to form
        max_iter (int): Max number of iterations
        tol (float): Tolerance for convergence

    Returns:
        tuple: (labels, cluster centers, feature weights, membership matrix)
            labels (np.ndarray): Hard cluster assignment for each sample
            cluster centers (np.ndarray): Coordinates of cluster centers
            feature weights (np.ndarray): Feature weight matrix
            membership matrix (np.ndarray): Fuzzy partition matrix
    """
    X = audio_features.to_numpy()
    num_samples, num_features = X.shape

    # Fuzzy partition matrix U, initialized randomly (num_samples, num_clusters)
    U = np.random.dirichlet(np.ones(num_clusters), size=num_samples) 
    # Cluster centers matrix V, computed with weighted averages (num_clusters, num_features)
    V = (U.T @ X) / np.sum(U.T, axis=1, keepdims=True)
    # Feature weights matrix W, initialized with equal weights 1/m (num_clusters, num_features)
    W = np.ones((num_clusters, num_features)) / num_features
    # Control parameters for fuzzy partitions
    lambda_i = np.ones(num_samples)
    # Control parameters for features weights of each cluster
    gamma = np.ones(num_clusters)

    for _ in range(max_iter):
        # Distance/dissimilarity matrix D
        D = np.zeros((num_samples, num_clusters, num_features))
        for j in range(num_clusters):
            diff = X - V[j]
            D[:, j, :] = diff**2

        # Update lambda_i (Eq. 17)
        eta = 1e-6
        for i in range(num_samples):
            numerator = np.sum(U[i][:, np.newaxis] * W * D[i])
            denominator = np.sum(-U[i] * np.log(U[i] + 1e-10)) + eta  
            lambda_i[i] = numerator / denominator
        
        # Update gamma (Eq. 18)
        for j in range(num_clusters):
            numerator = np.sum(U[:, j][:, np.newaxis] * W[j] * D[:, j, :])
            denominator = np.sum(-W[j] * np.log(W[j] + 1e-10)) + eta
            gamma[j] = numerator / denominator

        # Update feature weights w_jl in W (Eq. 15)
        for j in range(num_clusters):
            numerator = np.sum(U[:, j][:, np.newaxis] * D[:, j, :], axis=0)
            numerator = np.exp(-numerator/gamma[j])
            W[j] = numerator / np.sum(numerator)
        
        # Update fuzzy partitions u_ij in U (Eq. 16)
        for i in range(num_samples):
            numerator = np.sum(W * D[i], axis=1)
            numerator = np.exp(-numerator/lambda_i[i])
            U[i] = numerator / np.sum(numerator)

        # Update cluster centers v_jl in V (Eq. 5)
        V_new = (U.T @ X) / np.sum(U.T, axis=1, keepdims=True)

        # Check convergence
        if np.linalg.norm(V_new - V) < tol:
            break
        V = V_new
    labels = np.argmax(U, axis=1) # Top 1 cluster
    return labels, V, W, U

def unweighted_fuzzy_kmeans(audio_features, num_clusters):
    """
    Runs standard fuzzy c-means clustering (no feature weighting)

    Args:
        audio_features (pd.DataFrame): Scaled numeric audio features.
        num_clusters (int): Number of clusters to form.

    Returns:
        tuple: (labels, cluster_centers, feature_weights, membership_matrix)
            labels (np.ndarray): Hard cluster assignment for each sample
            cluster_centers (np.ndarray): Coordinates of cluster centers
            feature_weights (np.ndarray): Uniform feature weights
            membership_matrix (np.ndarray): Fuzzy partition matrix
    """
    X = audio_features.to_numpy().T
    centers, U, _, _, _, _, _ = fuzz.cluster.cmeans(X, c=num_clusters, m=2.0, error=1e-4, maxiter=100, init=None)
    
    labels = np.argmax(U, axis=0)
    weights = np.ones((num_clusters, X.shape[0])) / X.shape[0]  # Uniform weights

    return labels, centers, weights, U.T

def run_weighted_clustering(num_clusters=5, in_sessions=False):
    """
    Full pipeline for clustering listening history:
    - Updates recent history
    - Extracts audio features
    - Scales features
    - Clusters using feature-weighted fuzzy k-means

    Args:
        num_clusters (int): Number of clusters.

    Returns:
        tuple: (audio_features, numeric features, labels, centers, weights, partitions)
            audio features (pd.DataFrame): Audio features
            numeric features (list): List of feature column names.
            labels (np.ndarray): Hard cluster labels.
            centers (np.ndarray): Cluster center coordinates.
            weights (np.ndarray): Feature weight matrix.
            partitions (np.ndarray): Fuzzy partition matrix.
    """
    if not in_sessions:
        audio_features, numeric_features, scaler = get_data()
    else:
        audio_features, numeric_features, scaler = get_session_data()
    labels, centers, weights, partitions = feature_weighted_fuzzy_kmeans(audio_features[numeric_features], 
                                                                         num_clusters=num_clusters)
    return audio_features, numeric_features, labels, centers, weights, partitions, scaler

def recommend_tracks(audio_features, numeric_features, labels, centers, weights, partitions, scaler):
    audio_features.to_csv('scaled_audio_features.csv', index=False)
    unscaled_data = scaler.inverse_transform(audio_features[numeric_features])
    audio_features[numeric_features] = unscaled_data
    audio_features.to_csv('unscaled_audio_features.csv', index=False)

    

def run(in_sessions=False):
    output_dir = 'output'
    if not in_sessions:
        audio_features, numeric_features, w_labels, w_centers, w_weights, w_partitions, scaler = run_weighted_clustering()
    else:
        audio_features, numeric_features, w_labels, w_centers, w_weights, w_partitions, scaler = run_weighted_clustering(in_sessions=True)
        output_dir='sessions_output'
    uw_labels, uw_centers, uw_weights, uw_partitions = unweighted_fuzzy_kmeans(audio_features[numeric_features], num_clusters=5)
    compare_clusters(audio_features, numeric_features, w_labels, w_weights, uw_labels, uw_weights, output_dir=output_dir)
    recommend_tracks(audio_features, numeric_features, w_labels, w_centers, w_weights, w_partitions, scaler)


def compare_clusters(audio_features, numeric_features, w_labels, w_weights, uw_labels, uw_weights, output_dir='output'):
    """
    Compares clustering from feature weighted and unweighted fuzzy kmeans clustering.

    Args:
        audio_features (pd.DataFrame): Audio features
        numeric_features (List[str]): Numeric audio features
        w_labels (np.ndarray): Weighted cluster labels
        w_weights (np.ndarray): Weighted cluster feature weights
        uw_labels (np.ndarray): Unweighted cluster labels
        uw_weights (np.ndarray): Unweighted cluster weights (uniform)
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Feature weights comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    pd.DataFrame(w_weights, columns=numeric_features).T.plot(kind='bar', ax=axes[0], legend=False)
    axes[0].set_title("Feature-Weighted Weights")
    axes[0].set_ylabel("Weight")
    axes[0].tick_params(axis='x', rotation=45)
    
    pd.DataFrame(uw_weights, columns=numeric_features).T.plot(kind='bar', ax=axes[1], legend=False)
    axes[1].set_title("Unweighted Weights")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle("Feature Weights Comparison")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compare_weights.png")
    plt.close()

    # Create one figure for all boxplot comparisons
    num_features = len(numeric_features)
    fig, axes = plt.subplots(num_features, 2, figsize=(14, 5 * num_features), sharey='row')

    for i, feature in enumerate(numeric_features):
        feature_index = numeric_features.index(feature)

        # Feature-weighted boxplot
        audio_features['cluster'] = w_labels
        sns.boxplot(data=audio_features, x='cluster', y=feature, ax=axes[i, 0])
        axes[i, 0].set_title(f"Weighted: {feature}")
        axes[i, 0].tick_params(axis='x')

        # Set alpha with weight
        for j, patch in enumerate(axes[i, 0].patches[:len(w_weights)]):
            patch.set_alpha(w_weights[j, feature_index])

        # Unweighted boxplot
        audio_features['cluster'] = uw_labels
        sns.boxplot(data=audio_features, x='cluster', y=feature, ax=axes[i, 1])
        axes[i, 1].set_title(f"Unweighted: {feature}")
        axes[i, 1].tick_params(axis='x')

        # Fixed alpha
        for patch in axes[i, 1].patches[:len(uw_weights)]:
            patch.set_alpha(1 / num_features)
        
        # For weighted plot
        axes[i, 0].set_xticks(range(w_weights.shape[0]))
        axes[i, 0].set_xticklabels([
            f"{j} ({w_weights[j, feature_index]:.2f})" for j in range(w_weights.shape[0])
        ])

        # For unweighted plot
        axes[i, 1].set_xticks(range(uw_weights.shape[0]))
        axes[i, 1].set_xticklabels([
            str(j) for j in range(uw_weights.shape[0])
        ])

    plt.suptitle("Boxplot Comparison by Feature and Cluster", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{output_dir}/stacked_boxplots.png")
    plt.close()

    print(f'Cluster graphs saved to {output_dir}')

run(in_sessions=False)
#run(in_sessions=True)