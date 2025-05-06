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
import heapq

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")
spotify_dataset = 'spotify_data.csv'

def get_user_history(limit=50):
    """
    Updates and returns the user's recently played listening history.
    If a local history file exists, appends new entries and removes duplicates.
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
    if os.path.exists('listening_history.csv'):
        # Add to existing history
        existing_df = pd.read_csv('listening_history.csv')
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset='played_at', inplace=True)
    else:
        combined_df = new_df
    combined_df.to_csv('listening_history.csv', index=False)
    print(f'Listening history updated. Saved to "listening_history.csv"')
    return combined_df

def get_audio_features(recent_data):
    """
    Retrieves audio features for all tracks in recent_data.
    Uses artist averages when a track is missing in the dataset.
    """
    features = ['artist_name', 'track_name', 'danceability', 'energy', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness']

    artist_names = set(recent_data['artist'])

    artist_data = pd.concat(
        [chunk[chunk['artist_name'].isin(artist_names)]
        for chunk in pd.read_csv(spotify_dataset, usecols=features, chunksize=10000)],
        ignore_index=True
    )

    artist_avg = artist_data.groupby('artist_name').mean(numeric_only=True).reset_index()
    artist_avg['track_name'] = artist_avg['artist_name'] + '_average'

    all_data = pd.concat([artist_data, artist_avg], ignore_index=True)

    audio_features = []
    for _, row in recent_data.iterrows():
        track, artist = row['track_name'], row['artist']
        match = all_data[(all_data['track_name'] == track) & (all_data['artist_name'] == artist)]
        if match.empty:
            match = all_data[(all_data['track_name'] == f'{artist}_average')]
        if not match.empty:
            track_data = match.iloc[0].copy()
            track_data['played_at'] = row['played_at']
            audio_features.append(track_data)

    audio_features = pd.DataFrame(audio_features)
    audio_features.to_csv('audio_features.csv', index=False)
    print('Audio features updated. Saved to "audio_features.csv"')
    return audio_features

def get_data():
    """
    Retrieves the scaled audio features from recent listening history.
    - Gets listening history from Spotify
    - Gets audio features of tracks in history from dataset
    - Scales numeric features
    """
    recent_data = get_user_history()
    audio_features = get_audio_features(recent_data)
    numeric_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo', 'loudness']
    # Scale numeric audio features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(audio_features[numeric_features])
    audio_features[numeric_features] = scaled_data

    return audio_features, numeric_features, scaler

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

def unweighted_fuzzy_kmeans(audio_features, num_clusters, max_iter=100, error=1e-4):
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
    centers, U, _, _, _, _, _ = fuzz.cluster.cmeans(X, c=num_clusters, m=2.0, error=error, maxiter=max_iter, init=None)
    
    labels = np.argmax(U, axis=0)
    weights = np.ones((num_clusters, X.shape[0])) / X.shape[0]  # Uniform weights

    return labels, centers, weights, U.T

def recommend_tracks(numeric_features, centers, weights, scaler, num_tracks):
    """
    Recommends tracks based on cluster centers and feature weights
    """
    recommendations = [[] for _ in range(len(centers))]
    for chunk in pd.read_csv(spotify_dataset, chunksize=10000):
        scaled = scaler.transform(chunk[numeric_features])

        for i, (_, row) in enumerate(chunk.iterrows()):
            x = scaled[i]
            for i, center in enumerate(centers):
                diff = x - center
                dist = np.sum((weights[i] * diff)**2)

                track_info = (row['track_name'], row['artist_name'])
                if len(recommendations[i]) < num_tracks:
                    heapq.heappush(recommendations[i], (-dist, track_info))
                elif -dist > recommendations[i][0][0]:
                    heapq.heappushpop(recommendations[i], (-dist, track_info))
    for i in range(len(recommendations)):
        recommendations[i] = [track_info for _, track_info in sorted(recommendations[i])]
    
    return recommendations

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

def run(num_tracks=20, num_clusters=5):
    audio_features, numeric_features, scaler = get_data()
    w_labels, w_centers, w_weights, w_partitions = feature_weighted_fuzzy_kmeans(audio_features[numeric_features], num_clusters=num_clusters)
    uw_labels, uw_centers, uw_weights, uw_partitions = unweighted_fuzzy_kmeans(audio_features[numeric_features], num_clusters=num_clusters)
    compare_clusters(audio_features, numeric_features, w_labels, w_weights, uw_labels, uw_weights)
    w_recommended = recommend_tracks(numeric_features, w_centers, w_weights, scaler, num_tracks)
    uw_recommended = recommend_tracks(numeric_features, uw_centers, uw_weights, scaler, num_tracks)
    print('Feature weighted recommendations')
    for i, cluster_recommendations in enumerate(w_recommended):
        print(f'Cluster {i+1}:')
        for j, track_info in enumerate(cluster_recommendations):
            print(f'\t{j+1}. {track_info}')
    print('Feature unweighted recommendations')
    for i, cluster_recommendations in enumerate(uw_recommended):
        print(f'Cluster {i+1}:')
        for j, track_info in enumerate(cluster_recommendations):
            print(f'\t{j+1}. {track_info}')

if __name__ == "__main__":
    print('meow')
    run()