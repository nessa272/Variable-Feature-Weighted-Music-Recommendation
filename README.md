# Spotify Music Clustering & Recommendation

This project uses the Spotify Web API and clustering to analyze listening data and provide music recommendations based on clustered preferences.

## 🛠 Requirements

- Python 3.8+
- Spotify Developer account (to get API credentials)

### Python libraries used:

- `python-dotenv`
- `spotipy`
- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-fuzzy`
- `heapq`

## 📦 Setup Instructions
### Set up Spotify API keys in .env
```env
CLIENT_ID=your_spotify_client_id
CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI='http://127.0.0.1:5000/callback'