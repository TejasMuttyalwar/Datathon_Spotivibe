from flask import Flask, render_template, request
import pandas as pd
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Spotify API Setup
SPOTIPY_CLIENT_ID = "YOUR CLIENT ID"
SPOTIPY_CLIENT_SECRET = "YOUR SECRET ID"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

# Load ML models
models = {
    "SVM": pickle.load(open("models/SVM_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/Decision Tree_model.pkl", "rb")),
    "KNN": pickle.load(open("models/K-Nearest Neighbors_model.pkl", "rb")),
    "Random Forest": pickle.load(open("models/Random Forest_model.pkl", "rb")),
}

# Load popularity predictor
popularity_model = pickle.load(open("models/popularity_predictor.pkl", "rb"))

# Load dataset
df = pd.read_csv("D:\\new\\spotify_with_categories (1).csv")
  # Make sure this path is correct

# Mood to features mapping
mood_to_features = {
    "happy": {"valence": 0.9, "energy": 0.8, "danceability": 0.7, "mode": 1, "tempo": 130},
    "sad": {"valence": 0.2, "energy": 0.3, "acousticness": 0.7, "mode": 0, "tempo": 80},
    "chill": {"valence": 0.5, "energy": 0.4, "instrumentalness": 0.6, "mode": 1, "tempo": 100},
    "energetic": {"valence": 0.8, "energy": 0.9, "tempo": 140, "mode": 1},
    "romantic": {"valence": 0.7, "acousticness": 0.5, "speechiness": 0.2, "mode": 1, "tempo": 110}
}

GENRES = ["pop", "rock", "hip-hop", "jazz", "classical", "electronic", 
          "r&b", "country", "reggae", "metal", "indie", "folk"]

@app.route("/")
def home():
    return render_template("index.html", genres=GENRES)

@app.route("/find_my_vibe", methods=["GET", "POST"])
def find_my_vibe():
    if request.method == "POST":
        mood = request.form.get("mood")
        genre = request.form.get("genre")
        tempo_pref = request.form.get("tempo")

        base_features = {
            'duration_ms': 200000, 'explicit': 1, 'danceability': 0.5, 'energy': 0.5,
            'key': 5, 'loudness': -10.0, 'mode': 1, 'speechiness': 0.1,
            'acousticness': 0.5, 'instrumentalness': 0.0, 'liveness': 0.2,
            'valence': 0.5, 'tempo': 120, 'time_signature': 4, 'popularity': 50
        }

        base_features.update(mood_to_features.get(mood, {}))
        base_features['track_genre'] = genre

        if tempo_pref == "slow":
            base_features["tempo"] = min(100, base_features["tempo"])
        elif tempo_pref == "fast":
            base_features["tempo"] = max(130, base_features["tempo"])

        input_df = pd.DataFrame([base_features])

        predictions = {}
        for model_name, model in models.items():
            try:
                predictions[model_name] = model.predict(input_df)[0]
            except ValueError as e:
                print(f"Error in {model_name}: {e}")

        try:
            pop_pred = popularity_model.predict(input_df)[0]
        except Exception as e:
            print(f"Error in popularity predictor: {e}")
            pop_pred = None
        predictions["Popularity"] = pop_pred

        filtered = df[df['track_genre'] == genre]
        if tempo_pref == "slow":
            filtered = filtered[filtered['tempo'] < 110]
        elif tempo_pref == "fast":
            filtered = filtered[filtered['tempo'] > 130]
        filtered = filtered.drop_duplicates(subset="track_name")

        songs = []
        for _, row in filtered.head(10).iterrows():
            song_name = row['track_name']
            artist_name = row['artists']
            try:
                spotify_data = sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
                if spotify_data["tracks"]["items"]:
                    track = spotify_data["tracks"]["items"][0]
                    spotify_url = track["external_urls"]["spotify"]
                    album_art = track["album"]["images"][0]["url"] if track["album"]["images"] else None
                    preview_url = track["preview_url"]
                    spotify_uri = track["id"]
                else:
                    spotify_url = album_art = preview_url = spotify_uri = None
            except Exception as e:
                print(f"Error fetching Spotify data: {e}")
                spotify_url = album_art = preview_url = spotify_uri = None

            songs.append({
                "track_name": song_name,
                "artists": artist_name,
                "spotify_url": spotify_url,
                "album_art": album_art,
                "preview_url": preview_url,
                "spotify_uri": spotify_uri,
                "predictions": predictions
            })

        return render_template("vibe_results.html", mood=mood, genre=genre, tempo=tempo_pref, songs=songs)

    # ✅ FIX: Pass mood_to_features on GET
    return render_template("find_my_vibe.html", genres=GENRES, mood_to_features=mood_to_features)

@app.route("/popular")
def popular():
    sorted_df = df.sort_values(by="popularity", ascending=False).drop_duplicates(subset="track_name")
    songs = []
    for _, row in sorted_df.head(10).iterrows():
        song_name = row['track_name']
        artist_name = row['artists']
        try:
            spotify_data = sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
            if spotify_data["tracks"]["items"]:
                track = spotify_data["tracks"]["items"][0]
                spotify_url = track["external_urls"]["spotify"]
                album_art = track["album"]["images"][0]["url"] if track["album"]["images"] else None
                preview_url = track["preview_url"]
                spotify_uri = track["id"]
            else:
                spotify_url = album_art = preview_url = spotify_uri = None
        except Exception as e:
            print(f"Error fetching Spotify data: {e}")
            spotify_url = album_art = preview_url = spotify_uri = None

        songs.append({
            "track_name": song_name,
            "artists": artist_name,
            "spotify_url": spotify_url,
            "album_art": album_art,
            "preview_url": preview_url,
            "spotify_uri": spotify_uri,
            "predictions": {"Popularity": row['popularity']}
        })

    return render_template("vibe_results.html", mood="Popular Music", genre="All", tempo="N/A", songs=songs)

if __name__ == "__main__":
    app.run(debug=True)
