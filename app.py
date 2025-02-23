import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials (consider storing these securely, e.g., in environment variables)
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    """Return the album cover URL for a given song and artist using Spotify's API."""
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        # Return a default placeholder image URL if no results are found
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song):
    """Recommend 5 songs based on the given song using a precomputed similarity matrix."""
    # Get the index of the selected song
    index = music[music['song'] == song].index[0]
    # Retrieve and sort the similarity scores for this song
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_music_names = []
    recommended_music_posters = []
    
    # Skip the first entry as it is the song itself; take the next 5 songs
    for i in distances[1:6]:
        rec_song = music.iloc[i[0]].song
        rec_artist = music.iloc[i[0]].artist
        poster_url = get_song_album_cover_url(rec_song, rec_artist)
        recommended_music_names.append(rec_song)
        recommended_music_posters.append(poster_url)
        
    return recommended_music_names, recommended_music_posters

# App title
st.header('Music Recommender System')

# Load the preprocessed dataset and similarity matrix
try:
    music = pickle.load(open('df.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# Dropdown list for song selection
music_list = music['song'].values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)

# Display recommendations when the button is clicked
if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters = recommend(selected_song)
    # Create 5 columns for the recommended songs
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(recommended_music_names[idx])
            st.image(recommended_music_posters[idx])
