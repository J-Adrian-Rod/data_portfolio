{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "815cf8af",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "1e1168ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import json\n",
    "from typing import List\n",
    "import os\n",
    "from os import listdir\n",
    "import matplotlib.pyplot as plt\n",
    "import spotipy\n",
    "import spotipy.util as util\n",
    "from spotipy.oauth2 import SpotifyClientCredentials\n",
    "import plotly.express as px\n",
    "import plotly.io as pio\n",
    "from datetime import datetime\n",
    "import pytz \n",
    "from collections import Counter\n",
    "from sklearn.cluster import KMeans\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af1a8b4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: get_streaming\n",
    "\n",
    "# Description:\n",
    "# Scans a specified directory for files starting with 'Streaming_History_Audio_', \n",
    "# loads each file's JSON contents, \n",
    "# aggregates the streaming records from all files into a single list of dictionaries, \n",
    "# and returns the full combined history as structured data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4c82d4e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_streamings(path: str = 'data') -> List[dict]:\n",
    "    \"\"\"\n",
    "    Retrieves Spotify streaming history from the specified directory.\n",
    "\n",
    "    Parameters:\n",
    "        path (str): The directory path where Spotify streaming history data files are located.\n",
    "\n",
    "    Returns:\n",
    "        List[dict]: A list of dictionaries containing streaming history data.\n",
    "    \"\"\"\n",
    "\n",
    "    # List all files in the specified directory with the correct naming scheme\n",
    "    files = [f'{path}/{x}' for x in listdir(path) if x.startswith('Streaming_History_Audio_')]\n",
    "\n",
    "    all_streamings = []\n",
    "\n",
    "    for file in files:\n",
    "        with open(file, 'r', encoding='utf-8') as f:\n",
    "            new_streamings = json.load(f)\n",
    "            all_streamings += new_streamings\n",
    "\n",
    "    return all_streamings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40e4ef3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: minsec_to_seconds\n",
    "\n",
    "# Description:\n",
    "# Parses a time string in \"minutes:seconds\" format, \n",
    "# converts it to total seconds as an integer, \n",
    "# and returns 0 if the input is invalid or an error occurs during parsing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d4f1cda6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def minsec_to_seconds(time_str):\n",
    "    try:\n",
    "        minutes, seconds = map(int, time_str.split(\":\"))\n",
    "        return minutes * 60 + seconds\n",
    "    except:\n",
    "        return 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8204605",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: format_time\n",
    "\n",
    "# Description:\n",
    "# Takes an integer number of seconds, \n",
    "# converts it into a \"minutes:seconds\" string format (MM:SS), \n",
    "# and ensures the seconds portion is zero-padded to two digits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "13d84b1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def format_time(seconds):\n",
    "    return f\"{seconds // 60}:{seconds % 60:02d}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c841f9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: generate_top_10_songs_by_year\n",
    "\n",
    "# Description:\n",
    "# Defines Spotify Wrapped cutoff dates from 2017 to 2024, \n",
    "# filters streaming data for each year based on those cutoffs, \n",
    "# groups tracks by name, artist, album, and URI to preserve metadata, \n",
    "# aggregates play counts and listening duration, \n",
    "# ranks the top 10 songs by play count per year, \n",
    "# formats listening time as MM:SS, \n",
    "# and returns a dictionary mapping each year to its corresponding top 10 songs DataFrame."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2b68cd24",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_top_10_songs_by_year(streaming_data):\n",
    "    import pandas as pd\n",
    "\n",
    "    wrapped_end_dates = {\n",
    "        2017: \"2017-10-31T23:59:59Z\",\n",
    "        2018: \"2018-10-31T23:59:59Z\",\n",
    "        2019: \"2019-10-31T23:59:59Z\",\n",
    "        2020: \"2020-11-15T23:59:59Z\",\n",
    "        2021: \"2021-11-15T23:59:59Z\",\n",
    "        2022: \"2022-11-15T23:59:59Z\",\n",
    "        2023: \"2023-11-15T23:59:59Z\",\n",
    "        2024: \"2024-11-15T23:59:59Z\"\n",
    "    }\n",
    "\n",
    "    streaming_data['ts'] = pd.to_datetime(streaming_data['ts'])\n",
    "    top_10_songs_by_year = {}\n",
    "\n",
    "    for year, end_str in wrapped_end_dates.items():\n",
    "        print(f\"Processing Wrapped {year}...\")\n",
    "\n",
    "        start = pd.Timestamp(f\"{year}-01-01T00:00:00Z\")\n",
    "        end = pd.Timestamp(end_str)\n",
    "\n",
    "        year_data = streaming_data[\n",
    "            (streaming_data['ts'] >= start) & (streaming_data['ts'] <= end)\n",
    "        ]\n",
    "\n",
    "        if year_data.empty:\n",
    "            top_10_songs_by_year[year] = pd.DataFrame()\n",
    "            continue\n",
    "\n",
    "        # Group by full identity of a track to keep metadata\n",
    "        top_songs = (\n",
    "            year_data\n",
    "            .groupby([\n",
    "                'master_metadata_track_name',\n",
    "                'master_metadata_album_artist_name',\n",
    "                'master_metadata_album_album_name',\n",
    "                'spotify_track_uri'\n",
    "            ])\n",
    "            .agg({\n",
    "                'ts': 'count',\n",
    "                'ms_played': 'sum'\n",
    "            })\n",
    "            .reset_index()\n",
    "            .rename(columns={'ts': 'play_count'})\n",
    "            .sort_values(by='play_count', ascending=False)\n",
    "            .head(10)\n",
    "        )\n",
    "\n",
    "        # Add formatted listening length\n",
    "        top_songs['listening_length'] = top_songs['ms_played'].apply(\n",
    "            lambda ms: f\"{int(ms // 60000)}:{int((ms % 60000) // 1000):02d}\"\n",
    "        )\n",
    "\n",
    "        top_10_songs_by_year[year] = top_songs\n",
    "\n",
    "    return top_10_songs_by_year\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ccef020",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: save_cache\n",
    "\n",
    "# Description:\n",
    "# Writes the in-memory `cache` dictionary to a JSON file defined by `CACHE_FILE`, \n",
    "# using indentation for readability, to preserve updated artist-genre mappings for future use."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2ad3098",
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_cache():\n",
    "    with open(CACHE_FILE, 'w') as f:\n",
    "        json.dump(cache, f, indent=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b49ecab3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: search_track_id\n",
    "\n",
    "# Description:\n",
    "# Constructs a search query using the provided artist and track name, \n",
    "# queries the Spotify API for matching tracks, \n",
    "# returns the Spotify track ID if a result is found, \n",
    "# and handles errors gracefully by printing the exception and returning `None`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "5d0bae5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def search_track_id(artist, track):\n",
    "    try:\n",
    "        query = f\"artist:{artist} track:{track}\"\n",
    "        result = sp.search(q=query, type='track', limit=1)\n",
    "        items = result['tracks']['items']\n",
    "        if items:\n",
    "            return items[0]['id']\n",
    "        else:\n",
    "            return None\n",
    "    except Exception as e:\n",
    "        print(f\"Error searching for {track} by {artist}: {e}\")\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c292d76d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: get_audio_features\n",
    "\n",
    "# Description:\n",
    "# Checks if the audio features for a given Spotify track ID exist in the local cache, \n",
    "# queries the Spotify API if not cached, \n",
    "# stores the retrieved features in the cache and saves it to file, \n",
    "# and returns the feature data or `None` if the track is not found or an error occurs.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "2b6ff903",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_audio_features(track_id):\n",
    "    try:\n",
    "        if track_id in cache:\n",
    "            return cache[track_id]\n",
    "\n",
    "        features = sp.audio_features([track_id])\n",
    "        if features and features[0]:\n",
    "            cache[track_id] = features[0]\n",
    "            save_cache()\n",
    "            return features[0]\n",
    "        else:\n",
    "            return None\n",
    "    except Exception as e:\n",
    "        print(f\"Error fetching features for track ID {track_id}: {e}\")\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5652d1f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: load_wrapped_data\n",
    "\n",
    "# Description:\n",
    "# Reads a CSV file containing Spotify Wrapped streaming data, \n",
    "# filters out entries beyond the year 2024, \n",
    "# and returns the cleaned DataFrame for further analysis.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b40fb0ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_wrapped_data(path):\n",
    "    df = pd.read_csv(path)\n",
    "    df = df[df['year'] <= 2024]\n",
    "    return df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f172ecaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: create_vibe_clusters\n",
    "\n",
    "# Description:\n",
    "# Applies KMeans clustering to four Spotify audio features—energy, danceability, valence, and tempo— \n",
    "# assigns each track a cluster label stored in a new column `vibe_cluster`, \n",
    "# and returns the DataFrame with the added cluster assignments.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ab2cb3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_vibe_clusters(df):\n",
    "    features = ['energy', 'danceability', 'valence', 'tempo']\n",
    "    kmeans = KMeans(n_clusters=4, random_state=42)\n",
    "    df['vibe_cluster'] = kmeans.fit_predict(df[features])\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e7911b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: get_feature_trend\n",
    "\n",
    "# Description:\n",
    "# Groups the input DataFrame by year and calculates the average of a selected audio feature, \n",
    "# then creates a Plotly line chart with markers showing how that feature changes over time, \n",
    "# and returns the resulting figure for display or embedding in a dashboard."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "3305a8ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_feature_trend(df, selected_feature):\n",
    "    trend = df.groupby('year')[selected_feature].mean().reset_index()\n",
    "    fig = px.line(trend, x='year', y=selected_feature, markers=True, \n",
    "                  title=f\"Average {selected_feature.capitalize()} Over Time\")\n",
    "    return fig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "803129dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: generate_top_10_table\n",
    "\n",
    "# Description:\n",
    "# Sorts the DataFrame in descending order based on a given audio feature, \n",
    "# selects the top 10 tracks, \n",
    "# extracts track name, artist name, and the selected feature value, \n",
    "# and returns the result as a list of dictionaries for easy use in interactive tables or dashboards.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "3070aa98",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_top_10_table(df, feature):\n",
    "    top_10 = df.sort_values(by=feature, ascending=False).head(10)\n",
    "    return top_10[[\n",
    "        'master_metadata_track_name',\n",
    "        'master_metadata_album_artist_name',\n",
    "        feature\n",
    "    ]].to_dict(\"records\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e94ccbb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Title: load_full_wrapped_data\n",
    "\n",
    "# Description:\n",
    "# Loads cleaned Spotify streaming history and audio feature metadata, \n",
    "# merges them by track name and artist, \n",
    "# maps artist genres using a preprocessed genre file (`top_genres_clean.json`), \n",
    "# extracts the primary genre from the list if applicable, \n",
    "# filters the data to include only entries through 2024, \n",
    "# and returns the fully enriched DataFrame for analysis or visualization.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "fb35ba28",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def load_full_wrapped_data():\n",
    "    \"\"\"\n",
    "    Loads the full merged Spotify streaming data and Spotify features.\n",
    "    Also maps genre info using the top_genres_clean.json file.\n",
    "    \"\"\"\n",
    "    # Load data\n",
    "    streaming_data = pd.read_csv('wrapped_data/streaming_data.csv')\n",
    "    spotify_features = pd.read_csv('wrapped_data/SpotifyFeatures.csv')\n",
    "\n",
    "    # Merge on track + artist\n",
    "    merged = pd.merge(\n",
    "        streaming_data,\n",
    "        spotify_features,\n",
    "        how='inner',\n",
    "        left_on=['master_metadata_track_name', 'master_metadata_album_artist_name'],\n",
    "        right_on=['track_name', 'artist_name']\n",
    "    )\n",
    "\n",
    "    # Load genre map\n",
    "    with open('wrapped_data/top_genres_clean.json', 'r') as f:\n",
    "        genre_map = json.load(f)\n",
    "\n",
    "    # Map genres\n",
    "    merged['genre'] = merged['master_metadata_album_artist_name'].map(genre_map)\n",
    "    merged['genre'] = merged['genre'].apply(lambda g: g[0] if isinstance(g, list) else g)\n",
    "\n",
    "    # Only keep data up to 2024\n",
    "    merged = merged[merged['year'] <= 2024]\n",
    "\n",
    "    return merged"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
