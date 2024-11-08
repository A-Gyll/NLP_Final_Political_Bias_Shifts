
import csv
import os
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.discovery import build
from datetime import datetime
from dataclasses import dataclass
from textwrap import indent
from dotenv import load_dotenv


@dataclass
class query_string:
    NEWS_SOURCE: str
    API_KEY: str
    CHANNEL_ID: str
    PLAYLIST_ID: str
    NEXT_PAGE_TOKEN: None

def get_channel_response(API_resource, CHANNEL_ID):
    channel_response = API_resource.channels().list(
        part='contentDetails',
        id=CHANNEL_ID
    ).execute()
    uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    return uploads_playlist_id

def run_query(resource, query):
  video_links = []

  # Define the path to the data folder
  data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
  os.makedirs(data_folder, exist_ok=True)  # Create the folder if it doesn't exist

  while True:

    request = resource.playlistItems().list(
      part='snippet',
      playlistId=query.PLAYLIST_ID,
      pageToken=query.NEXT_PAGE_TOKEN,
      maxResults = 50
    )
    response = request.execute()

    # Extract video links and titles from response
    for item in response["items"]:
        video_id = item["snippet"]["resourceId"]["videoId"]
        video_title = item["snippet"]["title"]
        video_link = f"https://www.youtube.com/watch?v={video_id}"
        video_date = item["snippet"]["publishedAt"]
        video_links.append([video_title, video_date, video_link])

    query.NEXT_PAGE_TOKEN = response.get("nextPageToken")
    if not query.NEXT_PAGE_TOKEN:
        break # should be at 20,000th video

    # Define the filename with the path to the data folder
    fn = os.path.join(data_folder, f"{query.NEWS_SOURCE}_links.csv")

    # Write links to a CSV file in the data folder
    with open(fn, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Date Posted", "Link"])
        writer.writerows(video_links)


  print("Done creating: " + fn)


def main():
  """
    See Data Descriptions Doc in the Drive (going in order of left-->right)
  """
  load_dotenv()  
  api_key = os.getenv("API_KEY")

  if not api_key:
    print("Error: API key not found. Please add your API key to the .env file.")
    exit(1)
  
  CHANNEL_IDS = [
    'UCaXkIU1QidjPwiAYu6GcHjg', 
    'UCupvZG-5ko_eiXAupbDfxWw', 
    'UCvJJ_dzjViJCoLf5uKUTwoA', 
    'UCK7tptUDHh-RYDsdxO1-5QQ', 
    'UCrvhNP_lWuPIP6QZzJmM-bw',
    'UCw3fku0sH3qA3c3pZeJwdAw',
    'UCCXoCcu9Rp7NPbTzIvogpZg',
    'UCXIJgqnII2ZOINSWNOGFThA'
    ]
  NEWS_SOURCES = [
    'MSNBC', 
    'CNN', 
    'CNBC', 
    'Wall_Street_Journal', 
    'The_New York_Post', 
    'The_Daily_Mail',
    'Fox_Business',
    'Fox_News'
    ]
  
  for news_source, channel_id in zip(NEWS_SOURCES, CHANNEL_IDS):
    query = query_string(NEWS_SOURCE=news_source, API_KEY=api_key, CHANNEL_ID=channel_id, PLAYLIST_ID=None, NEXT_PAGE_TOKEN=None)

    youtube = build('youtube', 'v3', developerKey=query.API_KEY)

    query.PLAYLIST_ID = get_channel_response(youtube, query.CHANNEL_ID)

    query_response = run_query(youtube, query)


if __name__ == "__main__":
    main()