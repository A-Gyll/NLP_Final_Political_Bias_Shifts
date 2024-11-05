"""
YouTube Comments Scraper Script
-------------------------------

This script scrapes comments from a list of YouTube videos based on provided URLs and topics. It retrieves the
following information for each comment:
- Video metadata: channel title, video title, video publish year, and topic.
- Comment details: username, comment text, and comment date.

The script then saves all scraped comments to a raw CSV file for further preprocessing.

**Script Workflow:**
1. Reads a CSV file containing YouTube video URLs and associated topics.
2. Fetches comments for each video using the YouTube Data API.
3. Saves the combined dataset of comments to a CSV file.

**Prerequisites:**
1. Python 3.x installed on your system.
2. Required Libraries:
   - pandas
   - google-auth (for API access)
   - google-auth-oauthlib (for API access)
   - google-api-python-client
   - python-dotenv (to load environment variables)
   
3. YouTube Data API key saved in a `.env` file in the same directory as this script.

**Installation Instructions:**
To install the required libraries, run the following command in your terminal:

```bash
pip install pandas google-auth google-auth-oauthlib google-api-python-client python-dotenv

"""

import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

def get_video_comments(video_id, api_key):
    """
    Fetches comments from a specified YouTube video.

    Parameters:
    - video_id (str): ID of the YouTube video.
    - topic (str): Topic associated with the video.
    - api_key (str): YouTube API key.

    Returns:
    - pd.DataFrame: DataFrame containing comments and metadata.
    """
    comments_data = []

    # Initialize YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get video details
    try:
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        video_info = video_response['items'][0]['snippet']
        video_publish_year = video_info['publishedAt'][:4]
        video_title = video_info['title']
        channel_title = video_info['channelTitle']
    except Exception as e:
        print(f"Error fetching video details for video ID {video_id}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

    # Retrieve comments
    try:
        comment_response = youtube.commentThreads().list(
            part='snippet,replies', videoId=video_id, maxResults=100
        ).execute()

        # Process each page of comments
        while comment_response:
            for item in comment_response['items']:
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comment = comment_snippet['textDisplay']
                username = comment_snippet['authorDisplayName']
                comment_date = comment_snippet['publishedAt'][:10]  # Extract 'YYYY-MM-DD'

                # Append data to comments_data list
                comments_data.append({
                    "channel_title": channel_title,
                    "video_title": video_title,
                    "video_publish_year": video_publish_year,
                    "username": username,
                    "comment": comment,
                    "comment_date": comment_date,
                })

            # Check for next page of comments
            if 'nextPageToken' in comment_response:
                comment_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=comment_response['nextPageToken'],
                    maxResults=100
                ).execute()
            else:
                break

    except Exception as e:
        print(f"Error fetching comments for video ID {video_id}: {e}")

    # Convert comments_data to DataFrame
    return pd.DataFrame(comments_data)

def process_videos(input_df, api_key):
    """
    Processes a list of YouTube video links by topic and scrapes comments.

    Parameters:
    - input_df (pd.DataFrame): DataFrame with columns 'Topic' and 'Link'.
    - api_key (str): YouTube API key.

    Returns:
    - pd.DataFrame: Combined DataFrame of all comments scraped.
    """
    all_comments = []
    input_df['Published Date'] = pd.to_datetime(input_df['Published Date'])
    publish_years = input_df['Published Date'].dt.year.unique()

    for year in publish_years:
        print(f"Processing videos from Year: {year}...")
        grouped_df = input_df[input_df['Published Date'].dt.year == year]

        for video_id in grouped_df['Video ID']:
            if video_id:
                comments_df = get_video_comments(video_id, api_key)
                all_comments.append(comments_df)
            else:
                print(f"Invalid video ID: {video_id}")

    return pd.concat(all_comments, ignore_index=True)

def main():

    # Load environment variables from .env file
    load_dotenv()  

    # Load API key from environment variable
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API key not found. Please add your API key to the .env file.")
        exit(1)

    # Load video links from CSV file
    input_data_path = "../data/youtube_channel_political_videos.csv"

    try:
        input_df = pd.read_csv(input_data_path)
        print(f"Loaded {len(input_df)} video links from {input_data_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_data_path}")
        exit(1)

    # Process videos and scrape comments
    final_df = process_videos(input_df, api_key)
    print(f"Total comments scraped: {len(final_df)}")

    # Save the scraped comments to a CSV file
    output_file = "../data/yt_comments_raw.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Comments saved to {output_file}")

if __name__ == "__main__":
    main()
