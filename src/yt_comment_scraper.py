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
import re
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

def get_video_comments(video_url, api_key):
    """
    Fetches all comments from a specified YouTube video given its URL.

    Parameters:
    - video_url (str): URL of the YouTube video.
    - api_key (str): YouTube API key.

    Returns:
    - pd.DataFrame: DataFrame containing comments.
    """
    # Extract video ID from URL
    video_id = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id:
        raise ValueError("Invalid YouTube video URL.")
    video_id = video_id.group(1)

    comments_data = []
    youtube = build('youtube', 'v3', developerKey=api_key)

    try:
        # Retrieve comments
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
                    #"username": username,
                    "video_id": video_id,
                    "comment": comment,
                    "comment_date": comment_date,
                })

            # Check for the next page of comments
            if 'nextPageToken' in comment_response:
                comment_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=comment_response['nextPageToken'],
                    maxResults=100
                ).execute()
            else:
                break

        # Convert comments to DataFrame
        return pd.DataFrame(comments_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Comments so far: {len(comments_data)} ...")
        print(f"Returning collected comments so far ...")

        return pd.DataFrame([])  # Return an empty DataFrame in case of errors

def process_videos(input_df, api_key):
    """
    Processes a list of YouTube video links and scrapes comments.

    Parameters:
    - input_df (pd.DataFrame): DataFrame with columns 'Title', 'Date Posted', and 'Link'
    - api_key (str): YouTube API key.

    Returns:
    - pd.DataFrame: Combined DataFrame of all comments scraped.
    """
    all_comments = []

    # Ensure 'Date Posted' is in datetime format
    input_df['Date Posted'] = pd.to_datetime(input_df['Date Posted'])

    # Filter for videos posted in 2024 only
    grouped_df = input_df[input_df['Date Posted'].dt.year == 2024]

    print("Processing videos from the year 2024...")

    for link in grouped_df['Link']:
        if link:
            comments_df = get_video_comments(link, api_key)
            all_comments.append(comments_df)
        else:
            print(f"Invalid link: {link}")

    # Combine all comments into a single DataFrame
    return pd.concat(all_comments, ignore_index=True)

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Load environment variables from .env file located in the same directory as this script
    load_dotenv()  

    # Load API key from environment variable
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API key not found. Please add your API key to the .env file.")
        exit(1)

    # Load video links from CSV file
    channel_of_interest = 'CNN'
    input_data_path = os.path.join(script_dir, f"../data/Raw Data/{channel_of_interest}_links.csv")
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
    output_file = os.path.join(script_dir, f"../data/Raw Data/{channel_of_interest}_comments.csv")
    final_df.to_csv(output_file, index=False)
    print(f"Comments saved to {output_file}")

if __name__ == "__main__":
    main()