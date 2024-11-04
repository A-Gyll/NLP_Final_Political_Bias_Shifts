######################
# IMPORTS & ENV
######################

import pandas as pd
from googleapiclient.discovery import build

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

######################
# FUNCTION DEFINITIONS
######################

def get_video_comments(video_id, topic, api_key):
    # List to store comments data
    comments_data = []

    # Create YouTube resource object
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get video details (to retrieve the publish date)
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()

    # Extract video publish year
    video_publish_date = video_response['items'][0]['snippet']['publishedAt']
    video_publish_year = video_publish_date[:4]
    video_title = video_response['items'][0]['snippet']['title']
    channel_title = video_response['items'][0]['snippet']['channelTitle']

    # Retrieve YouTube video comments
    comment_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=100
    ).execute()

    # Iterate through the comment response
    while comment_response:
        for item in comment_response['items']:
            # Extract top-level comment, username, and comment date
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment_date = item['snippet']['topLevelComment']['snippet']['publishedAt'][:10]  # Extract date in 'YYYY-MM-DD' format
            
            #-------------Logic Left Here in case we need to pull comment replies------------------------
            # replies = []
            # If there are replies, extract each reply, username, and date
            # if item['snippet']['totalReplyCount'] > 0:
            #     for reply in item['replies']['comments']:
            #         reply_text = reply['snippet']['textDisplay']
            #         reply_user = reply['snippet']['authorDisplayName']
            #         reply_date = reply['snippet']['publishedAt'][:10]
            #         replies.append({"reply_user": reply_user, "reply_text": reply_text, "reply_date": reply_date})
            #--------------------------------------------------------------------------------

            # Append data to comments_data list
            comments_data.append({
                "topic": topic,
                "channel_title": channel_title,
                "video_title": video_title,
                "video_publish_year": video_publish_year,
                "username": username,
                "comment": comment,
                "comment_date": comment_date,
                # "replies": replies
            })

        # Check if there are more pages of comments
        if 'nextPageToken' in comment_response:
            comment_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=comment_response['nextPageToken'],
                maxResults=100
            ).execute()
        else:
            break

    # Create a DataFrame from comments_data
    df = pd.DataFrame(comments_data)
    return df

######################
#       MAIN
######################

if __name__ == "__main__":

    api_key = os.getenv("API_KEY") # API key in your .env file
    input_data_path = "clean_nlp_data.csv"

    final_df = pd.DataFrame()
    all_dfs = []

    # Read csv file that contains URLs
    with open(input_data_path, 'r') as file:
        input_df = pd.read_csv(file)
        print(f"Pulled {len(input_df)} YouTube videos from file...")

    unique_topics = input_df['Topic'].unique()
    print(f"Unique Topics: {unique_topics}")

    for topic in unique_topics:
        print(f"Processing videos from Topic: {topic}...")
        df = input_df[input_df['Topic'] == topic]

        for url in df['Link'].tolist():
            print(f"Processing URL: {url}")
            video_id = url.split('v=')[1]
            all_dfs.append(get_video_comments(video_id, topic, api_key))

    final_df = pd.concat(all_dfs, ignore_index=True) 
    print(f"Scraped a total of {len(final_df)} comments.")
    final_df.to_csv('politics_yt_comments.csv', index=False)
