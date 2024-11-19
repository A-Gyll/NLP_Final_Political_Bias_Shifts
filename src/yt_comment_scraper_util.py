import os
import re
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
import csv

class YoutubeCommentScraper:
  def __init__(self, api_key):
    self.youtube_api = build('youtube', 'v3', developerKey=api_key)
  
  def get_video_comments(self, video_id):
    comments_data = []
    try:
      # comment query
      comment_response = self.youtube_api.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
            ).execute()

            # while there is a response fromt he query, process the comments
      while comment_response:
        for item in comment_response['items']:
          comment_snippet = item['snippet']['topLevelComment']['snippet']
          comments_data.append({
              "video_id": video_id,
              "comment": comment_snippet['textDisplay'],
              "username": comment_snippet['authorDisplayName'],
              "comment_date": comment_snippet['publishedAt'],  # extract 'YYYY-MM-DD'
          })

        # check for the next page of comments
        if 'nextPageToken' in comment_response:
            comment_response = self.youtube_api.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=comment_response['nextPageToken'],
                maxResults=100
            ).execute()
        else:
            break
      return comments_data, None
    except Exception as e:
      error_content = e.content.decode('utf-8')  # Decode the error response
      if 'quotaExceeded' in error_content:
          print("Quota limit exceeded.")
          print("Last video_id: ", video_id)
          return None, "Quota Exceeded"
      else:
          print(f"Some other error occurred: {error_content}")
      return None, str(e)
    
class CommentScraperUtil:
  def load_video_ids(path_to_link_csv, start_date, most_recent_video_id=None):
    links_df = pd.read_csv(path_to_link_csv)#, quoting=csv.QUOTE_ALL, escapechar='\\')
    links_df['Date Posted'] = pd.to_datetime(links_df['Date Posted'], format='mixed')
    
    #links_df = links_df.sort_values(by='Date Posted')
    mask = links_df['Date Posted'] >= pd.to_datetime(start_date, format='mixed').tz_localize('UTC')
    filtered_links = links_df.loc[mask]

    video_ids = []
    for _, row in filtered_links.iloc[::-1].iterrows():
        link = row['Link']
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", link)
        if match:
            video_id = match.group(1)
            if most_recent_video_id and video_id == most_recent_video_id:
                break
            video_ids.append(video_id)
    video_ids.reverse()
    return video_ids
  
  def get_earliest_video_id(path_to_comment_csv):
    if not os.path.exists(path_to_comment_csv):
        return None, None
    comments_df = pd.read_csv(path_to_comment_csv)#, quoting=csv.QUOTE_ALL, escapechar='\\')
    comments_df['comment_date'] = pd.to_datetime(comments_df['comment_date'], format='mixed', errors='coerce')
    
    # Get the last row of the dataframe
    last_row = comments_df.iloc[-1]
    
    return last_row['video_id'], comments_df
  





  