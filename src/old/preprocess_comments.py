"""
YouTube Comments Data Preprocessing Script
------------------------------------------

This script preprocesses a dataset of YouTube comments, performing the following tasks:
- Cleans text data by removing URLs, special characters, numbers, and extra spaces.
- Removes common English stopwords using NLTK's stopword list and additional custom-defined words.
- Saves the cleaned data to a new CSV file.

**Prerequisites:**
1. Python 3.x installed on your system.
2. Required Libraries:
   - pandas
   - nltk
   - re (standard library, no installation required)
   - datetime (standard library, no installation required)

**Installation Instructions:**
To install the required libraries, run the following command in your terminal:

```bash
pip install pandas nltk

"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk_stop_words = set(stopwords.words('english'))

# Define additional common stopwords
common_stop_words = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", 
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
    "should", "now"
}

def load_data(file_path):
    """
    Loads and inspects the dataset from the given CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    print("Raw YouTube Comments dataset info:")
    df.info()
    return df

def preprocess_data(df):
    """
    Cleans and preprocesses the dataset by handling missing values, 
    converting data types, and applying text cleaning and stopword removal.

    Parameters:
    - df (pd.DataFrame): The raw DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Drop rows with missing values in essential columns
    print("Dropping rows with missing values...")
    df.dropna(subset=['comment', 'comment_date', 'username'], inplace=True)

    # Convert 'video_publish_year' to integer and 'comment_date' to datetime format
    print("Reformatting columns...")
    df['video_publish_year'] = df['video_publish_year'].astype(int)
    df['comment_date'] = pd.to_datetime(df['comment_date'], errors='coerce')
    
    # Apply text cleaning
    print("Cleaning the comments...")
    df['comment'] = df['comment'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)

    # Apply stopword removal
    print("Removing stopwords...")
    df['comment'] = df['comment'].apply(lambda x: remove_stopwords(x, nltk_stop_words | common_stop_words) if isinstance(x, str) else x)
    
    return df

def clean_text(text):
    """
    Cleans text by removing URLs, special characters, numbers, and extra spaces.

    Parameters:
    - text (str): The raw text to be cleaned.

    Returns:
    - str: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def remove_stopwords(text, stop_words):
    """
    Removes stopwords from the text based on a specified set of stop words.

    Parameters:
    - text (str): Text from which to remove stopwords.
    - stop_words (set): Set of stop words to be removed.

    Returns:
    - str: Text with stop words removed.
    """
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def save_data(df, cleaned_file_path):
    """
    Saves the cleaned dataset to a CSV file.

    Parameters:
    - df (pd.DataFrame): The cleaned DataFrame.
    - cleaned_file_path (str): Path where the cleaned CSV file will be saved.
    """
    df.to_csv(cleaned_file_path, index=False)
    print(f"Data cleaned and saved to {cleaned_file_path}")

def main():
    # Specify file paths
    file_path = '../data/yt_comments_raw.csv'
    cleaned_file_path = '../data/yt_comments_clean.csv'

    # Load, preprocess, and save data
    df = load_data(file_path)
    cleaned_df = preprocess_data(df)
    save_data(cleaned_df, cleaned_file_path)

if __name__ == "__main__":
    main()