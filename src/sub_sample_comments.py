import pandas as pd
import random
from datetime import datetime, timedelta, timezone

def filter_and_sample_comments(file_path, timestamp, time_range, number_of_comments, output_file):
    try:
        df = pd.read_csv(file_path)

        if 'comment_date' not in df.columns:
            raise ValueError("The CSV file must have a 'comment_date' column.")

        df['comment_date'] = pd.to_datetime(df['comment_date'], errors='coerce', utc=True)

        if isinstance(timestamp, str):
            ref_time = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
        else:
            ref_time = timestamp.replace(tzinfo=timezone.utc)

        start_time = ref_time - timedelta(days=time_range)

        filtered_df = df[(df['comment_date'] >= start_time) & (df['comment_date'] < ref_time)]

        if len(filtered_df) > number_of_comments:
            sampled_df = filtered_df.sample(n=number_of_comments, random_state=42, replace=False)
        else:
            sampled_df = filtered_df

        sampled_df.to_csv(output_file, index=False)

        print(f"Filtered and sampled data saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# ISO 8601 formatted timestamp
timestamp = "2024-09-02T00:00:00+00:00"
channel = "CNN"

filter_and_sample_comments(
    f"/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/{channel}_comments_clean.csv", 
    timestamp, 
    70, 
    300000, 
    f"/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Sub_samples/{channel}_{timestamp}.csv"
)