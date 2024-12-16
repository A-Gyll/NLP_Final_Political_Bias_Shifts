import pandas as pd
from datetime import datetime, timezone

def combine_datasets(fp_1, fp_2, output_file):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(fp_1)
    df2 = pd.read_csv(fp_2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output_file, index=False)

def standardize_timestamp(ts):
    try:
        if '+' in ts and ' ' in ts:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S%z")
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        return ts
    except Exception as e:
        print(f"Error processing timestamp: {ts}, Error: {e}")
        return None

if __name__ == "__main__":
    # File paths for the two CSV files
    # file1 = "/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/BBC_News_comments_clean.csv" 
    # file2 = "/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/The_Hill_comments_clean.csv"  
    # output_file = "/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/Combined_Neutral_comments_clean.csv"
    # combine_datasets(file1, file2, output_file)

    # Use this if timestamp is messed up
    dataset = "Combined_Neutral"
    file_path = f'/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/{dataset}_comments_clean.csv'
    df = pd.read_csv(file_path)
    timestamp_column = 'comment_date' 
    df[timestamp_column] = df[timestamp_column].apply(standardize_timestamp)
    df.to_csv(f"/home/ujx4ab/ondemand/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/{dataset}_comments_clean_2.csv", index=False)