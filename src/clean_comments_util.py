import numpy as np
import pandas as pd
import regex as re
import os
import time
import csv


#https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)

def clean_comments(path_to_csv, save_path):
    time_start = time.time()
    df = pd.read_csv(path_to_csv,encoding='utf-8')

    df['comment'] = df['comment'].astype(str)
    df['comment'] = df['comment'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
    df['comment'] = df['comment'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x)) # remove links
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x)) #remove links

    # leave emojis in for now since llama's tokenizer can handle it
    #df['comment'] = df['comment'].apply(lambda x: re.sub(EMOJI_PATTERN,'',x)) #remove emojis

    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(\x5Co\x2F)",'',x)) #remove \o/ emoji
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(\x5C_\x28\x29_\x2F)",'',x)) #remove \_()_/ emoji

    # for improper html alphanumerical conversions
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(&amp;#8211;)", '–', x)) #endash
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(&amp;)", '&', x))  # & symbol
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(&lt;)", '<', x))  # less than symbol
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(&gt;)", '>', x))  # greater than symbol
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(&#39;)", "'", x))  # single quote
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"(&quot;)", '"', x))  # double quote

    #deal with special characters
    #df['comment'] = df['comment'].apply(lambda x: re.sub(r"([\+\*])\1+",'',x))#only remove + and * when they are repeated for visual attention
    #df['comment'] = df['comment'].apply(lambda x: re.sub(r"(?<=!)!|(?<=\()\(|(?<=\))\)|(?<=-)-|(?<=\?)\?|(?<=\|)\|", '', x)) #remove repeating special characters

    #df['comment'] = df['comment'].str.lower() #make lowercase --> keep casing for now

    #remove mentions
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"@\x20[a-zA-Z0-9_]*|@[a-zA-Z0-9_]*",'',x))


    # remove extra enters and spaces
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"<br>", '\x20', x))
    df['comment'] = df['comment'].apply(lambda x: re.sub(r"\x20+", '\x20', re.sub(r"\u000a+|\u000d+|\u2028+|\u2029+", '\x20', x)))

    df = df[df['comment'].apply(lambda x: len(x.split()) >= 15)] # remove comments that are less than 15 words long
    

    df.to_csv(save_path, index=False)
    
    print((time.time() - time_start) / 60)