import pandas as pd
import numpy as np

def count_per_minute(original_df):
    df = original_df.copy()
    df['iso_time'] = df['time'].apply(lambda x: str(x)[:-3]+":00")
    df['iso_time'] = pd.to_datetime(df['iso_time'], dayfirst=False)
    df['iso_time'] = df['iso_time'].astype(np.int64)
    df['iso_time'] = df['iso_time'].apply(lambda x: int(x/60000000000))
    df = df.groupby(['iso_time']).count()
    
    df['time'] = df.index

    # drop everything else other than time and chat
    df = df[['time', 'chat']]
    df = df.reset_index(drop=True) 

    # rename chat to count
    df = df.rename(columns={'chat': 'count'})

    return df

def interesting_search(original_df, count_df):
    # find longest streak
    streak = 0
    streak_start = 0
    streak_end = 0
    for i in range(count_df.shape[0]-1):
        if count_df['time'][i+1] - count_df['time'][i] <= 3:
            streak += 1
        else:
            if streak > streak_end - streak_start:
                streak_start = i - streak
                streak_end = i
            streak = 0

    # print the streak
    print("Longest streak of messages sent in a row: ", streak_end - streak_start)
    print("Start time: ", count_df['time'][streak_start])
    print("End time: ", count_df['time'][streak_end])
    print("Total messages sent: ", count_df['count'][streak_start:streak_end].sum())
    print("Average messages sent per minute: ", count_df['count'][streak_start:streak_end].sum()/(streak_end - streak_start))

    # Average reply time of streak in seconds
    print("Average reply time of streak in seconds: ", (original_df['time'][streak_end] - original_df['time'][streak_start]).total_seconds()/(streak_end - streak_start))
    
    # find messages during longest streak
    original_df = original_df[original_df['iso_time'].isin(count_df['time'][streak_start:streak_end])]

    return original_df

def get_total_minutes(df):
    count_df = count_per_minute(df)
    total_mins = count_df.shape[0]
    return total_mins

