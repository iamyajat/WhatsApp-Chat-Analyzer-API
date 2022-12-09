import datetime
import json
import pandas as pd
import numpy as np


def count_per_minute(df):
    df["iso_time"] = df["time"].apply(lambda x: str(x)[:-3] + ":00")
    df["iso_time"] = pd.to_datetime(df["iso_time"], dayfirst=False)
    df["iso_time"] = df["iso_time"].astype(np.int64)
    df["iso_time"] = df["iso_time"].apply(lambda x: int(x / 60000000000))
    df = df.copy()
    df = df.groupby(["iso_time"]).count()

    df["time"] = df.index

    # drop everything else other than time and chat
    df = df[["time", "message"]]
    df = df.reset_index(drop=True)

    # rename chat to count
    df = df.rename(columns={"message": "count"})

    return df


def interesting_search(original_df, count_df):
    # find longest streak
    streak = 0
    streak_start = 0
    streak_end = 0
    for i in range(count_df.shape[0] - 1):
        if count_df["time"][i + 1] - count_df["time"][i] <= 3:
            streak += 1
        else:
            if streak > streak_end - streak_start:
                streak_start = i - streak
                streak_end = i
            streak = 0

    # stats for streak
    longest_streak = streak_end - streak_start
    longest_streak_start = count_df["time"][streak_start] * 60000
    longest_streak_end = count_df["time"][streak_end] * 60000
    total_messages_sent = count_df["count"][streak_start:streak_end].sum()
    average_reply_time = (longest_streak_end - longest_streak_start) / (
        total_messages_sent * 1000
    )

    # convert longest streak to datetime
    longest_streak_start_dt = datetime.datetime.fromtimestamp(
        longest_streak_start / 1000
    ).strftime("%B %d, %Y")
    longest_streak_end_dt = datetime.datetime.fromtimestamp(
        longest_streak_end / 1000
    ).strftime("%B %d, %Y")

    # print the stats
    print("Longest streak:\t", longest_streak, "minutes")
    print("Total messages sent:\t", total_messages_sent, "messages")
    print("Longest streak date:\t", longest_streak_start_dt)
    # print("Longest streak end: ", longest_streak_end_dt)
    print("Average reply time:\t", round(average_reply_time, 2), "seconds")

    # find messages during longest streak
    # original_df = original_df[
    #     original_df["iso_time"].isin(count_df["time"][streak_start:streak_end])
    # ]

    # odf_json_str = original_df[["time", "sender", "message"]].to_json(orient="records")
    # odf_json = json.loads(odf_json_str)
    # odf_json[-1]["message"] = odf_json[-1]["message"][:-1]

    # return as dictionary
    return {
        "streak_duration": longest_streak,
        "streak_start": int(longest_streak_start),
        "streak_end": int(longest_streak_end),
        "total_messages_sent": int(total_messages_sent),
        "average_reply_time": float(average_reply_time),
        # "messages_during_streak": odf_json,
    }


def get_total_minutes(df):
    count_df = count_per_minute(df)
    total_mins = count_df.shape[0]
    return total_mins, count_df
