import numpy as np
import pandas as pd
import re
import json
import datetime
import random
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
from datetime import timedelta

stopwords = set(STOPWORDS)


def time_extractor(x):
    y = x.find(" - ")
    return x[:y]


def chat_extractor(x):
    y = x.find(" - ") + 3
    return x[y:]


def person_extractor(x):
    y = x.find(": ")
    if y != -1:
        return x[:y]
    else:
        return np.nan


def message_extractor(x):
    y = x.find(": ") + 2
    s = ""
    if (y - 2) != -1:
        s = x[y:]
    else:
        s = x
    if (
        s == "<Media omitted>"
        or s == "This message was deleted"
        or s == "You deleted this message"
    ):
        return np.nan
    else:
        return s


def check_dates(dates):
    for date in dates:
        date = date[: date.find(", ")]
        day, month, year = date.split("/")
        try:
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            return False
    return True


def chats_to_df(chats):
    new_chats = []
    c = 0
    i = 0
    while i < len(chats):
        new_chats.append(chats[i])
        i += 1
        while i < len(chats) and not bool(
            re.search("^[0-9]+/[0-9]+/[0-9]+,\s[0-9]+:[0-9]+\s.*$", chats[i])
        ):
            new_chats[c] += "\n" + chats[i]
            i += 1
        c += 1

    wa_data = pd.DataFrame(new_chats, columns=["chats"])
    wa_data["time"] = wa_data["chats"].apply(time_extractor)
    wa_data["person_chat"] = wa_data["chats"].apply(chat_extractor)
    wa_data["person"] = wa_data["person_chat"].apply(person_extractor)
    wa_data["message"] = wa_data["person_chat"].apply(message_extractor)

    dayfirst = check_dates(list(wa_data["time"]))
    wa_data["time"] = pd.to_datetime(wa_data["time"], dayfirst=dayfirst)

    df = pd.DataFrame(wa_data["time"])
    df["sender"] = wa_data["person"]
    df["message"] = wa_data["message"]

    return df


def members(df):
    chat_members = df["sender"].unique()
    chat_members = [x for x in chat_members if str(x) != "nan"]
    return chat_members


def getYear2021(df):
    df = df[df["time"].dt.year == 2021]
    df.reset_index(drop=True, inplace=True)
    return df


def extract_emojis(s):
    return "".join(c for c in s if c in emoji.UNICODE_EMOJI["en"])


def chats_to_json(chats):
    df = chats_to_df(chats)
    df_json_str = df.to_json(orient="records")
    df_json = json.loads(df_json_str)
    df_json[-1]["message"] = df_json[-1]["message"][:-1]
    return {"no_of_messages": len(df_json), "chats": df_json}


def no_of_messages_per_member(df):
    count = df["sender"].value_counts()
    return count.to_dict()


def word_count(df):
    df = df.copy()
    df["no_of_words"] = df["message"].apply(lambda x: len(str(x).split()))
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    members = df["sender"].unique()
    word_count = {member: 0 for member in members}
    for member in members:
        sub_df = df[df["sender"] == member]
        word_count[member] = sum(sub_df["no_of_words"])
    series = pd.Series(word_count)
    series = series.rename("Word Count")
    return dict(
        sorted(series.to_dict().items(), key=lambda item: item[1], reverse=True)
    )


def chats_month(df):
    df["month"] = pd.DatetimeIndex(df["time"]).month
    m_count = df["month"].value_counts().to_dict()
    month_count = {
        "Jan": 0,
        "Feb": 0,
        "Mar": 0,
        "Apr": 0,
        "May": 0,
        "Jun": 0,
        "Jul": 0,
        "Aug": 0,
        "Sep": 0,
        "Oct": 0,
        "Nov": 0,
        "Dec": 0,
    }
    for mc in m_count:
        if mc == 1:
            month_count["Jan"] = m_count[mc]
        if mc == 2:
            month_count["Feb"] = m_count[mc]
        if mc == 3:
            month_count["Mar"] = m_count[mc]
        if mc == 4:
            month_count["Apr"] = m_count[mc]
        if mc == 5:
            month_count["May"] = m_count[mc]
        if mc == 6:
            month_count["Jun"] = m_count[mc]
        if mc == 7:
            month_count["Jul"] = m_count[mc]
        if mc == 8:
            month_count["Aug"] = m_count[mc]
        if mc == 9:
            month_count["Sep"] = m_count[mc]
        if mc == 10:
            month_count["Oct"] = m_count[mc]
        if mc == 11:
            month_count["Nov"] = m_count[mc]
        if mc == 12:
            month_count["Dec"] = m_count[mc]

    return month_count


def most_used_emoji(df):
    df = df.dropna()
    emoji_list = df["message"].apply(extract_emojis).tolist()
    emoji_str = "".join(emoji_list)
    res = Counter(emoji_str)
    top_10 = res.most_common(10)
    top_10_dict = {x[0]: x[1] for x in top_10}
    return top_10_dict


def chats_hour(df):
    df["hour"] = pd.DatetimeIndex(df["time"]).hour
    h_count = df["hour"].value_counts().to_dict()
    return h_count


def get_time_diff(df):
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df["time_diff"] = df["time"].diff()
    return df


def longest_wait(df):
    df = get_time_diff(df)
    max_gap = df["time_diff"].max()
    return max_gap * 1000


def who_texts_first(df):
    df = get_time_diff(df)
    df = df[df["time_diff"] > timedelta(minutes=30)]
    send_counts = df["sender"].value_counts().to_dict()
    max_send_counts = max(send_counts, key=send_counts.get)
    return max_send_counts


def throwback_chats(chats, n):
    df = chats_to_df(chats)
    df = df.dropna()
    df = df.drop("time", axis=1)
    x = df["sender"].size
    if x > n:
        r = random.randint(0, x - n - 1)
        df = df.iloc[r : r + n]
    df_json_str = df.to_json(orient="records")
    df_json = json.loads(df_json_str)
    df_json[-1]["message"] = df_json[-1]["message"][:-1]
    return {"throwback_chats": df_json}


def analyze(chats):
    df = chats_to_df(chats)
    chat_members = members(df)
    num_arr = no_of_messages_per_member(df)
    words = word_count(df)
    month = chats_month(df)

    return {
        "members": chat_members,
        "no_of_messages": len(df["message"]),
        "no_of_messages_per_member": num_arr,
        "word_count_per_member": words,
        "month_chats_count": month,
    }


def words_weight(df):
    df = df.dropna()
    chat_words = ""
    for val in df["message"]:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        chat_words += " ".join(tokens) + " "
    return chat_words


def word_cloud_words(df):
    chat_words = words_weight(df)
    words_dict = WordCloud().process_text(chat_words)
    words_dict = dict(sorted(words_dict.items(), key=lambda item: item[1], reverse=True))
    return [{"word": k, "freq": v} for k, v in words_dict.items()][:100]


def word_cloud(chats):
    df = chats_to_df(chats)
    chat_words = words_weight(df)
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stopwords,
        min_font_size=10,
    ).generate(chat_words)

    return np.array(wordcloud)


def wrap(chats):
    # WhatsApp Wrap 2021 features:
    # 1. Number of messages
    # 2. Number of messages per member
    # 3. Word count per member
    # 4. Most active month
    # 5. Monthly chats count
    # 6. Most active hour
    # 7. Hourly chats count
    # 8. Most used emoji
    # 9. Longest wait
    # 10. Who texts first
    # 11. Most used words (word cloud)

    df = getYear2021(chats_to_df(chats))
    chat_members = members(df)
    num_arr = no_of_messages_per_member(df)
    words = word_count(df)
    month = chats_month(df)
    hour = chats_hour(df)
    top_10_emoji = most_used_emoji(df)

    return {
        "members": chat_members,
        "total_no_of_chats": len(df.index),
        "no_of_messages_per_member": num_arr,
        "word_count_per_member": words,
        "most_active_month": max(month, key=month.get),
        "monthly_chats_count": month,
        "most_active_hour": max(hour, key=hour.get),
        "hourly_count": hour,
        "most_used_emoji": max(top_10_emoji, key=top_10_emoji.get),
        "top_10_emojis": top_10_emoji,
        "longest_wait": longest_wait(df),
        "who_texts_first": who_texts_first(df),
        "word_cloud": word_cloud_words(df),
    }
