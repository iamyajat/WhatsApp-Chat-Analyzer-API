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
import time
from dateutil import tz
import requests
import scipy.stats as st
import base64
import io
from PIL import Image


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
    count = df["sender"].value_counts().to_dict()
    count_list = [{"member": x, "count": count[x]} for x in count]
    return count_list


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
    word_dict = dict(
        sorted(series.to_dict().items(), key=lambda item: item[1], reverse=True)
    )
    word_list = [{"member": x, "count": word_dict[x]} for x in word_dict]
    return word_list


def chats_month(df):
    df["month"] = pd.DatetimeIndex(df["time"]).month
    m_count = df["month"].value_counts().to_dict()
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    month_count = [{"month": x, "count": 0} for x in months]

    for mc in m_count:
        month_count[mc - 1]["count"] = m_count[mc]

    month_df = pd.DataFrame(month_count)
    month_df["month_codes"] = pd.Series(range(1, 13))
    month_corr = month_df["month_codes"].corr(month_df["count"])
    return month_count, month_corr


def get_gender(name):
    URL = "https://api.genderize.io"
    PARAMS = {"name": name}
    r = requests.get(url=URL, params=PARAMS)
    data = r.json()
    return data


def get_category(names):
    n = []
    for name in names:
        x = name.split()
        n.append(x[0].lower())
    data = get_gender(n)
    genders = []
    gb = {"boy": False, "girl": False}
    try:
        for d in data:
            genders.append(d["gender"])
            if d["gender"] == "male":
                gb["boy"] = True
            elif d["gender"] == "female":
                gb["girl"] = True
    except:
        print("Gender API calls over")

    return gb


def most_used_emoji(df):
    df = df.dropna()
    emoji_list = df["message"].apply(extract_emojis).tolist()
    emoji_str = "".join(emoji_list)
    res = Counter(emoji_str)
    top_10 = res.most_common(10)
    top_10_list = [{"emoji": x[0], "count": x[1]} for x in top_10]

    return top_10_list


def chats_hour(df):
    df["hour"] = pd.DatetimeIndex(df["time"]).hour
    h_count = df["hour"].value_counts().to_dict()
    hour_count = [{"hour": x, "count": 0} for x in range(24)]
    for hc in h_count:
        hour_count[hc]["count"] = h_count[hc]
    return hour_count


def get_time_diff(df):
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df["time_diff"] = df["time"].diff()
    return df


def longest_wait(df):
    df = get_time_diff(df)
    df1 = df[df["time_diff"] == df["time_diff"].max()]
    max_gap = df1["time_diff"].max()
    date1 = df1["time"].iloc[0]
    date2 = date1 - max_gap
    return {
        "gap": max_gap * 1000,
        "start_time": int(date2.timestamp() * 1000),
        "end_time": int(date1.timestamp() * 1000),
    }


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


def words_weight(df):
    df = df.dropna()
    chat_words = ""
    for val in df["message"]:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        chat_words += " ".join(tokens) + " "
    # remove links
    chat_words = re.sub(r"http\S+", "", chat_words)
    return chat_words


def word_cloud_words(df):
    chat_words = words_weight(df)
    words_dict = WordCloud(
        stopwords=stopwords,
    ).process_text(chat_words)
    words_dict = dict(
        sorted(words_dict.items(), key=lambda item: item[1], reverse=True)
    )
    if len(words_dict) > 100:
        words_dict = {k: words_dict[k] for k in list(words_dict)[:100]}
    max_val = max(words_dict.values())
    min_val = min(words_dict.values()) - 1
    diff_val = max_val - min_val
    return [
        {"word": k, "count": v, "weight": ((v - min_val) / diff_val)}
        for k, v in words_dict.items()
    ]


def word_cloud(df):
    chat_words = words_weight(df)
    mask_arr = np.array(Image.open("assets/masks/walogo.jpg"))

    wordcloud = WordCloud(
        font_path="assets/fonts/Poppins-Medium.ttf",
        mask=mask_arr,
        width=800,
        height=800,
        stopwords=stopwords,
        min_font_size=10,
        colormap="rainbow",
    ).generate(chat_words)

    return wordcloud


def get_word_cloud(chats):
    df = chats_to_df(chats)
    return word_cloud_to_base64(df)


def word_cloud_to_base64(df):
    img = word_cloud(df)
    img_bytes = io.BytesIO()
    img.to_image().save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def most_active_day(df):
    df["date"] = pd.DatetimeIndex(df["time"]).date
    d_count = df["date"].value_counts()
    max_day = d_count.loc[d_count == d_count.max()]
    max_day_dict = max_day.to_dict()
    max_day_list = [
        {
            "date": datetime.datetime(
                year=x.year, month=x.month, day=x.day, tzinfo=tz.tzutc()
            ).timestamp()
            * 1000,
            "amount": max_day_dict[x],
        }
        for x in max_day_dict
    ][0]
    return max_day_list


def zscore(amt):
    mean = 25000
    std = 12000
    z = (amt - mean) / std
    p = st.norm.cdf(z)
    return z, p


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


def wrap(chats):
    df = getYear2021(chats_to_df(chats))
    chat_members = members(df)
    num_arr = no_of_messages_per_member(df)
    words = word_count(df)
    months, month_corr = chats_month(df)
    # get max month
    max_month = months[0]
    for m in months:
        if m["count"] > max_month["count"]:
            max_month = m
    hours = chats_hour(df)
    max_hour = hours[0]
    for h in hours:
        if h["count"] > max_hour["count"]:
            max_hour = h
    top_10_emoji = most_used_emoji(df)
    cloud_words = word_cloud_words(df)
    z, p = zscore(len(df.index))

    return {
        "group": len(chat_members) > 2,
        "members": chat_members,
        "gender": get_category(chat_members),
        "total_no_of_chats": len(df.index),
        "top_percent": p,
        "z_score": z,
        "most_active_member": num_arr[0],
        "no_of_messages_per_member": num_arr,
        "word_count_per_member": words,
        "most_active_month": max_month,
        "month_correlation": month_corr,
        "monthly_chats_count": months,
        "most_active_hour": max_hour,
        "hourly_count": hours,
        "most_active_day": most_active_day(df),
        "longest_gap": longest_wait(df),
        "who_texts_first": who_texts_first(df),
        "most_used_emoji": top_10_emoji[0],
        "top_10_emojis": top_10_emoji,
        "most_used_word": cloud_words[0],
        "word_cloud_words": cloud_words,
        "word_cloud_base64": word_cloud_to_base64(df),
    }
