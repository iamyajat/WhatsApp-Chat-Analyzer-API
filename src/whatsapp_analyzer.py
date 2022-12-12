from http.client import HTTPException
import numpy as np
import pandas as pd
import re
import json
import datetime
import random
from wordcloud import WordCloud
import emoji
from collections import Counter
from datetime import timedelta
import time
from dateutil import tz
import pickle
import requests
import scipy.stats as st
import base64
import io
from zipfile import ZipFile
from src.interesting_search import get_total_minutes, interesting_search
from src.utils import check_dayfirst, parse_datetime

with open("./assets/stopwords/stop_words.pkl", "rb") as f:
    stopwords = pickle.load(f)


def extract_zip(input_zip):
    input_zip = ZipFile(io.BytesIO(input_zip))
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def time_extractor(x, phone):
    y = 0
    if phone == "IOS":
        y = x.find("] ")
        return x[1:y]
    else:
        y = x.find(" - ")
        return x[:y]


def chat_extractor(x, phone):
    y = 0
    if phone == "IOS":
        y = x.find("] ") + 2
    else:
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
        or s.find("image omitted") != -1
        or s.find("video omitted") != -1
        or s.find("audio omitted") != -1
        or s.find("file omitted") != -1
        or s.find("sticker omitted") != -1
        or s.find("gif omitted") != -1
        or s.find("voice omitted") != -1
        or s.find("contact omitted") != -1
        or s.find("location omitted") != -1
        or s.find("document omitted") != -1
        or s.find("<MEDIA>") != -1
    ):
        return np.nan
    else:
        return s


def parse_message(s, phone):
    time = time_extractor(s, phone)
    person_chat = chat_extractor(s, phone)
    person = person_extractor(person_chat)
    message = message_extractor(person_chat)
    return [time.lower(), person, message]


def chats_to_df(chats):
    REGEX = {
        "IOS": "^[{1}[0-9]+[\/|\–|\-|\.][0-9]+[\/|\–|\-|\.][0-9]+,?\s[0-9]+[:|.][0-9]+[:|.][0-9]+.*$",
        "ANDROID": "^[0-9]+/[0-9]+/[0-9]+,?\s[0-9]+[:|.][0-9]+\s.*$",
    }
    new_chats = []
    phone = "ANDROID"
    if chats[0].find(" - ") == -1:
        phone = "IOS"
    c = 0
    i = 0
    while i < len(chats):
        chats[i] = chats[i].replace("\u200e", "").replace("\r", "")
        new_chats.append(chats[i])
        i += 1
        while i < len(chats) and not bool(re.search(REGEX[phone], chats[i])):
            new_chats[c] += "\n" + chats[i]
            i += 1
        c += 1

    wa_data = pd.DataFrame(new_chats, columns=["chats"])
    wa_data = wa_data["chats"].apply(parse_message, args=(phone,))

    wa_data = pd.DataFrame(wa_data.tolist(), columns=["time", "sender", "message"])

    wa_data.columns = ["time", "sender", "message"]

    dayfirst = check_dayfirst(list(wa_data["time"]))
    wa_data["time"] = wa_data["time"].apply(parse_datetime, args=(dayfirst,))

    return wa_data


def members(df):
    chat_members = df["sender"].unique()
    chat_members = [x for x in chat_members if str(x) != "nan"]
    return chat_members


def getYear2022(df):
    df = df[df["time"].dt.year == 2022]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_emojis(s):
    return "".join(c for c in s if c in emoji.EMOJI_DATA)


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


def chats_date(df):
    df["date"] = pd.DatetimeIndex(df["time"]).date


def check_chat_date(df, date):
    return date in df["date"].unique()


def convert_long_to_date(long_date):
    dt = datetime.datetime.fromtimestamp(long_date / 1000)
    date = datetime.date(dt.year, dt.month, dt.day)
    return date


def get_chat_date_string(df, longest_break_start, longest_break_end):
    chats_date(df)
    result_chat_date = ""
    first_day = False
    # loop through all of the days in the year and check if there is a chat on that day
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                d = datetime.date(2022, month, day)
            except ValueError:
                continue
            if (not first_day) and check_chat_date(df, d):
                first_day = True

            if first_day:
                start_gap = convert_long_to_date(longest_break_start)
                end_gap = convert_long_to_date(longest_break_end)
                if d > start_gap and d < end_gap:
                    result_chat_date += "2"
                    continue
                if check_chat_date(df, d):
                    result_chat_date += "0"
                else:
                    result_chat_date += "1"

            else:
                result_chat_date += "9"

    return result_chat_date


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
    emoji_list = df["message"].apply(extract_emojis).tolist()
    emoji_str = "".join(emoji_list)
    emoji_str = (
        emoji_str.replace("\U0001f3fb", "")
        .replace("\U0001f3fc", "")
        .replace("\U0001f3fd", "")
        .replace("\U0001f3fe", "")
        .replace("\U0001f3ff", "")
    )
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
    df["time_diff"] = df["time"].diff()
    return df


def longest_wait(df):
    try:
        df = get_time_diff(df)
        df1 = df[df["time_diff"] == df["time_diff"].max()]
        max_gap = df1["time_diff"].max()
        date1 = df1["time"].iloc[0]
        date2 = date1 - max_gap
        # convert max_gap to int 64
        max_gap = int(max_gap.total_seconds())
        return {
            "gap": int(max_gap) * 1000,
            "start_time": int(date2.timestamp() * 1000),
            "end_time": int(date1.timestamp() * 1000),
        }
    except:
        return {
            "gap": 0,
            "start_time": 0,
            "end_time": 0,
        }


def who_texts_first(df):
    df = get_time_diff(df)
    df1 = df[df["time_diff"] > timedelta(minutes=60)]
    send_counts = df1["sender"].value_counts().to_dict()
    if len(send_counts) == 0:
        return "No one"
    max_send_counts = max(send_counts, key=send_counts.get)
    return max_send_counts


def throwback_chats(chats, n):
    df = chats_to_df(chats)
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
    chat_words = ""
    for val in df["message"]:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        chat_words += " ".join(tokens) + " "
    chat_words = re.sub(r"http\S+", "", chat_words)
    if chat_words.strip() == "":
        return "chat unavailable"
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
    # mask_arr = np.array(Image.open("assets/masks/walogo.jpg"))
    wordcloud = WordCloud(
        font_path="assets/fonts/Poppins-Medium.ttf",
        # mask=mask_arr,
        min_word_length=2,
        width=360,
        height=480,
        stopwords=stopwords,
        min_font_size=12,
        colormap="gist_ncar",
    )
    wc = None
    try:
        wc = wordcloud.generate(chat_words)
    except:
        wc = wordcloud.generate("chat unavailable")

    return wc


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
    mean = 22000
    std = 12000
    z = (amt - mean) / std
    p = st.norm.cdf(z)
    return z, max(min(p, 0.999999), 0.0001)


# get median of time difference
def get_median_time_diff(df):
    time_df_list = list(df["time_diff"])[1:]
    time_df_list = [x.total_seconds() for x in time_df_list]
    time_df_list.sort()
    if len(time_df_list) == 0:
        return 0
    return np.median(time_df_list)


# get every 10%, 20%, 30%.... 90% of the time difference
def get_time_diff_percentile(df):
    time_df_list = list(df["time_diff"])[1:]
    time_df_list = [x.total_seconds() for x in time_df_list]
    time_df_list.sort()
    if len(time_df_list) == 0:
        return 0
    percentiles = []
    for i in range(1, 51):
        percentiles.append(np.percentile(time_df_list, i * 2))
    return percentiles


# get the reponsiveness of the chat
def get_responsiveness(df, percentiles):
    # get the first greater than zero percentile
    for i in range(len(percentiles)):
        if percentiles[i] > 0:
            print("Chat responsiveness:\t", (i / 50.0))
            return i
    return 0


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
    df = getYear2022(chats_to_df(chats))
    if df.shape[0] < 75:
        print("\nNot enough members or chats to analyze from 2022!\n\n")
        return None
    print("\n\n---------------------------------------------")
    print("Members")
    print("---------------------------------------------")
    total_chats = len(df["message"])
    chat_members = members(df)
    num_members = len(chat_members)
    if num_members < 2:
        return None
    num_arr = no_of_messages_per_member(df)
    # words = word_count(df)
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

    active_day = most_active_day(df)
    top_10_emoji = most_used_emoji(df)
    # cloud_words = word_cloud_words(df)
    z, p = zscore(len(df.index))

    top_percent = 1 - p

    if chat_members:
        # print chat members
        print(", ".join(chat_members))
    else:
        "No members found"

    longest_gap = longest_wait(df)

    talk_string = get_chat_date_string(
        df, longest_gap["start_time"], longest_gap["end_time"]
    )

    total_mins, count_df = get_total_minutes(df)

    print("\n\n\n---------------------------------------------")
    print(" Chat Statistics")
    print("---------------------------------------------")

    print("Total chats:\t\t " + str(total_chats))
    print("Total members:\t " + str(num_members))
    print("Total minutes:\t " + str(total_mins))

    top_percent_100 = round(top_percent * 100, 2)
    print("Top percentile:\t ", top_percent_100, "%", sep="")

    print("Most active month:\t " + max_month["month"])
    print("Month correlation:\t", round(month_corr, 4))

    # convert to 12 hour time
    m_hour = max_hour["hour"] % 12
    if m_hour == 0:
        m_hour = 12
    ampm = "AM"
    if max_hour["hour"] >= 12:
        ampm = "PM"
    print(
        "Most active hour:\t ",
        str(m_hour),
        " ",
        ampm,
        " (",
        max_hour["hour"],
        ")",
        sep="",
    )

    print(
        "Most active day:\t "
        + datetime.datetime.fromtimestamp(active_day["date"] / 1000).strftime(
            "%B %d, %Y"
        )
    )

    # get median of time difference
    # median_time_diff = get_median_time_diff(df)

    # get every 10%, 20%, 30%.... 90% of the time difference
    time_diff_percentile = get_time_diff_percentile(df)

    # get the reponsiveness of the chat
    responsiveness = get_responsiveness(df, time_diff_percentile)

    longest_gap_in_days = int(longest_gap["gap"] / (24 * 60 * 60 * 1000))
    longest_session = interesting_search(df, count_df)

    print("Longest gap:\t\t", longest_gap_in_days, "days")
    print(
        "Longest gap start:\t",
        datetime.datetime.fromtimestamp(longest_gap["start_time"] / 1000).strftime(
            "%B %d, %Y"
        ),
    )
    print(
        "Longest gap end:\t",
        datetime.datetime.fromtimestamp(longest_gap["end_time"] / 1000).strftime(
            "%B %d, %Y"
        ),
    )

    return {
        "group": len(chat_members) > 2,
        "members": chat_members,
        # "gender": get_category(chat_members),
        "total_no_of_chats": total_chats,
        "total_no_of_minutes": total_mins,
        "top_percent": top_percent,
        # "z_score": z,
        "most_active_member": num_arr[0] if len(num_arr) != 0 else "No one",
        "no_of_messages_per_member": num_arr,
        # "word_count_per_member": words,
        # "median_reply_time": (median_time_diff / 60.0),
        # "reply_time_percentile": [x / 60.0 for x in time_diff_percentile],
        "chat_responsiveness": responsiveness / 50.0,
        "most_active_month": max_month,
        "month_correlation": month_corr,
        "monthly_chats_count": months,
        "most_active_hour": max_hour,
        "hourly_count": hours,
        "most_active_day": active_day,
        "longest_session": longest_session,
        "longest_gap": longest_gap,
        "no_talk_string": talk_string,
        "who_texts_first": who_texts_first(df),
        # "most_used_emoji": top_10_emoji[0],
        "top_10_emojis": top_10_emoji,
        # "most_used_word": cloud_words[0],
        # "word_cloud_words": cloud_words,
        "word_cloud_base64": word_cloud_to_base64(df),
    }
