import numpy as np
import pandas as pd
import re
import json
import datetime
from wordcloud import WordCloud, STOPWORDS

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
    return dict(sorted(series.to_dict().items(), key=lambda item: item[1], reverse=True))


def random_chats(chats, n):
    df = chats_to_df(chats).sample(n)
    df = df.dropna()
    df = df.drop("time", axis=1)
    df_json_str = df.to_json(orient="records")
    df_json = json.loads(df_json_str)
    df_json[-1]["message"] = df_json[-1]["message"][:-1]
    return {"random_chats": df_json}


def analyze(chats):
    df = chats_to_df(chats)
    chat_members = members(df)
    num_arr = no_of_messages_per_member(df)
    words = word_count(df)

    return {
        "members": chat_members,
        "no_of_messages": len(df["message"]),
        "no_of_messages_per_member": num_arr,
        "word_count_per_member": words,
    }


def word_cloud(chats):
    df = chats_to_df(chats)
    df = df.dropna()
    chat_words = ""
    for val in df["message"]:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        chat_words += " ".join(tokens) + " "

    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stopwords,
        min_font_size=10,
    ).generate(chat_words)

    return np.array(wordcloud)
