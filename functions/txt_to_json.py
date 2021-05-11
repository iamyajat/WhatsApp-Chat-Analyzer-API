import numpy as np
import pandas as pd
import re
import json


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


def convert(
    chats,
    dayfirst=True,
):
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

    wa_data["time"] = pd.to_datetime(wa_data["time"], dayfirst=dayfirst)

    df = pd.DataFrame(wa_data["time"])
    df["sender"] = wa_data["person"]
    df["message"] = wa_data["message"]

    d = df.to_json(orient="records")

    j = json.loads(d)
    print(len(j))

    return j
