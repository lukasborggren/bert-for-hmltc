import json
from os.path import join

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def find_children(pairs, lvl, i):
    for pair in pairs:
        for top in lvl[i]:
            if pair[0] == top:
                lvl[i + 1].append(pair[1].strip())
    return lvl


def create_topic_list():
    lvl0 = [
        "Children’s Books",
        "Classics",
        "Fiction",
        "Humor",
        "Nonfiction",
        "Poetry",
        "Teen & Young Adult",
    ]

    lvl = [lvl0, [], [], []]

    with open("data/hierarchy.txt", "r") as f:
        pairs = [line.split("\t") for line in f.readlines()]

    for i in range(3):
        lvl = find_children(pairs, lvl, i)

    topic_list = [topic for lv in lvl for topic in lv]

    with open("data/topic_list.json", "w") as f:
        json.dump(topic_list, f)

    return load_topic_list()


def load_topic_list(DATA_PATH):
    with open(join(DATA_PATH, "topic_list.json"), "r") as f:
        topic_list = json.load(f)
    return topic_list


def to_ohe(indexes, num_topics):
    ohe = [0] * num_topics
    for i in indexes:
        ohe[i] = 1
    return ohe


def transform_data(blurbs, topics, num_topics):
    rows_list = []
    for i in tqdm(range(len(blurbs)), desc="Book", colour="green"):
        for j in range(3, -1, -1):
            if topics[j][i] or (j != 0 and topics[j - 1][i]):
                parent_cats = [cat for k in range(j) for cat in topics[k][i]]
                row = {
                    "blurb": blurbs[i],
                    "topics": to_ohe(topics[j][i], num_topics),
                    "parent_topics": to_ohe(parent_cats, num_topics),
                }
                rows_list.append(row)

    return rows_list


def load_data(dir, topic_list, use_parents):
    num_topics = len(topic_list)
    blurbs = []
    structure = ["d0", "d1", "d2", "d3"]

    if use_parents:
        topics = [[], [], [], []]
        # topics = {"d0": [], "d1": [], "d2": [], "d3": []}
    else:
        topics = []

    soup = BeautifulSoup(open(join(dir), "rt").read(), "html.parser")

    for i, book in enumerate(soup.findAll("book")):
        book_soup = BeautifulSoup(str(book), "html.parser")
        for t in book_soup.findAll("topics"):
            s1 = BeautifulSoup(str(t), "html.parser")
            for level in range(len(structure)):
                all_t = [
                    topic_list.index(t1.string) for t1 in s1.findAll(structure[level])
                ]
                if use_parents:
                    topics[level].append(all_t)
                else:
                    if level == 0:
                        topics.append(all_t)
                    else:
                        topics[i].extend(all_t)

        blurbs.append(str(book_soup.find("body").string))

    if use_parents:
        data = transform_data(blurbs, topics, num_topics)
    else:
        ohe_topics = []
        for t in topics:
            ohe_topics.append(to_ohe(t, num_topics))
        data = {"blurb": blurbs, "topics": ohe_topics}

    return pd.DataFrame(data)


# topic_list = create_topic_list()
DATA_PATH = "data"
topic_list = load_topic_list(DATA_PATH)
use_parents = True

print("Creating…")
train = load_data(
    join(DATA_PATH, "BlurbGenreCollection_EN_train.txt"), topic_list, use_parents
)
dev = load_data(
    join(DATA_PATH, "BlurbGenreCollection_EN_dev.txt"), topic_list, use_parents
)
test = load_data(
    join(DATA_PATH, "BlurbGenreCollection_EN_test.txt"), topic_list, use_parents
)

# train = load_data(join(DATA_PATH, "dummy_train.txt"), topic_list, use_parents)
# test = load_data(join(DATA_PATH, "dummy_train.txt"), topic_list, use_parents)

print("Saving…")
# Protocol 4 for Google Colab, not done for ext-files
train.to_pickle(join(DATA_PATH, "dataframes/train_ext.pkl"), protocol=4)
dev.to_pickle(join(DATA_PATH, "dataframes/dev_ext.pkl"), protocol=4)
test.to_pickle(join(DATA_PATH, "dataframes/test_ext.pkl"), protocol=4)

"""
Both split 64%/16%/20% into train/dev/test
Raw total: 91,892
Extended total: 306,231
"""
