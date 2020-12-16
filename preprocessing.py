import json
from os.path import join
from random import shuffle

from bs4 import BeautifulSoup
import pandas as pd


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

    return topic_list


def load_data(dir):
    body = []
    structure = ["d0", "d1", "d2", "d3"]
    categories = {"d0": [], "d1": [], "d2": [], "d3": []}

    soup = BeautifulSoup(open(join(dir), "rt").read(), "html.parser")

    for book in soup.findAll("book"):
        book_soup = BeautifulSoup(str(book), "html.parser")
        for t in book_soup.findAll("topics"):
            s1 = BeautifulSoup(str(t), "html.parser")
            for level in structure:
                all_t = [str(t1.string) for t1 in s1.findAll(level)]
                categories[level].append(all_t)
        body.append(str(book_soup.find("body").string))

    data = categories
    data.update({"body": body})

    return pd.DataFrame(data)


topic_list = create_topic_list()

data_dir = "data"
train = load_data(join(data_dir, "BlurbGenreCollection_EN_train.txt"))
dev = load_data(join(data_dir, "BlurbGenreCollection_EN_dev.txt"))
test = load_data(join(data_dir, "BlurbGenreCollection_EN_tes.txt"))
