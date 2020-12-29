# BERT Fine-Tuned for HMLTC
BERT fine-tuned for hierarchical multi-label text classification (HMLTC) using the [BlurbGenreCollection-EN dataset](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html).

This repo is part of a project in the course TDDE16 – Text Mining at Linköping University.

## Installation
To create an environment where the code can be run, run the following:
```
git clone https://github.com/lukasborggren/bert-for-hmltc
cd bert-for-hmltc
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```