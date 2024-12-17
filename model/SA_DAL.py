import json
import pandas as pd
import spacy
from textblob import TextBlob

def setup():
    data = []
    f = open('database/SA_database.json', 'w')
    json.dump(data, f)
    f.close()

def write_to_db(data):
    f = open('database/SA_database.json', 'w')
    json.dump(data, f)
    f.close

def get_db_as_dict():
    f = open('database/SA_database.json', 'r')
    data = json.load(f)
    f.close()
    return data

def SA_without_aspects():
    data = get_db_as_dict()

