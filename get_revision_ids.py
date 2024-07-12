import requests
import datetime
import sys
import tqdm
import pandas as pd

from collections import defaultdict
import time

def get_revision_id(topic, date):
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"

    REQUEST_PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": topic,
        "rvstart": date,
        "rvlimit": "1",
        "rvdir": "older",
        "rvprop": "ids",
        "format": "json",
    }

    R = S.get(url=URL, params=REQUEST_PARAMS)
    data = R.json()

    temp = list(data['query']['pages'].values())[0]

    if "revisions" not in temp:
        return -1
    
    revision = temp['revisions'][0]

    if "revid" not in revision:
        return -1

    return revision['revid']

def step_date(year, month):
    month += 1
    if month == 13:
        month = 1
        year += 1

    return year, month

def main():
    start_year = 2016
    start_month = 1 # January
    
    end_year = 2016
    end_month = 2 # December

    assert len(sys.argv) > 1, "Provide a file consisting of topics to collect revision ids of"
    
    most_popular_topics = list(pd.read_csv(sys.argv[1])["title"])

    revids = defaultdict(lambda: [])
    revids["title"] = most_popular_topics

    year = start_year
    month = start_month

    while True:
        if year == end_year and month == end_month:
            break

        date = datetime.datetime(year, month, 2)

        # In case of resuming 
        if date in revids and len(revids[date]) == 5000:
            year, month = step_date(year, month)
            continue

        for i, topic in tqdm.tqdm(enumerate(most_popular_topics), total = 5000):
            if i < len(revids[date]):
                continue

            while True:
                try:
                    revid = get_revision_id(topic, date)
                    break
                except TimeoutError:
                    time.sleep(3)
            
            revids[date].append(revid)

        year, month = step_date(year, month)

        df = pd.DataFrame(revids)
        df.to_csv("./most_popular_revids.csv")
        
if __name__ == "__main__":
    main()