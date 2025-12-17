import requests
import pandas as pd

bearer = "AAAAAAAAAAAAAAAAAAAAAIyq5QEAAAAAksNdai%2BeEmiWMDUE4ocgCTm57B4%3DGnuyjwhk43q1lkXkOsAVmYnrcUW3l08Ylrg9tVHOuvCeYCt67b"
headers = {"Authorization": f"Bearer {bearer}"}
query = '("Carlos Manzo" OR @CarlosManzo) lang:es -is:retweet'
url = f"https://api.x.com/2/tweets/search/recent?query={query}&max_results=100&tweet.fields=created_at,author_id"

r = requests.get(url, headers=headers)
data = r.json()

tweets = [[t["id"], t["created_at"], t["text"]] for t in data["data"]]
df = pd.DataFrame(tweets, columns=["id", "fecha", "texto"])
df.to_csv("corpus_carlos_manzo.csv", index=False, encoding="utf-8")
