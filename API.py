"""
import requests
import json
import pandas as pd

data = requests.get("https://fruityvice.com/api/fruit/all")

results = json.loads(data.text)
pd.DataFrame(results)

df2 = pd.json_normalize(results)

print(df2)

cherry = df2.loc[df2["name"] == 'Cherry']
(cherry.iloc[0]['family']) , (cherry.iloc[0]['genus'])
"""

import requests
import pandas as pd
import json

data = requests.get("https://official-joke-api.appspot.com/jokes/ten")

results = json.loads(data.text)

#df = pd.DataFrame(data)

df = pd.json_normalize(results)

print(df)
