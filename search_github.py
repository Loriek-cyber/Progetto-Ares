import urllib.request
import json
import time

query = "filename:fast_lane.ai+OR+\"fast_lane.ai\"+language:python"
url = f"https://api.github.com/search/code?q={query}"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        for item in data.get('items', [])[:5]:
            print(item['html_url'])
except Exception as e:
    print(e)
