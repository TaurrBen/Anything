import requests
url = "https://www.baidu.com"
proxy = {
    "http":"http://121.8.215.106:9797",
    "https":"https://121.8.215.106:9797"
}

response = requests.get(url,proxies=proxy)
print(response.text)