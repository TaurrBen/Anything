import requests
import re

f = open("top250.csv",mode="w",encoding="utf-8")

url = "https://movie.douban.com/top250"
headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
response = requests.get(url,headers=headers)

pageSource = response.text #text字符串 content字节形式（如读图片）
# print(pageSource)

# re.S 可以让正则中的.匹配换行符
obj = re.compile(r'<div class="item">.*?<span class="title">(?P<name>.*?)</span>.*?导演: '
                 r'(?P<director>.*?)&nbsp.*?<br>'
                 r'(?P<year>.*?)&nbsp.*?<span class="rating_num" property="v:average">'
                 r'(?P<rating_num>.*?)</span>.*?<span>'
                 r'(?P<num>.*?)人评价</span>.*?<span class="inq">'
                 r'(?P<quote>.*?)</span>',re.S)
result = obj.finditer(pageSource)
for item in result:
    name = item.group("name")
    director = item.group("director")
    year = item.group("year").strip() #去掉字符串前后的换行符
    rating_num = item.group("rating_num")
    num = item.group("num")
    quote = item.group("quote")
    print(name,director,year,rating_num,num,quote)
    f.write(f"{name},{director},{year},{rating_num},{num},{quote}\n")
f.close()

#通过会话方式  登录 并自动传递Cookies
session = requests.session()
#防盗链  如梨视频下载 会溯源
