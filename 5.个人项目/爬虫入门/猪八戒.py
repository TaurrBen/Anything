import requests
from lxml import etree

url = "https://beijing.zbj.com/search/service/?kw=saas&r=2"
response = requests.get(url)
response.encoding = "utf-8"
# print(response.text)

et = etree.HTML(response.text)
divs =  et.xpath("//div[@class='search-result-list-service']/div")
for div in divs:
    print(div.xpath("./div/div[3]/div[1]/span/text()"))
    print(div.xpath("./div/a/div[2]/div[1]/div/text()"))