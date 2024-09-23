
import requests
from lxml import etree
import asyncio
import aiofiles
import aiohttp


def get_every_chapter_url(url):
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    resp = requests.get(url,headers=headers)
    resp.encoding ="gbk"
    tree = etree.HTML(resp.text)
    href_list = tree.xpath("//ul[@class='section-list fix']/li/a/@href")
    print(href_list)
    return href_list

async def download_one(url):
    print(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.encoding = 'gbk'
            page_source = await resp.text()
            # page_source = await resp.text()
            print(page_source)
            tree = etree.HTML(page_source)
            title = tree.xpath("//h1[@class='title']/text()")
            content = tree.xpath("//div[@class='content']/text()")

            async with aiofiles.open(f"./明朝那些事儿/{title}.txt",mode="w",encoding='utf-8') as f:
                await f.write(content)


async def download(href_list):
    tasks = []
    x = 1
    for href in href_list:
        t = asyncio.create_task(download_one(href))
        tasks.append(t)
    await asyncio.wait(tasks)

def main():
    url = "https://www.zanghaihua.org/mingchaonaxieshier/"
    href_list = get_every_chapter_url(url)
    asyncio.run(download(href_list))

if __name__ == '__main__':
    main()
