import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import os
import concurrent.futures
from threading import Lock, Event
import time

# 目标URL
base_url = "http://137.175.55.201:8081/"

#其他地址
# https://videos.959102.com:51443/
# http://137.175.55.201:8081/
# http://142.4.97.59:9080
# http://142.4.97.61:9080
# http://142.0.136.92:9080



# 最大爬取深度,建议2
MAX_DEPTH = 2

# 终止爬取的URL
STOP_URL = "http://137.175.55.201:8081/20250131/"

# 用于存储已访问的URL，避免重复访问
visited_urls = set()
visited_urls_lock = Lock()

# 用于控制爬取是否应该停止
should_stop = Event()

# 文件锁，确保多线程环境下文件写入是线程安全的
file_lock = Lock()

# 输出文件路径
output_file = "m3u8_links.txt"

# 发送HTTP请求并获取页面内容
def fetch_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# 解析页面并提取所有.m3u8链接和子页面链接
def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    m3u8_links_found = []
    subpage_links = []

    # 查找所有 <a> 标签中的 href 属性
    for link in soup.find_all('a', href=True):
        href = link['href']
        abs_url = urljoin(base_url, href)
        abs_url, _ = urldefrag(abs_url)  # 去除URL中的锚点部分

        if href.endswith('.m3u8'):
            m3u8_links_found.append(abs_url)
        elif abs_url.startswith(base_url):  # 只处理同域的链接
            subpage_links.append(abs_url)

    return m3u8_links_found, subpage_links

# 立即将找到的.m3u8链接保存到文件中
def save_m3u8_link(link):
    with file_lock:
        with open(output_file, "a") as f:
            f.write(link + "\n")
        print(f"Saved .m3u8 link: {link}")

# 递归爬取页面
def crawl(url, depth=0):
    # 如果已经设置了终止标志，直接返回
    if should_stop.is_set():
        print(f"Crawling stopped at {url}.")
        return

    with visited_urls_lock:
        if url in visited_urls:
            print(f"Already visited {url}, skipping...")
            return
        visited_urls.add(url)

    print(f"Crawling {url} at depth {depth}...")

    html_content = fetch_page(url)
    if not html_content:
        return

    # 如果当前URL是终止URL，设置终止标志并返回
    if url == STOP_URL:
        print(f"Reached stop URL: {url}. Stopping further crawling.")
        should_stop.set()
        return

    # 提取.m3u8链接和子页面链接
    found_m3u8_links, subpage_links = extract_links(html_content, url)

    # 将找到的.m3u8链接立即保存到文件中
    for link in found_m3u8_links:
        save_m3u8_link(link)

    # 如果还有剩余的深度，递归爬取子页面
    if depth < MAX_DEPTH and not should_stop.is_set():
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(crawl, subpage, depth + 1) for subpage in subpage_links]
            concurrent.futures.wait(futures)

# 主函数
def main():
    start_url = base_url
    start_time = time.time()

    # 使用多线程爬取起始页面及其子页面
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.submit(crawl, start_url, 0)

    end_time = time.time()
    print(f"Crawling completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()