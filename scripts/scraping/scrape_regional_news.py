#!/usr/bin/env python3
"""
Skrip untuk scraping berita dalam bahasa daerah (Jawa, Sunda, Batak).
Output: Dataset berita real dalam bahasa daerah (CSV).
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin

# User-Agent untuk menghindari blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Delay untuk menghindari rate limiting
DELAY = 1  # detik


def scrape_jawapos(max_articles=30):
    """Scrape berita bahasa Jawa dari Jawapos.com."""
    url = "https://www.jawapos.com/"
    articles = []
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Temukan tautan artikel
        article_links = []
        for link in soup.select("a[href*='/radar-']"):  # Contoh: /radar-malang/
            href = link.get("href")
            if href and href not in article_links:
                article_links.append(urljoin(url, href))
        
        # Scrape setiap artikel
        for link in article_links[:max_articles]:
            try:
                article_response = requests.get(link, headers=HEADERS)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.text, "html.parser")
                
                # Ekstrak judul
                title = article_soup.select_one("h1").text.strip() if article_soup.select_one("h1") else "No title"
                
                # Ekstrak isi berita
                text = ""
                for selector in [".article-body", ".content-detail", ".post-content"]:
                    content = article_soup.select_one(selector)
                    if content:
                        text = content.text.strip()
                        break
                
                if text and text != "No text":
                    articles.append({
                        "text": text,
                        "title": title,
                        "label": 0,  # 0 = Real
                        "source": "Jawapos",
                        "language": "Jawa",
                        "url": link
                    })
                    print(f"✅ Scraped (Jawa): {title[:50]}...")
                else:
                    print(f"⚠️ No text found for: {link}")
                
                time.sleep(DELAY + random.uniform(0, 0.5))
                
            except Exception as e:
                print(f"❌ Error scraping {link}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
    
    return pd.DataFrame(articles)


def scrape_pikiran_rakyat(max_articles=30):
    """Scrape berita bahasa Sunda dari Pikiran Rakyat."""
    url = "https://www.pikiran-rakyat.com/"
    articles = []
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Temukan tautan artikel
        article_links = []
        for link in soup.select("a[href*='/sunda/'], a[href*='/priangan/']"):
            href = link.get("href")
            if href and href not in article_links:
                article_links.append(urljoin(url, href))
        
        # Scrape setiap artikel
        for link in article_links[:max_articles]:
            try:
                article_response = requests.get(link, headers=HEADERS)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.text, "html.parser")
                
                # Ekstrak judul
                title = article_soup.select_one("h1").text.strip() if article_soup.select_one("h1") else "No title"
                
                # Ekstrak isi berita
                text = ""
                for selector in [".article-body", ".content-detail", ".post-content"]:
                    content = article_soup.select_one(selector)
                    if content:
                        text = content.text.strip()
                        break
                
                if text and text != "No text":
                    articles.append({
                        "text": text,
                        "title": title,
                        "label": 0,  # 0 = Real
                        "source": "Pikiran Rakyat",
                        "language": "Sunda",
                        "url": link
                    })
                    print(f"✅ Scraped (Sunda): {title[:50]}...")
                else:
                    print(f"⚠️ No text found for: {link}")
                
                time.sleep(DELAY + random.uniform(0, 0.5))
                
            except Exception as e:
                print(f"❌ Error scraping {link}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
    
    return pd.DataFrame(articles)


def scrape_harian_medan(max_articles=30):
    """Scrape berita bahasa Batak dari Harian Medan."""
    url = "https://www.harianmedan.com/"
    articles = []
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Temukan tautan artikel
        article_links = []
        for link in soup.select("a[href*='/berita/']"):
            href = link.get("href")
            if href and href not in article_links:
                article_links.append(urljoin(url, href))
        
        # Scrape setiap artikel
        for link in article_links[:max_articles]:
            try:
                article_response = requests.get(link, headers=HEADERS)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.text, "html.parser")
                
                # Ekstrak judul
                title = article_soup.select_one("h1").text.strip() if article_soup.select_one("h1") else "No title"
                
                # Ekstrak isi berita
                text = ""
                for selector in [".article-body", ".content-detail", ".post-content"]:
                    content = article_soup.select_one(selector)
                    if content:
                        text = content.text.strip()
                        break
                
                if text and text != "No text":
                    articles.append({
                        "text": text,
                        "title": title,
                        "label": 0,  # 0 = Real
                        "source": "Harian Medan",
                        "language": "Batak",
                        "url": link
                    })
                    print(f"✅ Scraped (Batak): {title[:50]}...")
                else:
                    print(f"⚠️ No text found for: {link}")
                
                time.sleep(DELAY + random.uniform(0, 0.5))
                
            except Exception as e:
                print(f"❌ Error scraping {link}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
    
    return pd.DataFrame(articles)


def main():
    print("🚀 Memulai scraping berita bahasa daerah...")
    
    # Scrape Jawa (Jawapos)
    print("\n📰 Scraping Jawapos (Bahasa Jawa)...")
    df_jawa = scrape_jawapos(max_articles=20)
    print(f"✅ Berhasil scrape {len(df_jawa)} artikel bahasa Jawa.")
    
    # Scrape Sunda (Pikiran Rakyat)
    print("\n📰 Scraping Pikiran Rakyat (Bahasa Sunda)...")
    df_sunda = scrape_pikiran_rakyat(max_articles=20)
    print(f"✅ Berhasil scrape {len(df_sunda)} artikel bahasa Sunda.")
    
    # Scrape Batak (Harian Medan)
    print("\n📰 Scraping Harian Medan (Bahasa Batak)...")
    df_batak = scrape_harian_medan(max_articles=20)
    print(f"✅ Berhasil scrape {len(df_batak)} artikel bahasa Batak.")
    
    # Gabungkan dataset
    df_regional = pd.concat([df_jawa, df_sunda, df_batak], ignore_index=True)
    print(f"\n✅ Total artikel bahasa daerah: {len(df_regional)}")
    
    # Simpan ke CSV
    output_path = "../../data/regional_news.csv"
    df_regional.to_csv(output_path, index=False)
    print(f"💾 Dataset disimpan di: {output_path}")


if __name__ == "__main__":
    main()
