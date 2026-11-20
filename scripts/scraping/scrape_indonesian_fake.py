#!/usr/bin/env python3
"""
Skrip untuk scraping berita hoax dari TurnBackHoax.id.
Output: Dataset berita hoax dalam bahasa Indonesia (CSV).
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


def scrape_turnbackhoax(max_articles=50):
    """Scrape berita hoax dari TurnBackHoax.id."""
    url = "https://turnbackhoax.id/"
    articles = []
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Temukan semua tautan artikel
        article_links = []
        for link in soup.select("a[href*='/hoax/'], a[href*='/fake/']"):
            href = link.get("href")
            if href and href not in article_links:
                article_links.append(urljoin(url, href))
        
        # Jika tidak ditemukan, coba selector lain
        if not article_links:
            for link in soup.select("a[rel='bookmark']"):
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
                for selector in [".entry-content", ".post-content", ".article-body", ".content"]:
                    content = article_soup.select_one(selector)
                    if content:
                        text = content.text.strip()
                        break
                
                if text and text != "No text":
                    articles.append({
                        "text": text,
                        "title": title,
                        "label": 1,  # 1 = Fake
                        "source": "TurnBackHoax",
                        "language": "Indonesia",
                        "url": link
                    })
                    print(f"✅ Scraped: {title[:50]}...")
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
    print("🚀 Memulai scraping berita hoax (Bahasa Indonesia)...")
    
    # Scrape TurnBackHoax
    print("\n📰 Scraping TurnBackHoax.id...")
    df_fake = scrape_turnbackhoax(max_articles=50)
    print(f"✅ Berhasil scrape {len(df_fake)} artikel hoax.")
    
    # Simpan ke CSV
    output_path = "../../data/indonesian_fake_news.csv"
    df_fake.to_csv(output_path, index=False)
    print(f"💾 Dataset disimpan di: {output_path}")


if __name__ == "__main__":
    main()
