#!/usr/bin/env python3
"""
Skrip untuk scraping berita real dari situs berita terpercaya (Detik, Kompas).
Output: Dataset berita real dalam bahasa Indonesia (CSV).
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


def scrape_detik(max_articles=50):
    """Scrape berita dari Detik.com."""
    url = "https://www.detik.com/"
    articles = []
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Temukan semua artikel di halaman utama
        article_links = []
        for link in soup.select("a[href*='/news/']"):
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
                
                # Ekstrak isi berita (coba beberapa selector)
                text = ""
                for selector in [".detail__body-text", ".detail-text", ".article-body", ".content-detail"]:
                    content = article_soup.select_one(selector)
                    if content:
                        text = content.text.strip()
                        break
                
                if text and text != "No text":
                    articles.append({
                        "text": text,
                        "title": title,
                        "label": 0,  # 0 = Real
                        "source": "Detik",
                        "language": "Indonesia",
                        "url": link
                    })
                    print(f"✅ Scraped: {title[:50]}...")
                else:
                    print(f"⚠️ No text found for: {link}")
                
                time.sleep(DELAY + random.uniform(0, 0.5))  # Random delay
                
            except Exception as e:
                print(f"❌ Error scraping {link}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
    
    return pd.DataFrame(articles)


def scrape_kompas(max_articles=50):
    """Scrape berita dari Kompas.com."""
    url = "https://www.kompas.com/"
    articles = []
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Temukan semua artikel di halaman utama
        article_links = []
        for link in soup.select("a[href*='/read/']"):
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
                for selector in [".read__content", ".article-body", ".content-text"]:
                    content = article_soup.select_one(selector)
                    if content:
                        text = content.text.strip()
                        break
                
                if text and text != "No text":
                    articles.append({
                        "text": text,
                        "title": title,
                        "label": 0,  # 0 = Real
                        "source": "Kompas",
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
    print("🚀 Memulai scraping berita real (Bahasa Indonesia)...")
    
    # Scrape Detik
    print("\n📰 Scraping Detik.com...")
    df_detik = scrape_detik(max_articles=30)
    print(f"✅ Berhasil scrape {len(df_detik)} artikel dari Detik.")
    
    # Scrape Kompas
    print("\n📰 Scraping Kompas.com...")
    df_kompas = scrape_kompas(max_articles=30)
    print(f"✅ Berhasil scrape {len(df_kompas)} artikel dari Kompas.")
    
    # Gabungkan dataset
    df_real = pd.concat([df_detik, df_kompas], ignore_index=True)
    print(f"\n✅ Total artikel real: {len(df_real)}")
    
    # Simpan ke CSV
    output_path = "../../data/indonesian_real_news.csv"
    df_real.to_csv(output_path, index=False)
    print(f"💾 Dataset disimpan di: {output_path}")


if __name__ == "__main__":
    main()
