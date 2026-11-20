#!/usr/bin/env python3
"""
Skrip untuk menggabungkan dataset berita real dan hoax (Bahasa Indonesia + Bahasa Daerah).
Output: Dataset lengkap untuk pelatihan model.
"""

import pandas as pd
import os


def load_dataset(filepath):
    """Load dataset dari file CSV."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"⚠️ File tidak ditemukan: {filepath}")
        return pd.DataFrame()


def main():
    print("🚀 Menggabungkan dataset...")
    
    # Path dataset
    data_dir = "../../data"
    
    # Load dataset Bahasa Indonesia
    print("\n📂 Memuat dataset Bahasa Indonesia...")
    df_real_id = load_dataset(f"{data_dir}/indonesian_real_news.csv")
    df_fake_id = load_dataset(f"{data_dir}/indonesian_fake_news.csv")
    
    print(f"✅ Real (Indonesia): {len(df_real_id)} artikel")
    print(f"✅ Fake (Indonesia): {len(df_fake_id)} artikel")
    
    # Load dataset Bahasa Daerah
    print("\n📂 Memuat dataset Bahasa Daerah...")
    df_regional = load_dataset(f"{data_dir}/regional_news.csv")
    
    print(f"✅ Bahasa Daerah: {len(df_regional)} artikel")
    
    # Gabungkan semua dataset
    df_combined = pd.concat([df_real_id, df_fake_id, df_regional], ignore_index=True)
    
    # Hapus duplikasi
    df_combined = df_combined.drop_duplicates(subset=["text"], keep="first")
    
    # Cek keseimbangan kelas
    label_counts = df_combined["label"].value_counts()
    print(f"\n📊 Keseimbangan kelas:")
    print(f"   Real (0): {label_counts.get(0, 0)} artikel")
    print(f"   Fake (1): {label_counts.get(1, 0)} artikel")
    
    # Cek distribusi bahasa
    language_counts = df_combined["language"].value_counts()
    print(f"\n📊 Distribusi bahasa:")
    for lang, count in language_counts.items():
        print(f"   {lang}: {count} artikel")
    
    # Simpan dataset gabungan
    output_path = f"{data_dir}/combined_news.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"\n💾 Dataset gabungan disimpan di: {output_path}")
    print(f"✅ Total artikel: {len(df_combined)}")


if __name__ == "__main__":
    main()
