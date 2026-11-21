#!/usr/bin/env bash
# Jalankan sistem Fake News Detector dengan virtual environment.
# Tidak perlu mengaktifkan venv secara manual.

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

if [ ! -d venv ]; then
    echo "Virtual environment tidak ditemukan. Buat terlebih dahulu:"
    echo "  python3 -m venv venv"
    echo "  ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "🚀 Memulai Fake News Detector..."
./venv/bin/python app.py
