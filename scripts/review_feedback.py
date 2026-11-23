#!/usr/bin/env python3
"""CLI review masukan feedback dari /feedback endpoint.

Alur kerja:
1. /feedback menyimpan masukan baru ke data/feedback.csv dengan status "baru".
2. Jalankan skrip ini untuk meninjau baris "baru" satu per satu:
       python scripts/review_feedback.py
3. Setiap baris ditampilkan; pilih [s]etujui / [t]olak / [l]ewati / [b]atal.
4. Baris yang disetujui ditulis ke data/feedback_reviewed.csv dengan status
   "disetujui". Pipeline pelatihan (train_model*.py) hanya membaca baris
   dengan status "disetujui" dari feedback_reviewed.csv.

Contoh:
    python scripts/review_feedback.py --file data/feedback.csv \
        --output data/feedback_reviewed.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

DEFAULT_INPUT = os.path.join('data', 'feedback.csv')
DEFAULT_OUTPUT = os.path.join('data', 'feedback_reviewed.csv')

REVIEWED_FIELDS = ['id', 'timestamp', 'text_hash', 'text', 'label',
                   'status', 'reviewer']


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def save_reviewed(path: str, rows: list[dict]) -> None:
    need_header = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REVIEWED_FIELDS)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def truncate(text: str, width: int = 120) -> str:
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text if len(text) <= width else text[: width - 3] + '...'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Tinjau masukan feedback baru dan setujui/tolak.')
    parser.add_argument('--file', default=DEFAULT_INPUT,
                        help=f'Berkas masukan feedback (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Berkas hasil review (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--limit', type=int, default=0,
                        help='Batas jumlah baris yang ditinjau (0 = tanpa batas)')
    args = parser.parse_args()

    rows = load_rows(args.file)
    if not rows:
        print(f'Tidak ada baris feedback di {args.file}')
        return 0

    pending = [r for r in rows if r.get('status', '').strip() == 'baru']
    if not pending:
        print(f'Tidak ada baris dengan status "baru" di {args.file}')
        return 0

    # Kumpulkan text_hash yang sudah ada di output agar tidak meninjau dua kali.
    existing_hashes = set()
    if os.path.exists(args.output):
        for r in load_rows(args.output):
            h = r.get('text_hash')
            if h:
                existing_hashes.add(h)

    reviewed: list[dict] = []
    count = 0
    for i, row in enumerate(pending, 1):
        text_hash = row.get('text_hash', '')
        if text_hash in existing_hashes:
            print(f'[{i}/{len(pending)}] (lewat, sudah direview sebelumnya)')
            continue

        print(f'\n--- Baris {i} dari {len(pending)} ---')
        print(f'ID      : {row.get("id", "-")}')
        print(f'Waktu   : {row.get("timestamp", "-")}')
        print(f'Teks    : {truncate(row.get("text", ""))}')
        print(f'Label   : {row.get("label", "-")}  (0=Real, 1=Palsu)')
        if row.get('reviewer'):
            print(f'Reviewer: {row["reviewer"]}')

        while True:
            pilihan = input(
                '[s]etujui / [t]olak / [l]ewati / [b]atal: ').strip().lower()
            if pilihan in ('s', 'setujui'):
                reviewed.append({
                    'id': str(uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'text_hash': text_hash or '',
                    'text': row.get('text', ''),
                    'label': row.get('label', ''),
                    'status': 'disetujui',
                    'reviewer': '',
                })
                existing_hashes.add(text_hash)
                count += 1
                print('  -> DISETUJUI')
                break
            elif pilihan in ('t', 'tolak'):
                reviewed.append({
                    'id': str(uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'text_hash': text_hash or '',
                    'text': row.get('text', ''),
                    'label': row.get('label', ''),
                    'status': 'ditolak',
                    'reviewer': '',
                })
                existing_hashes.add(text_hash)
                count += 1
                print('  -> DITOLAK')
                break
            elif pilihan in ('l', 'lewati'):
                print('  -> LEWATI')
                break
            elif pilihan in ('b', 'batal'):
                print('  -> DIBATALKAN oleh pengguna')
                break
            else:
                print('  Pilihan tidak dikenal.')

        if pilihan in ('b', 'batal'):
            break
        if args.limit and count >= args.limit:
            print(f'\nBatas review tercapai ({args.limit}).')
            break

    if reviewed:
        save_reviewed(args.output, reviewed)
        print(f'\n{count} baris ditulis ke {args.output}')
    else:
        print('\nTidak ada baris yang ditulis.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
