"""Tes untuk peningkatan UX UI (belum di-commit, menunggu perintah pengguna).

Tes ini hanya memeriksa struktur template — tidak memerlukan browser.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(BASE_DIR, "templates", "index.html")


def html():
    with open(TEMPLATE, encoding="utf-8") as fh:
        return fh.read()


def test_alat_bantu_input_tersedia():
    page = html()
    for element_id in (
        "btn-sample-news",
        "btn-sample-chain",
        "btn-paste",
        "input-hint",
        "tool-note",
    ):
        assert f'id="{element_id}"' in page, element_id
    assert "Tips agar hasil lebih akurat" in page
    assert "Min. 10 karakter · Ctrl+Enter untuk memeriksa" in page


def test_status_ditampilkan_sebelum_detail_pencocokan():
    """Kesimpulan harus muncul lebih awal daripada bagian 'Mirip dengan klaim'."""
    page = html()
    badge = page.index("makeEl('span', 'badge '")
    debunk = page.index("const debunked =")
    assert badge < debunk


def test_fitur_salin_dan_coba_lagi_tersedia():
    page = html()
    assert "Salin ringkasan" in page
    assert "copySummary" in page
    assert "Coba lagi" in page
    assert "skeleton-line" in page
    assert "aria-busy" in page


def test_legenda_status_tersedia():
    page = html()
    assert "Apa arti status ini?" in page
    assert "Belum dapat disimpulkan" in page


def test_mode_keandalan_ada_di_atas_halaman():
    page = html()
    mode = page.index('id="system-mode"')
    hero = page.index('class="hero"')
    assert mode < hero


def test_tidak_ada_skrip_eksternal_tambahan():
    """CSP hanya mengizinkan inline script; jangan tambahkan <script src=...>."""
    page = html()
    assert not re.search(r"<script[^>]*\bsrc\s*=", page, re.IGNORECASE)


def test_pesan_503_dan_flag_mode_tetap_ada():
    """Perbaikan UX tidak boleh menghapus perbaikan sebelumnya."""
    page = html()
    assert "status === 503" in page
    assert "Model belum siap" in page
    assert "models_missing" in page
    assert 'data-mode="{{' in page


def test_urutan_blok_area_input():
    """textarea -> chip -> hint -> notice -> tips -> footer aksi."""
    page = html()
    pos = [
        page.index('id="news-text"'),
        page.index('class="input-tools"'),
        page.index('id="input-hint"'),
        page.index('class="privacy-notice"'),
        page.index('class="tips"'),
        page.index('class="actions"'),
    ]
    assert pos == sorted(pos)


def test_aksesibilitas_area_input():
    page = html()
    for label in (
        'aria-label="Muat contoh berita"',
        'aria-label="Muat contoh pesan berantai"',
        'aria-label="Tempel teks dari clipboard"',
        'aria-expanded="false"',
        'aria-describedby="btn-check-help"',
        'id="btn-check-help"',
        "Masukkan minimal 10 karakter",
    ):
        assert label in page, label
    assert "tips-arrow" in page
    assert ".btn-primary:focus-visible" in page
    assert ".chip:focus-visible" in page


def test_skala_spasi_dan_state_disabled_memakai_token():
    page = html()
    for token in ("--space-1: 4px", "--space-2: 8px", "--space-3: 12px",
                  "--space-4: 16px", "--space-6: 24px", "--disabled-bg"):
        assert token in page, token
    assert "min-height: 36px" in page
    assert "min-height: 44px" in page
    assert "min-height: 46px" in page
    assert "cursor: not-allowed" in page
    assert "@media (max-width: 30rem)" in page
