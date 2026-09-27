"""
Module: gacha_fetcher.py
Deskripsi: Pengambil data resmi history gacha Zenless Zone Zero (ZZZ) via HoYoverse API.
Dikhususkan untuk 2 Banner Utama:
1. Karakter Eksklusif (Type 2 - Pity 90, 50:50)
2. W-Engine Eksklusif (Type 3 - Pity 80, 75:25)
"""

import time
import requests
import json
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

# Agen Standar S-Rank ZZZ (Jika dapat ini di banner karakter = Kalah 50:50)
STANDARD_AGENTS = [
    "Von Lycaon", "Alexandrina", "Grace Howard", 
    "Koleda Belobog", "Nekomiya Mana", "Soldier 11",
    "Lycaon", "Rina", "Grace", "Koleda", "Nekomata"
]

# W-Engine Standar S-Rank ZZZ (Jika dapat ini di banner W-Engine = Kalah 75:25)
STANDARD_W_ENGINES = [
    "The Brimstone", "Steel Cushion", "Fusion Compiler", 
    "The Restrained", "Weeping Cradle", "Hellfire Gears",
    "Belerang", "Bantalan Baja", "Kompilator Fusi",
    "Pengekang", "Buaian Menangis", "Roda Gigi Api Neraka"
]

# HANYA FOKUS KE 2 BANNER: KARAKTER EKSKLUSIF & W-ENGINE EKSKLUSIF
BANNER_CONFIG = [
    {"type": "2", "name": "Karakter Eksklusif", "max_pity": 90},
    {"type": "3", "name": "W-Engine Eksklusif", "max_pity": 80}
]

def clean_gacha_url(raw_url):
    parsed = urlparse(raw_url)
    qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    if "lang" not in qs:
        qs["lang"] = "id"
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return base_url, qs

def fetch_gacha_history(raw_url, max_pages_per_banner=80, sleep_sec=0.35):
    """
    Menarik seluruh history tarikan Karakter Eksklusif dan W-Engine Eksklusif.
    """
    base_url, base_params = clean_gacha_url(raw_url)
    if "authkey" not in base_params:
        return {"success": False, "error": "URL tidak valid: Parameter 'authkey' tidak ditemukan."}

    all_records = []
    uid = None

    for banner in BANNER_CONFIG:
        b_type = banner["type"]
        end_id = "0"
        page = 1
        has_more = True

        while has_more and page <= max_pages_per_banner:
            params = dict(base_params)
            params["gacha_type"] = b_type
            params["real_gacha_type"] = b_type
            params["end_id"] = end_id
            params["page"] = str(page)

            retry_count = 0
            res_json = None
            while retry_count < 3:
                try:
                    resp = requests.get(base_url, params=params, timeout=12)
                    res_json = resp.json()
                    if res_json.get("retcode") == -110:
                        time.sleep(1.2)
                        retry_count += 1
                        continue
                    break
                except Exception as e:
                    retry_count += 1
                    time.sleep(1.0)
                    if retry_count >= 3:
                        return {"success": False, "error": f"Koneksi gagal ke server HoYoverse: {str(e)}"}

            if not res_json:
                break

            retcode = res_json.get("retcode", -1)
            message = res_json.get("message", "")

            if retcode != 0:
                if "time out" in message.lower() or "auth key time out" in message.lower():
                    return {
                        "success": False, 
                        "error": "Authkey Kedaluwarsa (Expired)! Silakan buka menu Signal Search History di game ZZZ sebentar, lalu gunakan link terbaru."
                    }
                break

            data = res_json.get("data") or {}
            items = data.get("list", [])

            if not items:
                has_more = False
                break

            for it in items:
                if uid is None and "uid" in it:
                    uid = it["uid"]
                all_records.append(it)

            if items[-1]["id"] == end_id:
                has_more = False
                break

            end_id = items[-1]["id"]
            page += 1
            time.sleep(sleep_sec)

    return process_records(all_records, uid=uid)

def load_sample_history():
    sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_user_history.json")
    if not os.path.exists(sample_file):
        return {"success": False, "error": "File sample_user_history.json tidak ditemukan."}
    with open(sample_file, "r") as f:
        data = json.load(f)
    items = data.get("data", {}).get("list", [])
    return process_records(items, uid="150248911 (Akun Demo)")

def process_records(raw_items, uid=None):
    if not raw_items:
        return {"success": False, "error": "Tidak ada data tarikan untuk Banner Karakter / W-Engine dalam 6 bulan terakhir."}

    df = pd.DataFrame(raw_items)
    
    if "time" in df.columns:
        df = df.sort_values("time", ascending=True).reset_index(drop=True)

    df["rank_type"] = df["rank_type"].astype(int)
    df["gacha_type"] = df["gacha_type"].astype(str)

    # Hanya ambil banner 2 (Karakter) dan banner 3 (W-Engine)
    df = df[df["gacha_type"].isin(["2", "3"])].copy().reset_index(drop=True)
    if df.empty:
        return {"success": False, "error": "Tidak ditemukan data tarikan pada Banner Karakter Eksklusif maupun W-Engine Eksklusif."}

    banners = ["2", "3"]
    processed_dfs = []

    banner_stats = {}

    for b in banners:
        b_df = df[df["gacha_type"] == b].copy().reset_index(drop=True)
        max_p = 80 if b == "3" else 90
        b_name = "W-Engine Eksklusif" if b == "3" else "Karakter Eksklusif"
        
        pity_counter = 0
        pities = []
        is_guaranteed_list = []
        win_loss_list = []

        is_guaranteed = False
        for _, row in b_df.iterrows():
            pity_counter += 1
            pities.append(pity_counter)
            is_guaranteed_list.append(int(is_guaranteed))

            if row["rank_type"] == 4:
                item_name = row["name"]
                if b == "2": # Banner Karakter
                    if any(std.lower() in item_name.lower() for std in STANDARD_AGENTS):
                        win_loss_list.append("Kalah 50:50 (Standard)")
                        is_guaranteed = True
                    else:
                        win_loss_list.append("Menang 50:50 (Promosi)")
                        is_guaranteed = False
                elif b == "3": # Banner W-Engine
                    if any(std.lower() in item_name.lower() for std in STANDARD_W_ENGINES):
                        win_loss_list.append("Kalah 75:25 (Standard)")
                        is_guaranteed = True
                    else:
                        win_loss_list.append("Menang 75:25 (Promosi)")
                        is_guaranteed = False
                pity_counter = 0
            else:
                win_loss_list.append("-")

        b_df["pity"] = pities
        b_df["is_guaranteed"] = is_guaranteed_list
        b_df["gacha_result"] = win_loss_list
        processed_dfs.append(b_df)

        # Hitung statistik per banner
        if not b_df.empty:
            s_ranks = b_df[b_df["rank_type"] == 4]
            last_row = b_df.iloc[-1]
            current_pity = int(last_row["pity"]) if last_row["rank_type"] != 4 else 0
            is_guar = bool(last_row["is_guaranteed"])
            s_count = len(s_ranks)
            avg_pity = float(s_ranks["pity"].mean()) if s_count > 0 else (65.0 if b == "3" else 75.0)
            prev_pity = int(s_ranks.iloc[-1]["pity"]) if s_count > 0 else (65 if b == "3" else 75)

            wins = len(s_ranks[s_ranks["gacha_result"].str.contains("Menang", na=False)])
            losses = len(s_ranks[s_ranks["gacha_result"].str.contains("Kalah", na=False)])
            total_rate_events = wins + losses
            ratio_name = "75:25" if b == "3" else "50:50"
            default_win = 75.0 if b == "3" else 50.0
            winrate = (wins / total_rate_events * 100) if total_rate_events > 0 else (100.0 if s_count > 0 else default_win)

            banner_stats[b] = {
                "name": b_name,
                "max_pity": max_p,
                "current_pity": current_pity,
                "is_guaranteed": is_guar,
                "total_pulls": len(b_df),
                "s_count": s_count,
                "avg_pity": round(avg_pity, 1),
                "prev_pity": prev_pity,
                "winrate": round(winrate, 1),
                "rate_ratio_name": ratio_name
            }
        else:
            banner_stats[b] = {
                "name": b_name,
                "max_pity": max_p,
                "current_pity": 0,
                "is_guaranteed": False,
                "total_pulls": 0,
                "s_count": 0,
                "avg_pity": 65.0 if b == "3" else 75.0,
                "prev_pity": 65 if b == "3" else 75,
                "winrate": 75.0 if b == "3" else 50.0,
                "rate_ratio_name": "75:25" if b == "3" else "50:50"
            }

    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.sort_values("time", ascending=False).reset_index(drop=True)

    summary_stats = {
        "uid": uid or "Unknown",
        "total_pulls": len(final_df),
        "total_s_ranks": len(final_df[final_df["rank_type"] == 4]),
        "polychrome_spent": len(final_df) * 160,
        "banners": banner_stats
    }

    return {
        "success": True,
        "data": final_df,
        "stats": summary_stats
    }
