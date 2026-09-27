"""
Module: gacha_simulator.py
Deskripsi: Generator data simulasi gacha ZZZ realistis berbasis aturan resmi HoYoverse.
Menghasilkan dataset representatif untuk 2 Banner Utama:
1. Karakter Eksklusif (Type 2 - Pity 90, Soft 74, 50:50, Base 0.6%)
2. W-Engine Eksklusif (Type 3 - Pity 80, Soft 65, 75:25, Base 1.0%)
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

STANDARD_S_AGENTS = [
    "Von Lycaon", "Alexandrina", "Grace Howard", 
    "Koleda Belobog", "Nekomiya Mana", "Soldier 11"
]

FEATURED_S_AGENTS = [
    "Ellen Joe", "Zhu Yuan", "Qingyi", "Jane Doe", 
    "Caesar King", "Burnice White", "Yanagi", "Lighter", "Hoshimi Miyabi"
]

STANDARD_S_W_ENGINES = [
    "The Brimstone", "Steel Cushion", "Fusion Compiler", 
    "The Restrained", "Weeping Cradle", "Hellfire Gears"
]

FEATURED_S_W_ENGINES = [
    "Deep Sea Visitor", "Riot Suppressor Mark VI", "Ice-Jade Teapot",
    "Sharpened Stinger", "Tusks of Fury", "Flamemaker Shaker", "Timeweaver"
]

def get_pull_rate(pity_count, banner_type=2):
    """
    Menghitung probabilitas S-Rank berdasarkan jenis banner:
    - Banner Karakter (2): Base 0.6%, Soft 74-89, Hard 90
    - Banner W-Engine (3): Base 1.0%, Soft 65-79, Hard 80
    """
    if banner_type == 3:
        if pity_count < 65:
            return 0.010
        elif pity_count >= 80:
            return 1.0
        else:
            return 0.010 + 0.066 * (pity_count - 64)
    else:
        if pity_count < 74:
            return 0.006
        elif pity_count >= 90:
            return 1.0
        else:
            return 0.006 + 0.062 * (pity_count - 73)

def simulate_player_history(player_id="player_001", total_pulls=600, banner_type=2):
    """
    Simulasi riwayat tarikan pemain dengan tracking pity dan status guarantee.
    """
    max_pity = 80 if banner_type == 3 else 90
    win_prob = 0.75 if banner_type == 3 else 0.50
    featured_list = FEATURED_S_W_ENGINES if banner_type == 3 else FEATURED_S_AGENTS
    standard_list = STANDARD_S_W_ENGINES if banner_type == 3 else STANDARD_S_AGENTS
    
    records = []
    pity_s = 0
    pity_a = 0
    is_guaranteed = False
    
    total_pity_s_list = []
    win_streak = 0
    current_time = datetime.now() - timedelta(days=90)
    
    for i in range(1, total_pulls + 1):
        pity_s += 1
        pity_a += 1
        current_time += timedelta(minutes=random.randint(2, 60))
        
        prob_s = get_pull_rate(pity_s, banner_type=banner_type)
        rand_val = random.random()
        
        prev_s_pity = total_pity_s_list[-1] if len(total_pity_s_list) > 0 else (65 if banner_type == 3 else 75)
        avg_pity = float(np.mean(total_pity_s_list)) if len(total_pity_s_list) > 0 else (65.0 if banner_type == 3 else 75.0)
        
        snapshot = {
            "player_id": player_id,
            "pull_number": i,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_pity": pity_s,
            "pity_a": pity_a,
            "is_guaranteed": int(is_guaranteed),
            "prev_s_pity": prev_s_pity,
            "avg_pity_history": avg_pity,
            "win_streak": win_streak,
            "banner_type": banner_type
        }
        
        if rand_val < prob_s:
            # Dapat S-Rank
            rank_type = 4
            pity_record = pity_s
            total_pity_s_list.append(pity_record)
            
            # Cek status promosi
            if is_guaranteed:
                item_name = random.choice(featured_list)
                is_guaranteed = False
                is_promotional = 1
                win_streak += 1
            else:
                if random.random() < win_prob:
                    item_name = random.choice(featured_list)
                    is_guaranteed = False
                    is_promotional = 1
                    win_streak += 1
                else:
                    item_name = random.choice(standard_list)
                    is_guaranteed = True
                    is_promotional = 0
                    win_streak = 0
                    
            item_type = "W-Engine" if banner_type == 3 else "Agent"
            pity_s = 0
            pity_a = 0
        elif pity_a >= 10 or random.random() < 0.094:
            rank_type = 3
            item_name = f"A-Rank Item {random.randint(1, 10)}"
            item_type = "W-Engine" if banner_type == 3 else "Agent"
            is_promotional = 0
            pity_record = pity_a
            pity_a = 0
        else:
            rank_type = 2
            item_name = f"B-Rank Item {random.randint(1, 10)}"
            item_type = "W-Engine"
            is_promotional = 0
            pity_record = 1
            
        snapshot["item_name"] = item_name
        snapshot["item_type"] = item_type
        snapshot["rank_type"] = rank_type
        snapshot["pity_used"] = pity_record
        snapshot["is_promotional"] = is_promotional
        
        records.append(snapshot)
        
    return pd.DataFrame(records)

def generate_dual_baseline_datasets(num_players=30, pulls_per_player=800):
    """
    Membuat 2 dataset baseline terpisah:
    1. data/baseline_char.csv (Karakter Eksklusif)
    2. data/baseline_wengine.csv (W-Engine Eksklusif)
    """
    for b_type, filename in [(2, "data/baseline_char.csv"), (3, "data/baseline_wengine.csv")]:
        all_dfs = []
        name = "W-Engine" if b_type == 3 else "Karakter"
        print(f"[*] Men-generate dataset {name} dari {num_players} akun pemain...")
        for idx in range(1, num_players + 1):
            pid = f"UID_{15000000 + idx}"
            df_p = simulate_player_history(player_id=pid, total_pulls=pulls_per_player, banner_type=b_type)
            all_dfs.append(df_p)
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(filename, index=False)
        print(f"[BERHASIL] Dataset {filename} selesai dibuat ({len(full_df)} baris).")

if __name__ == "__main__":
    generate_dual_baseline_datasets()
