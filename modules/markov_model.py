"""
Module: markov_model.py
Deskripsi: Pemodelan Rantai Markov Penyerap (Absorbing Markov Chain)
dan Mesin Simulasi Monte Carlo Real-Time (10.000 Percobaan)
untuk Karakter Eksklusif (Pity 90, 50:50) dan W-Engine Eksklusif (Pity 80, 75:25).
"""

import numpy as np
import pandas as pd

def get_single_pull_rate(pity, banner_type="character"):
    """
    Menghitung probabilitas S-Rank pada tarikan ke-(pity).
    - Karakter: Base 0.6%, Soft Pity 74-89, Hard Pity 90 (100%)
    - W-Engine: Base 1.0%, Soft Pity 65-79, Hard Pity 80 (100%)
    """
    if banner_type == "w_engine":
        if pity < 65:
            return 0.010
        elif pity >= 80:
            return 1.0
        else:
            return 0.010 + 0.066 * (pity - 64)
    else:
        if pity < 74:
            return 0.006
        elif pity >= 90:
            return 1.0
        else:
            return 0.006 + 0.062 * (pity - 73)

class GachaMarkovChain:
    def __init__(self, banner_type="character"):
        self.banner_type = banner_type
        self.max_pity = 80 if banner_type == "w_engine" else 90
        self.promo_winrate = 0.75 if banner_type == "w_engine" else 0.50
        self.num_transient = self.max_pity
        self.Q, self.R = self._build_transition_matrix()
        self.N, self.expected_steps = self._solve_fundamental_matrix()

    def _build_transition_matrix(self):
        Q = np.zeros((self.num_transient, self.num_transient))
        R = np.zeros((self.num_transient, 1))

        for k in range(self.num_transient):
            next_pull_number = k + 1
            p_success = get_single_pull_rate(next_pull_number, self.banner_type)
            p_failure = 1.0 - p_success

            R[k, 0] = p_success

            if k < self.num_transient - 1:
                Q[k, k + 1] = p_failure

        return Q, R

    def _solve_fundamental_matrix(self):
        I = np.eye(self.num_transient)
        N = np.linalg.inv(I - self.Q)
        ones = np.ones((self.num_transient, 1))
        t = np.dot(N, ones).flatten()
        return N, t

    def get_expected_pulls_left(self, current_pity):
        if current_pity >= self.max_pity:
            return 0.0
        idx = int(np.clip(current_pity, 0, self.num_transient - 1))
        return float(self.expected_steps[idx])

    def get_cumulative_probability_curve(self, current_pity):
        curr = int(current_pity)
        pulls = []
        probabilities = []
        cumulative_not_hit = 1.0

        max_steps = self.max_pity - curr
        for step in range(1, max_steps + 1):
            target_pity = curr + step
            p_success = get_single_pull_rate(target_pity, self.banner_type)

            cumulative_not_hit *= (1.0 - p_success)
            p_hit = 1.0 - cumulative_not_hit
            pulls.append(step)
            probabilities.append(p_hit)

            if p_hit >= 0.9999 or target_pity >= self.max_pity:
                break

        return pd.DataFrame({
            "Tarikan_Tambahan": pulls,
            "Total_Pity": [curr + s for s in pulls],
            "Probabilitas_Kumulatif": probabilities
        })

    def get_expected_pulls_promotional(self, current_pity, is_guaranteed):
        """
        Ekspektasi tarikan sampai dapat target PROMOSI:
        Jika guaranteed: E[promo] = E[S-Rank saat ini]
        Jika 50:50 atau 75:25:
        E[promo] = E[S-Rank saat ini] + (Peluang Kalah * E[S-Rank dari pity 0])
        """
        e_first = self.get_expected_pulls_left(current_pity)
        if is_guaranteed:
            return e_first
        else:
            e_from_zero = self.get_expected_pulls_left(0)
            loss_prob = 1.0 - self.promo_winrate
            return e_first + (loss_prob * e_from_zero)

    def run_realtime_monte_carlo(self, current_pity, is_guaranteed, n_trials=10000):
        """
        Simulasi Real-Time 10.000 percobaan untuk mendapatkan distribusi empiris
        tarikan menuju karakter/W-Engine promosi.
        """
        rates = np.zeros(self.max_pity + 1)
        soft_p = 65 if self.banner_type == "w_engine" else 74
        base_r = 0.010 if self.banner_type == "w_engine" else 0.006
        inc_r = 0.066 if self.banner_type == "w_engine" else 0.062

        for p in range(1, self.max_pity + 1):
            if p < soft_p:
                rates[p] = base_r
            elif p >= self.max_pity:
                rates[p] = 1.0
            else:
                rates[p] = base_r + inc_r * (p - (soft_p - 1))

        results = []
        for _ in range(n_trials):
            # Tarikan S-Rank pertama dari pity saat ini
            pity = current_pity
            p1 = 0
            while True:
                p1 += 1
                pity += 1
                if np.random.random() < rates[min(pity, self.max_pity)]:
                    break

            if is_guaranteed or (np.random.random() < self.promo_winrate):
                results.append(p1)
            else:
                # Kalah, garansi aktif dari pity 0
                pity = 0
                p2 = 0
                while True:
                    p2 += 1
                    pity += 1
                    if np.random.random() < rates[min(pity, self.max_pity)]:
                        break
                results.append(p1 + p2)

        arr = np.array(results)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "prob_10": float(np.mean(arr <= 10) * 100),
            "prob_20": float(np.mean(arr <= 20) * 100),
            "prob_30": float(np.mean(arr <= 30) * 100),
            "prob_50": float(np.mean(arr <= 50) * 100)
        }

# Instance singleton
markov_char_model = GachaMarkovChain(banner_type="character")
markov_wengine_model = GachaMarkovChain(banner_type="w_engine")
