"""
Module: ml_models.py
Deskripsi: Pelatihan dan evaluasi model Machine Learning:
- Random Forest (Bagging Ensemble)
- XGBoost (Gradient Boosting)
Dilatih secara spesifik untuk 2 Banner:
1. Karakter Eksklusif (baseline_char.csv - Pity 90, 50:50)
2. W-Engine Eksklusif (baseline_wengine.csv - Pity 80, 75:25)
Mendukung prediksi S-Rank Promosi (Target Utama) dan S-Rank Terdekat (Next S-Rank).
"""

import time
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from modules.markov_model import markov_char_model, markov_wengine_model
except ImportError:
    from markov_model import markov_char_model, markov_wengine_model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from xgboost import XGBRegressor, XGBClassifier

CHAR_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "baseline_char.csv")
WENGINE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "baseline_wengine.csv")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "models_cache.pkl")

def categorize_pity(pity, is_wengine=False):
    if is_wengine:
        if pity <= 35:
            return 0 # Early
        elif pity <= 64:
            return 1 # Mid
        else:
            return 2 # Soft / Hard (65 - 80)
    else:
        if pity <= 40:
            return 0 # Early
        elif pity <= 73:
            return 1 # Mid
        else:
            return 2 # Soft / Hard (74 - 90)

def prepare_training_data(csv_path, is_wengine=False):
    df = pd.read_csv(csv_path)
    samples = []
    players = df["player_id"].unique()
    
    for pid in players:
        p_df = df[df["player_id"] == pid].sort_values("pull_number").copy()
        s_all = p_df[p_df["rank_type"] == 4]["pull_number"].tolist()
        s_promo = p_df[(p_df["rank_type"] == 4) & (p_df["is_promotional"] == 1)]["pull_number"].tolist()
        
        if not s_all or not s_promo:
            continue
            
        s_idx = 0
        p_idx = 0
        for _, row in p_df.iterrows():
            current_pull = row["pull_number"]
            while s_idx < len(s_all) and s_all[s_idx] < current_pull:
                s_idx += 1
            while p_idx < len(s_promo) and s_promo[p_idx] < current_pull:
                p_idx += 1
                
            if s_idx >= len(s_all) or p_idx >= len(s_promo):
                break
                
            pulls_s = s_all[s_idx] - current_pull
            pulls_promo = s_promo[p_idx] - current_pull
            target_pity_achieved = row["current_pity"] + pulls_s
            
            samples.append({
                "current_pity": row["current_pity"],
                "is_guaranteed": row["is_guaranteed"],
                "prev_s_pity": row["prev_s_pity"],
                "win_streak": row["win_streak"],
                "pulls_s": pulls_s,
                "pulls_promo": pulls_promo,
                "pity_achieved": target_pity_achieved,
                "pity_class": categorize_pity(target_pity_achieved, is_wengine=is_wengine)
            })
            
    return pd.DataFrame(samples)

class ModelManager:
    def __init__(self):
        self.char_models = {"reg_promo": {}, "reg_s": {}, "clf": {}}
        self.wengine_models = {"reg_promo": {}, "reg_s": {}, "clf": {}}
        self.char_metrics = {"reg": {}, "clf": {}}
        self.wengine_metrics = {"reg": {}, "clf": {}}
        self.feature_importance = {}
        self.is_trained = False
        self.feature_names = ["current_pity", "is_guaranteed", "win_streak", "prev_s_pity"]

    def _train_suite(self, csv_path, is_wengine=False):
        data = prepare_training_data(csv_path, is_wengine=is_wengine)
        X = data[self.feature_names]
        y_promo = data["pulls_promo"]
        y_s = data["pulls_s"]
        y_clf = data["pity_class"]

        X_train, X_test, yp_train, yp_test, ys_train, ys_test, yc_train, yc_test = train_test_split(
            X, y_promo, y_s, y_clf, test_size=0.2, random_state=42
        )

        # 1. Regresi Target Promosi (Random Forest & XGBoost)
        regs_promo = {
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
        }
        reg_metrics = {}
        fitted_regs_promo = {}
        for name, model in regs_promo.items():
            t0 = time.time()
            model.fit(X_train, yp_train)
            t_train = time.time() - t0
            preds = model.predict(X_test)
            mae = mean_absolute_error(yp_test, preds)
            rmse = np.sqrt(mean_squared_error(yp_test, preds))
            r2 = r2_score(yp_test, preds)
            fitted_regs_promo[name] = model
            reg_metrics[name] = {
                "MAE (Rata-rata Selisih Pull)": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2 Score": round(r2, 4),
                "Training Time (detik)": round(t_train, 4)
            }

        # 2. Regresi Target S-Rank Terdekat (Next S-Rank)
        regs_s = {
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        }
        fitted_regs_s = {}
        for name, model in regs_s.items():
            model.fit(X_train, ys_train)
            fitted_regs_s[name] = model

        # 3. Klasifikasi Kategori Pity (Random Forest & XGBoost)
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
        }
        clf_metrics = {}
        fitted_clfs = {}
        for name, model in classifiers.items():
            t0 = time.time()
            model.fit(X_train, yc_train)
            t_train = time.time() - t0
            preds = model.predict(X_test)
            acc = accuracy_score(yc_test, preds)
            prec = precision_score(yc_test, preds, average="weighted", zero_division=0)
            rec = recall_score(yc_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(yc_test, preds, average="weighted", zero_division=0)
            fitted_clfs[name] = model
            clf_metrics[name] = {
                "Akurasi": f"{acc * 100:.2f}%",
                "Precision": f"{prec * 100:.2f}%",
                "Recall": f"{rec * 100:.2f}%",
                "F1-Score": f"{f1 * 100:.2f}%",
                "Training Time (detik)": round(t_train, 4)
            }

        # Evaluasi Rantai Markov (Absorbing Markov Chain) pada Test Set
        active_mc = markov_wengine_model if is_wengine else markov_char_model
        mc_reg_preds = [
            active_mc.get_expected_pulls_promotional(row["current_pity"], bool(row["is_guaranteed"]))
            for _, row in X_test.iterrows()
        ]
        mc_mae = mean_absolute_error(yp_test, mc_reg_preds)
        mc_rmse = np.sqrt(mean_squared_error(yp_test, mc_reg_preds))
        mc_r2 = r2_score(yp_test, mc_reg_preds)

        all_reg_metrics = {
            "Absorbing Markov Chain": {
                "MAE (Rata-rata Selisih Pull)": round(mc_mae, 2),
                "RMSE": round(mc_rmse, 2),
                "R2 Score": round(mc_r2, 4),
                "Training Time (detik)": 0.0000
            }
        }
        all_reg_metrics.update(reg_metrics)

        # Evaluasi Klasifikasi Markov Chain (Maksimum Likelihood State: Soft/Hard Pity)
        mc_clf_preds = [2] * len(yc_test)
        mc_acc = accuracy_score(yc_test, mc_clf_preds)
        mc_prec = precision_score(yc_test, mc_clf_preds, average="weighted", zero_division=0)
        mc_rec = recall_score(yc_test, mc_clf_preds, average="weighted", zero_division=0)
        mc_f1 = f1_score(yc_test, mc_clf_preds, average="weighted", zero_division=0)

        all_clf_metrics = {
            "Absorbing Markov Chain": {
                "Akurasi": f"{mc_acc * 100:.2f}%",
                "Precision": f"{mc_prec * 100:.2f}%",
                "Recall": f"{mc_rec * 100:.2f}%",
                "F1-Score": f"{mc_f1 * 100:.2f}%",
                "Training Time (detik)": 0.0000
            }
        }
        all_clf_metrics.update(clf_metrics)

        return fitted_regs_promo, fitted_regs_s, fitted_clfs, all_reg_metrics, all_clf_metrics

    def train_and_evaluate_all(self, force_retrain=False):
        if not force_retrain and os.path.exists(CACHE_FILE):
            try:
                cached = joblib.load(CACHE_FILE)
                self.char_models = cached["char_models"]
                self.wengine_models = cached["wengine_models"]
                self.char_metrics = cached["char_metrics"]
                self.wengine_metrics = cached["wengine_metrics"]
                self.feature_importance = cached["feature_importance"]
                self.is_trained = True
                return self.char_metrics["reg"], self.char_metrics["clf"]
            except Exception:
                pass

        # Latih Karakter
        c_rp, c_rs, c_clfs, c_reg_m, c_clf_m = self._train_suite(CHAR_CSV, is_wengine=False)
        self.char_models["reg_promo"] = c_rp
        self.char_models["reg_s"] = c_rs
        self.char_models["clf"] = c_clfs
        self.char_metrics["reg"] = c_reg_m
        self.char_metrics["clf"] = c_clf_m

        # Latih W-Engine
        w_rp, w_rs, w_clfs, w_reg_m, w_clf_m = self._train_suite(WENGINE_CSV, is_wengine=True)
        self.wengine_models["reg_promo"] = w_rp
        self.wengine_models["reg_s"] = w_rs
        self.wengine_models["clf"] = w_clfs
        self.wengine_metrics["reg"] = w_reg_m
        self.wengine_metrics["clf"] = w_clf_m

        # Feature Importance
        self.feature_importance["Random Forest"] = dict(zip(
            self.feature_names, 
            np.round(self.char_models["reg_promo"]["Random Forest"].feature_importances_, 4)
        ))
        self.feature_importance["XGBoost"] = dict(zip(
            self.feature_names, 
            np.round(self.char_models["reg_promo"]["XGBoost"].feature_importances_, 4)
        ))

        # Simpan ke Cache File untuk pemuatan instan (<0.3s) di Cloud
        try:
            joblib.dump({
                "char_models": self.char_models,
                "wengine_models": self.wengine_models,
                "char_metrics": self.char_metrics,
                "wengine_metrics": self.wengine_metrics,
                "feature_importance": self.feature_importance
            }, CACHE_FILE)
        except Exception:
            pass

        self.is_trained = True
        return self.char_metrics["reg"], self.char_metrics["clf"]

    def predict_user_state(self, current_pity, is_guaranteed=0, prev_s_pity=None, avg_pity_history=None, win_streak=0, banner_type="2"):
        if not self.is_trained:
            self.train_and_evaluate_all()

        is_wengine = (str(banner_type) == "3")
        max_pity = 80 if is_wengine else 90
        
        if prev_s_pity is None:
            prev_s_pity = 65 if is_wengine else 75
        if avg_pity_history is None:
            avg_pity_history = 65.0 if is_wengine else 75.0
        
        regs_promo = self.wengine_models["reg_promo"] if is_wengine else self.char_models["reg_promo"]
        regs_s = self.wengine_models["reg_s"] if is_wengine else self.char_models["reg_s"]
        clfs = self.wengine_models["clf"] if is_wengine else self.char_models["clf"]

        features = pd.DataFrame([{
            "current_pity": current_pity,
            "is_guaranteed": int(is_guaranteed),
            "win_streak": win_streak,
            "prev_s_pity": prev_s_pity
        }])

        results = {
            "regression": {},
            "regression_next_s": {},
            "classification": {}
        }
        
        if is_wengine:
            class_map = {0: "Early (1 - 35)", 1: "Mid (36 - 64)", 2: "Soft/Hard Pity (65 - 80)"}
        else:
            class_map = {0: "Early (1 - 40)", 1: "Mid (41 - 73)", 2: "Soft/Hard Pity (74 - 90)"}

        # Prediksi Target Promosi
        max_possible_promo = (max_pity - current_pity) if is_guaranteed else (max_pity - current_pity + max_pity)
        for name, model in regs_promo.items():
            val = float(model.predict(features)[0])
            clamped = max(1, min(int(round(val)), max_possible_promo))
            results["regression"][name] = clamped

        # Prediksi S-Rank Terdekat
        for name, model in regs_s.items():
            val_s = float(model.predict(features)[0])
            clamped_s = max(1, min(int(round(val_s)), max_pity - current_pity))
            results["regression_next_s"][name] = clamped_s

        # Klasifikasi Rentang Pity
        for name, model in clfs.items():
            pred_class = int(model.predict(features)[0])
            probs = model.predict_proba(features)[0]
            confidence = float(np.max(probs))
            results["classification"][name] = {
                "kategori": class_map.get(pred_class, "Soft/Hard Pity"),
                "confidence": f"{confidence * 100:.1f}%",
                "probabilities": {class_map[i]: round(probs[i] * 100, 1) for i in range(len(probs))}
            }

        return results

ml_manager = ModelManager()

if __name__ == "__main__":
    ml_manager.train_and_evaluate_all()
    print("[BERHASIL] Model Random Forest & XGBoost selesai dilatih!")
