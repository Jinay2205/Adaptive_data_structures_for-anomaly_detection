import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import CountMinSketch, AdaSketch, StableLearnedBloomFilter, SlidingWindowCBF
from utils import load_nab_file, get_oracle_labels, quantize, evaluate_predictions

def run_experiment():
    file_target = 'art_daily_jumpsup.csv' 
    
    if not os.path.exists(file_target):
        print(f"File {file_target} not found! Please upload it.")
        return

    print(f"Running Analysis on {file_target}...")
    
    # 1. Load Data
    df = load_nab_file(file_target)
    gt_labels = get_oracle_labels(df, contamination=0.03) # Oracle finds top 3% anomalies
    
    # 2. Initialize Models with Tuned Parameters
    cms = CountMinSketch(width=1000, depth=5)
    
    # AdaSketch: Decay=0.9 means it forgets 10% of counts every 50 steps
    ada = AdaSketch(width=1000, depth=5, decay=0.9, decay_window=50) 
    
    cbf = SlidingWindowCBF(width=2000, window_size=2000)
    slbf = StableLearnedBloomFilter(width=2000, window_size=2000)
    
    results = {
        'cms_scores': [],
        'ada_scores': [],
        'cbf_anoms': [],
        'slbf_anoms': []
    }
    
    print("Streaming data...")
    for x in df['value']:
        qx = quantize(x, scale=1) # Reduced scale to group noise together
        
        # CMS
        cnt_cms = cms.query(qx)
        cms.update(qx)
        results['cms_scores'].append(1.0 / (cnt_cms + 0.1))
        
        # AdaSketch
        cnt_ada = ada.query(qx)
        ada.update(qx)
        results['ada_scores'].append(1.0 / (cnt_ada + 0.1))
        
        # CBF
        is_anom_cbf = 0 if cbf.contains(qx) else 1
        cbf.add(qx)
        results['cbf_anoms'].append(is_anom_cbf)
        
        # SLBF
        is_anom_slbf = 1 if slbf.check_and_add(qx, x) else 0
        results['slbf_anoms'].append(is_anom_slbf)

    # 3. Dynamic Thresholding (Percentile Based)
    # We assume roughly 3% of data is anomalous (matching Oracle)
    # This aligns the "sensitivity" of our sketch to the Oracle
    threshold_cms = np.percentile(results['cms_scores'], 97)
    threshold_ada = np.percentile(results['ada_scores'], 97)
    
    pred_cms = (np.array(results['cms_scores']) > threshold_cms).astype(int)
    pred_ada = (np.array(results['ada_scores']) > threshold_ada).astype(int)
    
    metrics = {
        'CMS': evaluate_predictions(gt_labels, pred_cms),
        'Ada-Sketch': evaluate_predictions(gt_labels, pred_ada),
        'CBF': evaluate_predictions(gt_labels, results['cbf_anoms']),
        'SLBF': evaluate_predictions(gt_labels, results['slbf_anoms'])
    }
    
    print("\n--- Final Results (vs Batch Oracle) ---")
    print(json.dumps(metrics, indent=2))
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['value'], label='Stream', color='gray', alpha=0.5)
    anom_idx = np.where(gt_labels==1)[0]
    plt.scatter(df['timestamp'].iloc[anom_idx], df['value'].iloc[anom_idx], color='red', label='Oracle (Ground Truth)', s=15)
    plt.title("Ground Truth")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], results['ada_scores'], label='Ada Score', color='blue', alpha=0.6)
    plt.plot(df['timestamp'], results['cms_scores'], label='CMS Score', color='orange', alpha=0.6, linestyle='--')
    plt.axhline(threshold_ada, color='blue', linestyle=':', label='Ada Threshold')
    plt.title("Anomaly Scores: Ada-Sketch (Blue) vs CMS (Orange)")
    plt.yscale('log') # Log scale helps see the differences
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['value'], color='gray', alpha=0.3)
    slbf_idx = np.where(np.array(results['slbf_anoms'])==1)[0]
    plt.scatter(df['timestamp'].iloc[slbf_idx], df['value'].iloc[slbf_idx], marker='^', color='green', label='SLBF Detection', s=20)
    plt.title("Membership Detection (SLBF)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('final_project_plot.png')
    print("Plot saved.")
    
    with open('final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    run_experiment()