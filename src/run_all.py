"""
run_all.py — Master Experiment Runner
======================================
Runs the complete SHAP-DA-IDS pipeline in sequence:
  1. Data Loading
  2. Preprocessing
  3. SHAP Feature Selection
  4. Difficulty-Adaptive Augmentation
  5. Model Training & Evaluation
  6. SHAP Explainability Analysis
  7. Figure Generation

Logs are written to results/experiment_log.txt.
All results are saved automatically.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import sys, os, time, io
from datetime import datetime

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
SRC_DIR = os.path.join(BASE_DIR, "src")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "experiment_log.txt")

sys.path.insert(0, SRC_DIR)


class Tee:
    """Redirect stdout to both console and log file."""
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = file
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_phase(phase_num, phase_name, func):
    """Run a pipeline phase with timing and error handling."""
    print(f"\n{'='*70}")
    print(f"PHASE {phase_num}: {phase_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        print(f"\n✓ Phase {phase_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n✗ Phase {phase_num} FAILED after {elapsed:.1f}s")
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(log_file)
        
        print("=" * 70)
        print("SHAP-DA-IDS EXPERIMENT PIPELINE")
        print("Difficulty-Adaptive Augmentation with Explainable Feature Selection")
        print("for Imbalanced IoT Intrusion Detection")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base directory: {BASE_DIR}")
        print(f"Log: {LOG_PATH}")
        
        total_start = time.time()
        results = {}
        
        # Phase 1: Data Loading
        import importlib.util
        
        def load_phase():
            spec = importlib.util.spec_from_file_location(
                "data_loading", os.path.join(SRC_DIR, "01_data_loading.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.load_dataset(mod.DATA_PATH)
        
        def preprocess_phase():
            spec = importlib.util.spec_from_file_location(
                "preprocessing", os.path.join(SRC_DIR, "02_preprocessing.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.load_and_preprocess()
        
        def shap_selection_phase():
            spec = importlib.util.spec_from_file_location(
                "shap_sel", os.path.join(SRC_DIR, "03_shap_feature_selection.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        
        def augmentation_phase():
            spec = importlib.util.spec_from_file_location(
                "aug", os.path.join(SRC_DIR, "04_difficulty_adaptive_augmentation.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        
        def training_phase():
            spec = importlib.util.spec_from_file_location(
                "training", os.path.join(SRC_DIR, "05_model_training.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        
        def explainability_phase():
            spec = importlib.util.spec_from_file_location(
                "explain", os.path.join(SRC_DIR, "06_shap_explainability.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        
        def figures_phase():
            spec = importlib.util.spec_from_file_location(
                "figures", os.path.join(SRC_DIR, "07_generate_figures.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        
        phases = [
            (1, "Data Loading and Exploration", load_phase),
            (2, "Preprocessing (imputation, encoding, scaling, split)", preprocess_phase),
            (3, "SHAP Feature Selection", shap_selection_phase),
            (4, "Difficulty-Adaptive Augmentation", augmentation_phase),
            (5, "Model Training and Evaluation", training_phase),
            (6, "SHAP Explainability Analysis", explainability_phase),
            (7, "Figure Generation", figures_phase),
        ]
        
        for phase_num, phase_name, func in phases:
            success = run_phase(phase_num, phase_name, func)
            results[phase_name] = "✓ Success" if success else "✗ Failed"
            if not success:
                print(f"\nStopping pipeline at phase {phase_num}.")
                break
        
        total_elapsed = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"PIPELINE SUMMARY")
        print(f"Total runtime: {total_elapsed:.1f}s")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        for phase_name, status in results.items():
            print(f"  {status}  {phase_name}")
        
        sys.stdout = sys.stdout.terminal
    
    print(f"\nExperiment log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
