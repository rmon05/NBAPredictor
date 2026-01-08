import subprocess
import sys

PYTHON_EXE = sys.executable
STEPS = [
    # "playwright/stathead_scraper.py",
    "playwright/nba_live_scraper.py",
    "core/raw_to_clean.py",
    "core/clean_to_joined.py",
    "../ml/predict.py"
]

def main():    
    for script in STEPS:
        print(f"\nRunning {script}...")
        try:
            result = subprocess.run(
                [PYTHON_EXE, script], 
                check=True, 
                text=True,
                capture_output=False
            )
            print(f"Finished {script} successfully.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {script} failed. Stopping pipeline.")
            sys.exit(1)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()