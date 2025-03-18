#!/usr/bin/env python
"""
Run script for the Deforestation Analyser application.
"""
import os
import subprocess
import sys

def main():
    # Get the directory of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Change to the script directory
    os.chdir(dir_path)
    
    # Run Streamlit
    print("Starting Deforestation Analyser...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "deforestation_ui.py"])

if __name__ == "__main__":
    main() 