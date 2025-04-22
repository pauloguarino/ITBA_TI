"""Template script for testing modules from ti_modules"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import ti_modules

if 'first_run' not in globals():
    print(f"Script for testing {ti_modules.__name__}")
    modules = [ti_modules.__name__]
    first_run = modules

# Run the following line on your python virtual environmen to execute the script
# exec(open("tests_script.py").read())

# Script starts here, variables will be local to your workspace
# Restart python if any of the ti_modules being used was changed
