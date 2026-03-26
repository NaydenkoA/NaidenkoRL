#!/bin/bash
set -e
if [ -f ~/sb3env/bin/activate ]; then
    source ~/sb3env/bin/activate
else
    echo "Error: Virtual environment not found at ~/sb3env/bin/activate"
    exit 1
fi
python train_model_variable_leverage_recurrent_v4.py
python train_model_variable_leverage_recurrent_v4_phase_1.py
python train_model_variable_leverage_recurrent_v4_phase_2.py
echo "✅ All training phases completed."