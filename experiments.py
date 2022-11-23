import os

os.system("python -m examples.symregv3 --experiment_name dumps/BasicGrammar-t0/RMSE  --seed 791025 --parameters parameters/standardv3.yml --run 1")
os.system("python -m examples.symregv3 --experiment_name dumps/BasicGrammar-t0/RMSE  --seed 791135 --parameters parameters/standardv3.yml --run 2")
os.system("python -m examples.symregv3 --experiment_name dumps/BasicGrammar-t0/RMSE  --seed 791155 --parameters parameters/standardv3.yml --run 3")