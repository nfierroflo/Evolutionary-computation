import os

os.system("python -m examples.symregv3 --experiment_name dumps/expcompgram3/RRMSE/freevars/max --seed 791025 --parameters parameters/standardv3.yml --run 1")
os.system("python -m examples.symregv3 --experiment_name dumps/expcompgram3/RRMSE/freevars/max --seed 791035 --parameters parameters/standardv3.yml --run 2")
os.system("python -m examples.symregv3 --experiment_name dumps/expcompgram3/RRMSE/freevars/max --seed 791045 --parameters parameters/standardv3.yml --run 3")