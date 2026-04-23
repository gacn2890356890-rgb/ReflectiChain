import sys
sys.path.insert(0, r"c:\Users\Administrator\Desktop\cv")
from src.experiments.ma_experiments import main as ma_main
from src.experiments.adv_experiments import main as adv_main
ma_main()
adv_main()
