import os
import sys

for etap in ["etap1.py", "etap2.py", "etap3.py", "etap4.py", "etap5.py", "etap6.py"]:
    print("=====", etap)
    os.system('"' + sys.executable + '" ' + etap)
