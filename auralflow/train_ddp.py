# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import auralflow.__main__ as main_script
import sys
import os
from pathlib import Path
import importlib

importlib.reload(main_script)

if __name__ == "__main__":
    
    # manually hack arguments for now
    file_path = str(Path(Path(__file__).absolute().parent, '__main__.py'))
    print(file_path)
    
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--train":
            break
        i += 1 
        
    sys.argv = ['/home/ec2-user/SageMaker/auralflow/auralflow/__main__.py'] + sys.argv[1:i]
    
    # config model
#     os.system(f"python3 {file_path} " + " ".join(sys.argv[1:i]))
#     sys.argv = [file_path] + sys.argv[:i]
    
#     print(sys.argv)
#     main_script.main()
    
#     # train model
#     sys.argv = [file_path] + sys.argv[i:]
#     auralflow.__main__.main()
