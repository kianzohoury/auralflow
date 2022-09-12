# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import auralflow.__main__
import sys


if __name__ == "__main__":
    
    # manually hack arguments for now
    file_path = sys.argv.pop(0)
    
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--train":
            break
        i += 1    
    
    # config model
    sys.argv = [file_path] + sys.argv[:i]
    
    print(sys.argv)
    auralflow.__main__.main()
    
    # train model
    sys.argv = [file_path] + sys.argv[i:]
    auralflow.__main__.main()
