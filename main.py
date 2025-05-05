# main.py
# simple driver: read rat src file, run parser

import sys
from parser import parse

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python main.py <source_file.rat>")
        sys.exit(1)
    src = open(sys.argv[1]).read()
    parse(src)
