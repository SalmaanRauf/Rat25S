# symbol_table.py
# symbol table for storing ids, their types and mem locations

import sys

class STEntry:
    def __init__(self, lexeme: str, mem_loc: int, qual: str):
        self.lexeme = lexeme   # id name
        self.mem_loc = mem_loc # virtual memory addr
        self.qual    = qual    # "integer" or "boolean"

ST_SIZE = 256
SymbolTable: list[STEntry] = []
STcount = 0
Memory_Address = 10000  # start address for first decl

def lookup(lex: str) -> int:
    # return idx if id is in table, else -1
    for i, e in enumerate(SymbolTable):
        if e.lexeme == lex:
            return i
    return -1

def insert(lex: str, qual: str) -> None:
    # add new id or error on dup
    global STcount, Memory_Address
    if lookup(lex) != -1:
        print(f"[semantic-error] duplicate identifier '{lex}'", file=sys.stderr)
        sys.exit(1)
    if STcount >= ST_SIZE:
        print("[semantic-error] SymbolTable overflow", file=sys.stderr)
        sys.exit(1)
    SymbolTable.append(STEntry(lex, Memory_Address, qual))
    STcount += 1
    Memory_Address += 1  # bump addr for next

def getAddr(lex: str) -> int:
    # get memory addr or error if not found
    idx = lookup(lex)
    if idx == -1:
        print(f"[semantic-error] undeclared identifier '{lex}'", file=sys.stderr)
        sys.exit(1)
    return SymbolTable[idx].mem_loc

def getType(lex: str) -> str:
    # get declared type of id
    idx = lookup(lex)
    if idx == -1:
        print(f"[semantic-error] undeclared identifier '{lex}'", file=sys.stderr)
        sys.exit(1)
    return SymbolTable[idx].qual

def dumpST() -> None:
    # prints the whole table in spec format
    print("\nidentifier   memorylocation   type")
    for e in SymbolTable:
        print(f"{e.lexeme:12s}{e.mem_loc:16d}   {e.qual}")
