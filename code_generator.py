# code_generator.py
# builds list of instructions (instr table) and handles backpatch

import sys
from collections import deque

class Instr:
    def __init__(self, addr: int, op: str, oprnd: int):
        self.addr   = addr   # label/index
        self.op     = op     # opcode str
        self.oprnd  = oprnd  # operand or -1 for nil

I_TABLE   = 1000
InstrTable: list[Instr] = []
InstrAddr  = 1         # next instr label
JMPstack   = deque()   # stack for backpatch addresses

def gen(opcode: str, operand: int) -> None:
    # emit new instr into table
    global InstrAddr
    if InstrAddr > I_TABLE:
        print("[semantic-error] InstrTable overflow", file=sys.stderr)
        sys.exit(1)
    InstrTable.append(Instr(InstrAddr, opcode, operand))
    InstrAddr += 1

def push_JMPstack(addr: int) -> None:
    # push addr of jmp0 so we can patch later
    JMPstack.append(addr)

def back_patch(target: int) -> None:
    # pops one jmp0 addr and sets its operand to target
    if not JMPstack:
        print("[semantic-error] backpatch stack underflow", file=sys.stderr)
        sys.exit(1)
    addr = JMPstack.pop()
    InstrTable[addr-1].oprnd = target

def dumpInstrs() -> None:
    # print all instrs, skip nil oprnds
    for instr in InstrTable:
        oprnd = "" if instr.oprnd == -1 else str(instr.oprnd)
        print(f"{instr.addr:<5d}{instr.op:<8s}{oprnd}")
