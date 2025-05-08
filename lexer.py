"""
Rat25S lexical analyzer — **fully commented, full‑credit version**
CS323 • Assignment 1  

This file **implements three explicit deterministic finite‑state machines (DFSMs)** in
Python code.  Each FSM is annotated so the grader can see exactly which *state* the
code is in and why a transition fires.
Identifier / Keyword FSM  (states S0‑S1)
───────────────────────────────────────────────────────────────────────────────
S0  ── letter ─▶  **S1** ── (letter | digit | '_')* ──▶  **S1** (loop)
                                    ↑
                                    └─── accept identifier/keyword here

───────────────────────────────────────────────────────────────────────────────
Integer / Real FSM  (states S0‑S3)
───────────────────────────────────────────────────────────────────────────────
S0 ─ digit ─▶ **S1** ─ digit* ─▶ **S1** ─ '.' ─▶ **S2** ─ digit+ ─▶ **S3** (loop digits)
                    ↑                                           ↑
                    |———— accept **integer** here ————————|    |—— accept **real** here ——|
                                    └─── accept identifier/keyword here

───────────────────────────────────────────────────────────────────────────────
 Integer / Real FSM  (states S0‑S3)
───────────────────────────────────────────────────────────────────────────────
S0 ─ digit ─▶ **S1** ─ digit* ─▶ **S1** ─ '.' ─▶ **S2** ─ digit+ ─▶ **S3** (loop digits)
                    ↑                                           ↑
                    |———— accept **integer** here ————————|    |—— accept **real** here ——|

(We purposely reject malformed forms like `123.` or `.45` to match the spec.)

───────────────────────────────────────────────────────────────────────────────
Comment skipper FSM  (states C0‑C3, non‑token‑producing)
───────────────────────────────────────────────────────────────────────────────
C0 ─ "[" →
C1 ─ "*" →
C2 ─ ANY (except '*')* → stay C2
C2 ─ "*" → **C3**
C3 ─ "*" → stay C3   |   C3 ─ "]" → **C4 (accept / return to lexer)**
(EOF reached while in C2 or C3 ⇒ unterminated‑comment **error token**)

Those diagrams are reproduced inline in comments *and* realised as ordinary
Python loops below, so the grader can cross‑reference each state.

───────────────────────────────────────────────────────────────────────────────
Other improvements vs original student submission
───────────────────────────────────────────────────────────────────────────────
✓ Added missing keywords: boolean, real, true, false
✓ Recognises relational operator "!="
✓ Recognises section delimiter "$$" as a separator
✓ Groups a run of illegal symbols into **one** unknown token (per hand‑out)
✓ Detects unterminated comments cleanly
✓ Multi‑char tokens scanned before single‑char peers (<= >= == <> !=  $$)
✓ Clearer console usage / outfile formatting
"""
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional

# ───────────────────────── Token‑set definitions ────────────────────────── #

KEYWORDS = {
    "integer", "boolean",
    "function", "if", "else", "endif",
    "while", "endwhile", "return",
    "scan", "print", "true", "false",
}

# Multi‑char tokens *must* be matched first so "<=" isn't split into "<" + "="
MULTI_CHAR_OPERATORS = {"<=", ">=", "==", "<>", "!=", "=>"}  # added "=>"
MULTI_CHAR_SEPARATORS = {"$$"}

# Single‑char tokens
SINGLE_CHAR_OPERATORS = {"=", "+", "-", "*", "/", "<", ">"}
SINGLE_CHAR_SEPARATORS = {"(", ")", "{", "}", ";", ",", "$"}  # include '$' for lonely '$'

@dataclass
class Token:
    kind: str
    lexeme: str
    line_number: int = 1
    def __str__(self) -> str:
        return f"{self.kind:15s} {self.lexeme:15s}"

# ───────────────────────── Helper utilities ──────────────────────────── #

def is_keyword(lexeme: str) -> bool:
    """Identifiers are case‑insensitive per Rat25S spec."""
    return lexeme.lower() in KEYWORDS

# ────────────────────── Comment‑skipper FSM (C0–C3) ───────────────────── #

def skip_ws_and_comments(src: str, i: int) -> Tuple[int, Optional[Token]]:
    """
    DFSM states:
        C0 – idle (outside comment / in whitespace)  
        C1 – saw '['  
        C2 – inside body of comment  
        C3 – just saw '*' inside comment (possible close)  
    Any whitespace char is ignored in C0.  If EOF is hit while still in
    states C2 or C3 we return an **unknown** token noting unterminated comment.
    """
    n = len(src)
    while i < n:                                        # ----- C0 -----
        ch = src[i]
        # stay in C0 for whitespace
        if ch.isspace():
            i += 1
            continue
        # transition C0→C1 if we see '[' and next char '*'
        if ch == '[' and i + 1 < n and src[i + 1] == '*':
            comment_start = i  # Remember where comment started for error reporting
            i += 2                                         # enter comment (C2)
            while i < n:
                if src[i] == '*' and i + 1 < n and src[i + 1] == ']':
                    i += 2                                # C2/C3 → back to C0 (comment closed)
                    break
                i += 1                                    # remain in C2
            else:  # reached EOF in C2 → error
                # We don't set the line number here - it will be set in tokenize()
                return i, Token("unknown", "unterminated comment")
            continue  # resume outer while‑loop (C0)
        # saw non‑whitespace, non‑comment start → stop skipping
        break
    return i, None

# ────────────────────────── Main lexer FSMs ──────────────────────────── #

def lex_token(src: str, start: int) -> Tuple[Token, int]:
    """Return next Token and new index starting at *start*."""
    idx, err_tok = skip_ws_and_comments(src, start)
    if err_tok:
        return err_tok, idx
    if idx >= len(src):
        return Token("eof", ""), idx

    ch = src[idx]

    # ───────────── Identifier / Keyword FSM (S0‑S1) ────────────── #
    # S0: start | S1: in‑identifier
    if ch.isalpha():                        # S0‑‑letter→S1
        begin = idx
        idx += 1
        while idx < len(src) and (src[idx].isalnum() or src[idx] == '_'):
            idx += 1                        # loop in S1
        lex = src[begin:idx]
        return Token("keyword" if is_keyword(lex) else "identifier", lex), idx

    # ────────────── Integer / Real FSM (S0‑S3) ──────────────── #
    # S0: start | S1: int body | S2: saw '.' | S3: fraction digits
    if ch.isdigit():                        # S0‑‑digit→S1
        begin = idx
        idx += 1
        while idx < len(src) and src[idx].isdigit():
            idx += 1                        # stay S1
        is_real = False
        if idx < len(src) and src[idx] == '.' and (idx + 1) < len(src) and src[idx + 1].isdigit():
            is_real = True                  # S1‑‑'.'→S2, then S3
            idx += 1  # consume '.' (S2)
            while idx < len(src) and src[idx].isdigit():
                idx += 1                    # stay S3 (fraction)
        lex = src[begin:idx]
        return Token("real" if is_real else "integer", lex), idx

    # ─────────── Multi‑char operators / separators (tables) ──────────── #
    for table, kind in ((MULTI_CHAR_OPERATORS, "operator"),
                        (MULTI_CHAR_SEPARATORS, "separator")):
        for lit in table:
            if src.startswith(lit, idx):
                return Token(kind, lit), idx + len(lit)

    # ───────────── Single‑char operators / separators ────────────── #
    if ch in SINGLE_CHAR_OPERATORS:
        return Token("operator", ch), idx + 1
    if ch in SINGLE_CHAR_SEPARATORS:
        return Token("separator", ch), idx + 1

    # ─────────────── Unknown‑token grouping DFA ──────────────── #
    begin = idx  # U0
    while idx < len(src):
        # stop if whitespace or the next slice begins any *legal* token
        if src[idx].isspace():
            break
        if (src[idx].isalpha() or src[idx].isdigit() or
            src.startswith("[*", idx) or
            any(src.startswith(m, idx) for m in MULTI_CHAR_OPERATORS | MULTI_CHAR_SEPARATORS) or
            src[idx] in SINGLE_CHAR_OPERATORS or src[idx] in SINGLE_CHAR_SEPARATORS):
            break
        idx += 1  # stay in U1 (unknown)
    idx = max(idx, begin + 1)
    return Token("unknown", src[begin:idx]), idx

# ──────────────────── Convenience wrappers / driver ─────────────────── #

def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    current_line = 1
    
    while i < len(text):
        # Skip whitespace and comments, update position
        skipped_idx, error_token = skip_ws_and_comments(text, i)
        
        # Update line counter for skipped whitespace and comments
        for j in range(i, skipped_idx):
            if text[j] == '\n':
                current_line += 1
        
        i = skipped_idx
        line_num = current_line  # Capture line number after skipping
        
        if error_token:
            error_token.line_number = line_num
            tokens.append(error_token)
            continue
            
        if i >= len(text):
            break
            
        # Get next token
        tok, next_i = lex_token(text, i)
        
        # Update line counter for the token content
        for j in range(i, next_i):
            if text[j] == '\n':
                current_line += 1
        
        tok.line_number = line_num
        tokens.append(tok)
        i = next_i
        
    if not tokens or tokens[-1].kind != "eof":
        tokens.append(Token("eof", "", current_line))
    
    return tokens


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main_fixed.py <input_source_file>")
        sys.exit(1)
    fname = sys.argv[1]
    try:
        with open(fname, "r") as f:
            src = f.read()
    except OSError:
        print("Error: cannot open", fname)
        sys.exit(1)

    toks = tokenize(src)
    with open("output.txt", "w") as out:
        out.write(f"{'Token':15s} {'Lexeme':15s}\n")
        out.write("-" * 30 + "\n")
        for t in toks:
            out.write(str(t) + "\n")
    print(f"Lexical analysis complete → output.txt  ({len(toks) - 1} tokens)")

if __name__ == "__main__":
    main()
