# lexer.py
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional

# ─── Token-set definitions ──────────────────────────────────────────── #
KEYWORDS = {
    "integer", "real", "boolean",
    "function", "if", "else", "endif",
    "while", "endwhile", "return",
    "scan", "print", "true", "false",
}

# multi-char must be first so "<=" isn't split
MULTI_CHAR_OPERATORS = {"<=", ">=", "==", "<>", "!=", "=>"}  # added "=>"
MULTI_CHAR_SEPARATORS= {"$$"}

SINGLE_CHAR_OPERATORS = {"=", "+", "-", "*", "/", "<", ">"}
SINGLE_CHAR_SEPARATORS= {"(", ")", "{", "}", ";", ",", "$"}

@dataclass
class Token:
    kind: str
    lexeme: str
    line_number: int = 1
    def __str__(self) -> str:
        return f"{self.kind:15s} {self.lexeme:15s} {self.line_number}"

def is_keyword(lex: str) -> bool:
    return lex.lower() in KEYWORDS

def skip_ws_and_comments(src: str, i: int) -> Tuple[int, Optional[Token]]:
    n = len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1; continue
        # comment start "[*"
        if ch == '[' and i+1<n and src[i+1]=='*':
            i += 2
            while i+1<n and not (src[i]=='*' and src[i+1]==']'):
                i += 1
            if i+1>=n:
                return i, Token("unknown","unterminated comment")
            i += 2
            continue
        break
    return i, None

def lex_token(src: str, start: int) -> Tuple[Token,int]:
    i, err = skip_ws_and_comments(src, start)
    if err:
        return err, i
    if i>=len(src):
        return Token("eof",""), i
    ch = src[i]
    # identifier/keyword
    if ch.isalpha():
        b=i; i+=1
        while i<len(src) and (src[i].isalnum() or src[i]=='_'):
            i+=1
        lex = src[b:i]
        return Token("keyword" if is_keyword(lex) else "identifier", lex), i
    # integer/real
    if ch.isdigit():
        b=i; i+=1
        while i<len(src) and src[i].isdigit(): i+=1
        is_real=False
        if i<len(src) and src[i]=='.' and i+1<len(src) and src[i+1].isdigit():
            is_real=True; i+=1
            while i<len(src) and src[i].isdigit(): i+=1
        lex=src[b:i]
        return Token("real" if is_real else "integer", lex), i
    # multi-char ops/seps
    for table,kind in ((MULTI_CHAR_OPERATORS,"operator"),
                       (MULTI_CHAR_SEPARATORS,"separator")):
        for lit in table:
            if src.startswith(lit,i):
                return Token(kind,lit), i+len(lit)
    # single-char ops/seps
    if ch in SINGLE_CHAR_OPERATORS:
        return Token("operator",ch), i+1
    if ch in SINGLE_CHAR_SEPARATORS:
        return Token("separator",ch), i+1
    # unknown
    b=i
    while i<len(src) and not src[i].isspace() and \
          not src.startswith("[*",i) and \
          not any(src.startswith(m,i) for m in MULTI_CHAR_OPERATORS|MULTI_CHAR_SEPARATORS) and \
          src[i] not in SINGLE_CHAR_OPERATORS|SINGLE_CHAR_SEPARATORS:
        i+=1
    return Token("unknown",src[b:max(i,b+1)]), i

def tokenize(text: str) -> List[Token]:
    tokens=[]; i=0; line=1
    while i<len(text):
        tok,i2 = None, i
        i2, err = skip_ws_and_comments(text,i)
        if err:
            err.line_number=line
            tokens.append(err)
            i=i2; continue
        if i2>=len(text): break
        tok,i3 = lex_token(text,i)
        tok.line_number = line
        tokens.append(tok)
        line += text[i:i3].count('\n')
        i = i3
    tokens.append(Token("eof","",line))
    return tokens

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python lexer.py <source_file>")
        sys.exit(1)
    src = open(sys.argv[1]).read()
    for t in tokenize(src):
        print(t)
