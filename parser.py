# parser.py
# recursive-descent parser for simplified rat25s
# has semantic actions to build ST and code gen

import sys
from typing import List
import lexer
import symbol_table as st
import code_generator as cg

TRACE = False
tokens: List[lexer.Token] = []
pos = 0
error_count = 0

def current() -> lexer.Token:
    # peek at current token
    if pos >= len(tokens):
        return lexer.Token("eof","", tokens[-1].line_number if tokens else 1)
    return tokens[pos]

def advance() -> None:
    # move to next token
    global pos
    pos += 1

def expect(kind: str, lexeme: str=None) -> None:
    # consume expected token or error
    tok = current()
    if tok.kind != kind or (lexeme and tok.lexeme != lexeme):
        error(f"expected {kind}{(' '+lexeme) if lexeme else ''}")
    else:
        advance()

def error(msg: str) -> None:
    # simple syntax error handling
    global error_count
    tok = current()
    print(f"[syntax-error] line {tok.line_number}: at '{tok.lexeme}' — {msg}", file=sys.stderr)
    error_count += 1
    advance()

def prod(rule: str) -> None:
    # optional trace of prods
    if TRACE:
        tok = current()
        print(f"[prod] line {tok.line_number}: {rule}", file=sys.stderr)

# --------------- grammar rules with semantic actions --------------- #

def Rat25S() -> None:
    # entry point
    expect("separator","$$")
    prod("<Rat25S> -> $$ <OptDeclList> $$ <StatementList> $$")
    OptDeclList()
    expect("separator","$$")
    StatementList()
    expect("separator","$$")
    expect("eof")

def OptDeclList() -> None:
    # either list of decls or empty
    if current().lexeme in {"integer","boolean"}:
        prod("<OptDeclList> -> <DeclarationList>")
        DeclarationList()
    else:
        prod("<OptDeclList> -> ε")

def DeclarationList() -> None:
    prod("<DeclarationList> -> <Declaration> ; <DeclarationListPrime>")
    Declaration()
    expect("separator",";")
    DeclarationListPrime()

def DeclarationListPrime() -> None:
    if current().lexeme in {"integer","boolean"}:
        prod("<DeclarationListPrime> -> <DeclarationList>")
        DeclarationList()
    else:
        prod("<DeclarationListPrime> -> ε")

def Declaration() -> None:
    # parse qualifier then id list
    prod("<Declaration> -> <Qualifier> <IDs>")
    qual = current().lexeme
    Qualifier()
    collect_IDs(qual)

def collect_IDs(qual: str) -> None:
    # for each id, do st.insert
    if current().kind != "identifier":
        error("identifier expected")
    while True:
        name = current().lexeme
        st.insert(name, qual)  # semantic: add to ST
        advance()
        if current().lexeme != ",":
            break
        advance()  # skip ','

def Qualifier() -> None:
    prod("<Qualifier> -> integer|boolean")
    if current().lexeme in {"integer","boolean"}:
        advance()
    else:
        error("qualifier expected")

def StatementList() -> None:
    prod("<StatementList> -> <Statement> <StatementListPrime>")
    Statement()
    StatementListPrime()

def StatementListPrime() -> None:
    if current().lexeme in {"identifier","if","while","print","scan","{"}:
        prod("<StatementListPrime> -> <Statement> <StatementListPrime>")
        Statement()
        StatementListPrime()
    else:
        prod("<StatementListPrime> -> ε")

def Statement() -> None:
    # dispatch by lookahead
    lex = current().lexeme
    if current().kind=="identifier":
        Assign()
    elif lex=="if":
        If()
    elif lex=="while":
        While()
    elif lex=="print":
        Print()
    elif lex=="scan":
        Scan()
    elif lex=="{":
        Compound()
    else:
        error("statement expected")

def Assign() -> None:
    # id = expr ;
    prod("<Assign> -> id = <Expression> ;")
    name = current().lexeme
    if st.lookup(name) == -1:
        error(f"undeclared identifier '{name}'")
    advance()
    expect("operator","=")
    Expression()
    cg.gen("POPM", st.getAddr(name))  # semantic: store result
    expect("separator",";")

def If() -> None:
    # if ( cond ) stmt endif
    prod("<If> -> if ( <Condition> ) <Statement> endif")
    expect("keyword","if")
    expect("separator","(")
    Condition()
    expect("separator",")")
    Statement()
    cg.back_patch(cg.InstrAddr)      # backpatch the jmp0
    expect("keyword","endif")

def While() -> None:
    # while ( cond ) stmt endwhile
    prod("<While> -> while ( <Condition> ) <Statement> endwhile")
    expect("keyword","while")
    loop_start = cg.InstrAddr
    cg.gen("LABEL",-1)                # mark loop start
    expect("separator","(")
    Condition()
    expect("separator",")")
    Statement()
    cg.gen("JMP", loop_start)        # jump back to start
    cg.back_patch(cg.InstrAddr)      # patch exit jmp0
    expect("keyword","endwhile")

def Print() -> None:
    # print(expr);
    prod("<Print> -> print ( <Expression> ) ;")
    expect("keyword","print")
    expect("separator","(")
    Expression()
    cg.gen("SOUT",-1)                 # semantic: output top of stack
    expect("separator",")")
    expect("separator",";")

def Scan() -> None:
    # scan(id, id, ...);
    prod("<Scan> -> scan ( <IDs> ) ;")
    expect("keyword","scan")
    expect("separator","(")
    if current().kind != "identifier":
        error("identifier expected")
    while True:
        name = current().lexeme
        if st.lookup(name) == -1:
            error(f"undeclared identifier '{name}'")
        cg.gen("SIN",-1)             # semantic: read into stack
        cg.gen("POPM", st.getAddr(name))
        advance()
        if current().lexeme != ",":
            break
        advance()
    expect("separator",")")
    expect("separator",";")

def Compound() -> None:
    # { stmt stmt ... }
    prod("<Compound> -> { <StatementList> }")
    expect("separator","{")
    StatementList()
    expect("separator","}")

def Condition() -> None:
    # expr relop expr
    prod("<Condition> -> <Expression> <Relop> <Expression>")
    Expression()
    op = current().lexeme
    Relop()
    Expression()
    mapping = {"<":"LES",">":"GRT","==":"EQU","!=":"NEQ","<=":"LEQ","=>":"GEQ"}
    cg.gen(mapping[op], -1)           # semantic: compare
    cg.push_JMPstack(cg.InstrAddr)    # remember where jmp0 is
    cg.gen("JMP0",-1)                 # jump if false

def Relop() -> None:
    # one of == != < > <= =>
    prod("<Relop> -> ==|!=|>|<|<=|=>")
    if current().kind=="operator" and current().lexeme in {"==","!=","<",">","<=","=>"}:
        advance()
    else:
        error("relational operator expected")

def Expression() -> None:
    # expr -> term expr'
    prod("<Expression> -> <Term> <ExpressionPrime>")
    Term(); ExpressionPrime()

def ExpressionPrime() -> None:
    # handles + and -
    if current().kind=="operator" and current().lexeme in {"+","-"}:
        op = current().lexeme
        prod(f"<ExpressionPrime> -> {op} <Term> <ExpressionPrime>")
        advance(); Term()
        cg.gen("A" if op=="+" else "S", -1)  # add or sub
        ExpressionPrime()
    else:
        prod("<ExpressionPrime> -> ε")

def Term() -> None:
    prod("<Term> -> <Factor> <TermPrime>")
    Factor(); TermPrime()

def TermPrime() -> None:
    # handles * and /
    if current().kind=="operator" and current().lexeme in {"*","/"}:
        op = current().lexeme
        prod(f"<TermPrime> -> {op} <Factor> <TermPrime>")
        advance(); Factor()
        cg.gen("M" if op=="*" else "D", -1)  # mul or div
        TermPrime()
    else:
        prod("<TermPrime> -> ε")

def Factor() -> None:
    # unary minus or primary
    if current().lexeme == "-":
        prod("<Factor> -> - <Primary>")
        advance(); Primary()
    else:
        prod("<Factor> -> <Primary>")
        Primary()

def Primary() -> None:
    tok = current()
    if tok.kind=="identifier":
        prod("<Primary> -> IDENTIFIER")
        if st.lookup(tok.lexeme) == -1:
            error(f"undeclared identifier '{tok.lexeme}'")
        cg.gen("PUSHM", st.getAddr(tok.lexeme))  # push var
        advance()
    elif tok.kind=="integer":
        prod("<Primary> -> INTEGER")
        cg.gen("PUSHI", int(tok.lexeme))         # push literal
        advance()
    elif tok.kind=="keyword" and tok.lexeme in {"true","false"}:
        prod("<Primary> -> true|false")
        val = 1 if tok.lexeme=="true" else 0
        cg.gen("PUSHI", val)                     # bool literal
        advance()
    elif tok.lexeme == "(":
        prod("<Primary> -> ( <Expression> )")
        expect("separator","(")
        Expression()
        expect("separator",")")
    else:
        error("invalid primary")

def parse(src: str) -> None:
    # main entry for parser
    global tokens, pos, error_count
    tokens = lexer.tokenize(src)
    pos = 0; error_count = 0
    Rat25S()
    if error_count > 0:
        print(f"completed with {error_count} errors")
    else:
        cg.dumpInstrs()
        st.dumpST()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python parser.py <source_file.rat>")
        sys.exit(1)
    parse(open(sys.argv[1]).read())
