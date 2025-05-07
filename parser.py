from __future__ import annotations
from typing import List, Optional
import sys
import lexer

TRACE: bool = True
OUTFILE = None

class SymbolTable:
    def __init__(self):
        self.table = {}
        self.memory_address = 10000  # Starting memory address as specified
    
    def lookup(self, lexeme):
        """Check if identifier exists in the symbol table"""
        return lexeme in self.table
    
    def insert(self, lexeme, type_name):
        """Insert a new identifier into the symbol table"""
        if self.lookup(lexeme):
            return False  # Already exists
        
        self.table[lexeme] = {
            'memory_address': self.memory_address,
            'type': type_name
        }
        self.memory_address += 1
        return True
    
    def get_address(self, lexeme):
        """Get memory address of an identifier"""
        if not self.lookup(lexeme):
            return None
        return self.table[lexeme]['memory_address']
    
    def get_type(self, lexeme):
        """Get type of an identifier"""
        if not self.lookup(lexeme):
            return None
        return self.table[lexeme]['type']
    
    def print_table(self):
        """Print the symbol table"""
        print("\nSymbol Table")
        print("=" * 50)
        print(f"{'Identifier':<15} {'MemoryLocation':<15} {'Type':<10}")
        print("-" * 50)
        for lexeme, info in self.table.items():
            print(f"{lexeme:<15} {info['memory_address']:<15} {info['type']:<10}")
        print("=" * 50)

class AssemblyGenerator:
    def __init__(self):
        self.instructions = []
        self.instr_address = 1  # Instructions start from 1
        self.jump_stack = []    # Stack for storing jump addresses for backpatching
    
    def generate(self, op, oprnd=None):
        """Generate an assembly instruction"""
        self.instructions.append({
            'address': self.instr_address,
            'op': op,
            'oprnd': oprnd
        })
        self.instr_address += 1
        return self.instr_address - 1  # Return the address of the instruction just added
    
    def push_jump_stack(self, address):
        """Push an address onto the jump stack for later backpatching"""
        self.jump_stack.append(address)
    
    def pop_jump_stack(self):
        """Pop an address from the jump stack"""
        if not self.jump_stack:
            return None
        return self.jump_stack.pop()
    
    def back_patch(self, jump_address):
        """Backpatch a jump instruction"""
        addr = self.pop_jump_stack()
        if addr is not None:
            self.instructions[addr-1]['oprnd'] = jump_address
    
    def print_assembly(self):
        """Print the generated assembly code"""
        print("\nAssembly Code Listing")
        print("=" * 50)
        for instr in self.instructions:
            if instr['oprnd'] is not None:
                print(f"{instr['address']:<5} {instr['op']:<10} {instr['oprnd']}")
            else:
                print(f"{instr['address']:<5} {instr['op']}")
        print("=" * 50)

# Global variables for parser
tokens: List[lexer.Token] = []
pos: int = 0
error_count: int = 0  # Track the total number of errors found
symbol_table = SymbolTable()
assembly = AssemblyGenerator()
current_qualifier = None  # Track the current qualifier for declarations

def semantic_error(msg: str) -> None:
    tok = current()
    print(f"[semantic-error] line {tok.line_number}: at token '{tok.lexeme}' ({tok.kind}) — {msg}", file=sys.stderr)
    if TRACE and OUTFILE:
        OUTFILE.write(f"[semantic-error] line {tok.line_number}: at token '{tok.lexeme}' ({tok.kind}) — {msg}\n")
    sys.exit(1)

def current() -> lexer.Token:
    global pos
    # Make sure pos is within valid range
    if pos >= len(tokens):
        # If we've consumed all real tokens, return a dummy EOF token
        # with the line number from the last real token
        last_line = tokens[-1].line_number if tokens else 1
        return lexer.Token("eof", "", last_line)
    return tokens[pos]

def advance() -> None:
    """
    Consume the current token.  *Must* be called only after ensuring it matches
    the expected grammar symbol.
    """
    global pos
    
    # print token / lexeme BEFORE advancing, per spec
    # Only print if this is not an EOF token
    if OUTFILE and current().kind != "eof":
        curr_token = current()
        OUTFILE.write(f"Token: {curr_token.kind:15s} Lexeme: {curr_token.lexeme:15s} Line: {curr_token.line_number}\n")
    
    # AFTER printing the token info, then advance the position
    pos += 1
    
    # Return early if we've moved to EOF to prevent multiple EOF prints
    # Note: This check happens AFTER advancing pos and AFTER printing the token
    if pos >= len(tokens):
        return

def expect(kind: str, lexeme: Optional[str] = None) -> None:
    tok = current()
    if tok.kind != kind or (lexeme is not None and tok.lexeme != lexeme):
        error(f"expected {kind}{' ' + lexeme if lexeme else ''}")
        # error() now calls advance(), so we don't need to do it again here
    else:
        # Only advance if the token matched
        advance()

def error(msg: str) -> None:
    global error_count
    tok = current()
    # Output to stderr with the [syntax-error] prefix
    print(f"[syntax-error] line {tok.line_number}: at token '{tok.lexeme}' ({tok.kind}) — {msg}", file=sys.stderr)
    
    # Also write to the OUTFILE if tracing is enabled
    if TRACE and OUTFILE:
        OUTFILE.write(f"[syntax-error] line {tok.line_number}: at token '{tok.lexeme}' ({tok.kind}) — {msg}\n")
    
    error_count += 1  # Increment the error count
    # Don't exit, so parsing can continue
    advance()  # Skip the problematic token to allow parsing to continue

def prod(rule: str) -> None:
    if TRACE and OUTFILE:
        line_num = current().line_number
        OUTFILE.write(f"\tline {line_num}: {rule}\n")

# Grammar procedures (after factoring) 

# R1  <Rat25S> ::= $$ <Opt Function Definitions> $$ <Opt Declaration List> $$ <Statement List> $$
def Rat25S() -> None:
    # First consume the $$ token - this will print the token
    expect("separator", "$$")
    
    # Then print the production rule
    prod("<Rat25S> -> $$ <OptFunctionDefinitions> $$ <OptDeclarationList> $$ <StatementList> $$")
    
    # Continue with the rest of the parsing
    OptFunctionDefinitions()
    expect("separator", "$$")
    OptDeclarationList()
    expect("separator", "$$")
    StatementList()
    expect("separator", "$$")
    
    # After consuming the final $$, check for EOF
    expect("eof")   # must reach end cleanly

# R2
def OptFunctionDefinitions() -> None:
    # For simplified Rat25S, we don't have function definitions
    prod("<OptFunctionDefinitions> -> ε")

# R10, R11, R12, R13
def OptDeclarationList() -> None:
    if current().lexeme in {"integer", "boolean"}:
        prod("<OptDeclarationList> -> <DeclarationList>")
        DeclarationList()
    else:
        prod("<OptDeclarationList> -> ε")

def DeclarationList() -> None:
    prod("<DeclarationList> -> <Declaration> ; <DeclarationListPrime>")
    Declaration()
    expect("separator", ";")
    DeclarationListPrime()
        
def DeclarationListPrime() -> None:
    if current().lexeme in {"integer", "boolean"}:
        prod("<DeclarationListPrime> -> <DeclarationList>")
        DeclarationList()
    else:
        prod("<DeclarationListPrime> -> ε")

def Declaration() -> None:
    global current_qualifier
    prod("<Declaration> -> <Qualifier> <IDs>")
    
    # Save the qualifier for use in IDs
    if current().lexeme in {"integer", "boolean"}:
        current_qualifier = current().lexeme
        Qualifier()
        IDs()
    else:
        error("qualifier (integer/boolean) expected")

def Qualifier() -> None:
    prod("<Qualifier> -> integer | boolean")
    if current().lexeme in {"integer", "boolean"}:
        advance()
    else:
        error("qualifier (integer/boolean) expected")

def IDs() -> None:
    global current_qualifier
    prod("<IDs> -> IDENTIFIER <IDsPrime>")
    
    if current().kind == "identifier":
        # Add to symbol table
        if not symbol_table.insert(current().lexeme, current_qualifier):
            semantic_error(f"Identifier '{current().lexeme}' already declared")
        
        advance()
        IDsPrime()
    else:
        error("identifier expected")

def IDsPrime() -> None:
    global current_qualifier
    if current().lexeme == ",":
        prod("<IDsPrime> -> , IDENTIFIER <IDsPrime>")
        expect("separator", ",")
        
        if current().kind == "identifier":
            # Add to symbol table
            if not symbol_table.insert(current().lexeme, current_qualifier):
                semantic_error(f"Identifier '{current().lexeme}' already declared")
            
            advance()
            IDsPrime()
        else:
            error("identifier expected")
    else:
        prod("<IDsPrime> -> ε")

# R14, R15
def StatementList() -> None:
    # Check if we're at a token that can't start a statement (including the final $$)
    tok = current()
    if ((tok.kind == "separator" and tok.lexeme in {"}", "$$"}) or 
        tok.kind == "eof"):
        # Empty statement list
        prod("<StatementList> -> ε")
        return
        
    prod("<StatementList> -> <Statement> <StatementListPrime>")
    Statement()
    StatementListPrime()

def StatementListPrime() -> None:
    # Check for more statements
    starters = {"separator": "{", "identifier": None, "keyword": {
        "if","while","return","scan","print"}}
    
    tok = current()
    # Explicitly check for tokens that can't start a statement
    if ((tok.kind == "separator" and tok.lexeme in {"}", "$$"}) or 
        tok.kind == "eof"):
        prod("<StatementListPrime> -> ε")
        return
        
    # Standard statement starters
    if ((tok.kind == "separator" and tok.lexeme == "{") or
        tok.kind == "identifier" or
        (tok.kind == "keyword" and tok.lexeme in starters["keyword"])):
        prod("<StatementListPrime> -> <Statement> <StatementListPrime>")
        Statement()
        StatementListPrime()

def Statement() -> None:
    tok = current()
    if tok.kind == "separator" and tok.lexeme == "{":
        prod("<Statement> -> <Compound>")
        Compound()
    elif tok.kind == "identifier":
        prod("<Statement> -> <Assign>")
        Assign()
    elif tok.kind == "keyword":
        if tok.lexeme == "if":
            prod("<Statement> -> <If>")
            If()
        elif tok.lexeme == "while":
            prod("<Statement> -> <While>")
            While()
        elif tok.lexeme == "print":
            prod("<Statement> -> <Print>")
            Print()
        elif tok.lexeme == "scan":
            prod("<Statement> -> <Scan>")
            Scan()
        else:
            error("unexpected keyword in statement")
    else:
        error("invalid statement start")

# R16
def Compound() -> None:
    prod("<Compound> -> { <StatementList> }")
    expect("separator", "{")
    StatementList()
    expect("separator", "}")

# R17
def Assign() -> None:
    prod("<Assign> -> IDENTIFIER = <Expression> ;")
    
    # Check if identifier is declared
    if not symbol_table.lookup(current().lexeme):
        semantic_error(f"Identifier '{current().lexeme}' used but not declared")
    
    # Save the identifier for later use
    id_lexeme = current().lexeme
    
    expect("identifier")
    expect("operator", "=")
    
    Expression()  # This will push the result onto the stack
    
    # Generate code to pop the result into the identifier's memory location
    assembly.generate("POPM", symbol_table.get_address(id_lexeme))
    
    expect("separator", ";")

# R18
def If() -> None:
    prod("<If> -> if ( <Condition> ) <Statement> <IfPrime>")
    expect("keyword", "if")
    expect("separator", "(")
    
    Condition()  # This will push 0 or 1 onto the stack
    
    # Generate conditional jump - if condition is false (0), jump to else or endif
    jump_addr = assembly.generate("JMP0", None)
    assembly.push_jump_stack(jump_addr)
    
    expect("separator", ")")
    Statement()
    IfPrime()
        
def IfPrime() -> None:
    if current().lexeme == "else":
        prod("<IfPrime> -> else <Statement> endif")
        
        # Generate jump to skip the else part
        else_jump = assembly.generate("JMP", None)
        
        # Backpatch the if-condition jump to jump to here (start of else)
        assembly.back_patch(assembly.instr_address)
        
        expect("keyword", "else")
        Statement()
        
        # Backpatch the jump at the end of the if-part to jump to here (end of else)
        assembly.instructions[else_jump-1]['oprnd'] = assembly.instr_address
        
        expect("keyword", "endif")
    else:
        # Handle the one-arm form
        prod("<IfPrime> -> endif")
        
        # Backpatch the if-condition jump to jump to here (end of if)
        assembly.back_patch(assembly.instr_address)
        
        expect("keyword", "endif")

# R21 (note duplicated number in spec for Print/Scan)
def Print() -> None:
    prod("<Print> -> print ( <Expression> ) ;")
    expect("keyword", "print")
    expect("separator", "(")
    
    Expression()  # This will push the value to print onto the stack
    
    # Generate code to output the value
    assembly.generate("SOUT")
    
    expect("separator", ")")
    expect("separator", ";")

def Scan() -> None:
    prod("<Scan> -> scan ( <IDs> ) ;")
    expect("keyword", "scan")
    expect("separator", "(")
    
    # Save the current position to process identifiers
    start_pos = pos
    
    # First pass: collect identifiers
    identifiers = []
    if current().kind == "identifier":
        # Check if identifier is declared
        if not symbol_table.lookup(current().lexeme):
            semantic_error(f"Identifier '{current().lexeme}' used but not declared")
        
        identifiers.append(current().lexeme)
        advance()
        
        while current().lexeme == ",":
            advance()
            if current().kind == "identifier":
                # Check if identifier is declared
                if not symbol_table.lookup(current().lexeme):
                    semantic_error(f"Identifier '{current().lexeme}' used but not declared")
                
                identifiers.append(current().lexeme)
                advance()
            else:
                error("identifier expected")
    
    # Generate code for each identifier
    for id_lexeme in identifiers:
        # Generate code to get input and store it
        assembly.generate("SIN")
        assembly.generate("POPM", symbol_table.get_address(id_lexeme))
    
    expect("separator", ")")
    expect("separator", ";")

# R22
def While() -> None:
    prod("<While> -> while ( <Condition> ) <Statement> endwhile")
    expect("keyword", "while")
    expect("separator", "(")
    
    # Generate a label for the start of the loop
    loop_start = assembly.instr_address
    assembly.generate("LABEL")
    
    Condition()  # This will push 0 or 1 onto the stack
    
    # Generate conditional jump - if condition is false (0), jump to end
    jump_addr = assembly.generate("JMP0", None)
    assembly.push_jump_stack(jump_addr)
    
    expect("separator", ")")
    Statement()
    
    # Generate jump back to the start of the loop
    assembly.generate("JMP", loop_start)
    
    # Backpatch the conditional jump to jump to here (end of loop)
    assembly.back_patch(assembly.instr_address)
    
    expect("keyword", "endwhile")

# R23
def Condition() -> None:
    prod("<Condition> -> <Expression> <Relop> <Expression>")
    Expression()
    
    relop = current().lexeme
    Relop()
    
    Expression()
    
    # Generate code for the comparison
    if relop == "==":
        assembly.generate("EQU")
    elif relop == "!=":
        assembly.generate("NEQ")
    elif relop == ">":
        assembly.generate("GRT")
    elif relop == "<":
        assembly.generate("LES")
    elif relop == ">=":
        assembly.generate("GEQ")
    elif relop == "<=":
        assembly.generate("LEQ")

# R24
def Relop() -> None:
    prod("<Relop> -> == | != | > | < | <= | =>")
    lex = current().lexeme
    if current().kind == "operator" and lex in {"==","!=","<",">","<=","=>"}:
        advance()
    else:
        error("relational operator expected")

# Expression, Term, Factor, Primary with left‑recursion removal
def Expression() -> None:
    prod("<Expression> -> <Term> <ExpressionPrime>")
    Term()
    ExpressionPrime()

def ExpressionPrime() -> None:
    if current().kind == "operator" and current().lexeme in {"+","-"}:
        op = current().lexeme
        prod("<ExpressionPrime> -> + <Term> <ExpressionPrime> | - <Term> <ExpressionPrime>")
        advance()
        Term()
        
        # Generate code for addition or subtraction
        if op == "+":
            assembly.generate("A")
        else:  # op == "-"
            assembly.generate("S")
            
        ExpressionPrime()
    else:
        prod("<ExpressionPrime> -> ε")

def Term() -> None:
    prod("<Term> -> <Factor> <TermPrime>")
    Factor()
    TermPrime()

def TermPrime() -> None:
    if current().kind == "operator" and current().lexeme in {"*","/"}:
        op = current().lexeme
        prod("<TermPrime> -> * <Factor> <TermPrime> | / <Factor> <TermPrime>")
        advance()
        Factor()
        
        # Generate code for multiplication or division
        if op == "*":
            assembly.generate("M")
        else:  # op == "/"
            assembly.generate("D")
            
        TermPrime()
    else:
        prod("<TermPrime> -> ε")

def Factor() -> None:
    if current().lexeme == "-":
        prod("<Factor> -> - <Primary>")
        advance()
        Primary()
        
        # Generate code for negation (multiply by -1)
        assembly.generate("PUSHI", -1)
        assembly.generate("M")
    else:
        prod("<Factor> -> <Primary>")
        Primary()

# R28 (Primary)
def Primary() -> None:
    tok = current()
    if tok.kind == "identifier":
        # Check if identifier is declared
        if not symbol_table.lookup(tok.lexeme):
            semantic_error(f"Identifier '{tok.lexeme}' used but not declared")
        
        # Generate code to push the value of the identifier onto the stack
        assembly.generate("PUSHM", symbol_table.get_address(tok.lexeme))
        
        prod("<Primary> -> IDENTIFIER")
        advance()
    elif tok.kind == "integer":
        # Generate code to push the integer value onto the stack
        assembly.generate("PUSHI", int(tok.lexeme))
        
        prod("<Primary> -> INTEGER")
        advance()
    elif tok.kind == "keyword" and tok.lexeme in {"true", "false"}:
        # Generate code to push 1 (true) or 0 (false) onto the stack
        if tok.lexeme == "true":
            assembly.generate("PUSHI", 1)
        else:  # tok.lexeme == "false"
            assembly.generate("PUSHI", 0)
            
        prod("<Primary> -> true | false")
        advance()
    elif tok.kind == "separator" and tok.lexeme == "(":
        prod("<Primary> -> ( <Expression> )")
        expect("separator", "(")
        Expression()
        expect("separator", ")")
    else:
        error("invalid primary")

# Driver entry point
def parse(src_text: str, outfile_name: str = "sa_output.txt") -> None:
    global tokens, pos, OUTFILE, error_count
    tokens = lexer.tokenize(src_text)
    pos = 0
    error_count = 0  # Reset error count
    
    # Open the output file for the entire parsing process
    with open(outfile_name, "w") as out_file:
        global OUTFILE
        OUTFILE = out_file
        OUTFILE.write(f"{'Token':15s} {'Lexeme':15s}\n")
        OUTFILE.write("-"*35 + "\n")
        
        # Parse the entire program
        Rat25S()
        
        # Write symbol table to output file
        OUTFILE.write("\nSymbol Table\n")
        OUTFILE.write("=" * 50 + "\n")
        OUTFILE.write(f"{'Identifier':<15} {'MemoryLocation':<15} {'Type':<10}\n")
        OUTFILE.write("-" * 50 + "\n")
        for lexeme, info in symbol_table.table.items():
            OUTFILE.write(f"{lexeme:<15} {info['memory_address']:<15} {info['type']:<10}\n")
        OUTFILE.write("=" * 50 + "\n")
        
        # Write assembly code to output file
        OUTFILE.write("\nAssembly Code Listing\n")
        OUTFILE.write("=" * 50 + "\n")
        for instr in assembly.instructions:
            if instr['oprnd'] is not None:
                OUTFILE.write(f"{instr['address']:<5} {instr['op']:<10} {instr['oprnd']}\n")
            else:
                OUTFILE.write(f"{instr['address']:<5} {instr['op']}\n")
        OUTFILE.write("=" * 50 + "\n")
        
        # Make sure OUTFILE is properly closed
        OUTFILE.flush()
    
    # Also print to console for convenience
    symbol_table.print_table()
    assembly.print_assembly()
    
    if error_count > 0:
        print(f"Parsing completed with {error_count} errors – output in {outfile_name}")
    else:
        print(f"Parsing completed successfully – output in {outfile_name}")

# Stand‑alone CLI
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parser.py <source_file>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        parse(f.read())
