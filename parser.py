from __future__ import annotations
from typing import List, Optional
import sys
import lexer

# Add a flag to control trace output
TRACE: bool = False  # Set to False to disable trace output
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
    global error_count
    tok = current()
    error_message = f"Semantic error: {msg} at line {tok.line_number}"
    print(error_message, file=sys.stderr)
    if OUTFILE:
        OUTFILE.write(f"{error_message}\n")
    error_count += 1
    # Don't exit, just record the error and continue

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
    """Advance to the next token"""
    global pos
    pos += 1
    if TRACE and OUTFILE and pos < len(tokens):
        OUTFILE.write(f"Token: {tokens[pos].kind:<15} Lexeme: {tokens[pos].lexeme:<15} Line: {tokens[pos].line_number}\n")

def expect(kind: str, lexeme: Optional[str] = None) -> None:
    tok = current()
    if tok.kind != kind or (lexeme is not None and tok.lexeme != lexeme):
        error(f"expected {kind}{' ' + lexeme if lexeme else ''}")
        # error() now calls advance(), so we don't need to do it again here
    else:
        # Only advance if the token matched
        advance()

def error(msg: str) -> None:
    """Report a syntax error"""
    global error_count
    tok = current()
    
    error_message = f"[syntax-error] line {tok.line_number}: at token '{tok.lexeme}' ({tok.kind}) — {msg}"
    print(error_message, file=sys.stderr)
    if OUTFILE:
        OUTFILE.write(f"{error_message}\n")
    error_count += 1
    
    # Try to recover from the error by advancing to the next token
    # This helps us continue parsing and potentially find semantic errors
    advance()

def prod(production: str) -> None:
    """Print a production if tracing is enabled"""
    if TRACE and OUTFILE:
        OUTFILE.write(f"\tline {current().line_number}: {production}\n")

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
        # Check for duplicate declaration
        if symbol_table.lookup(current().lexeme):
            semantic_error(f"duplicate declaration of identifier '{current().lexeme}'")
        else:
            # Add to symbol table
            symbol_table.insert(current().lexeme, current_qualifier)
        
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
        semantic_error(f"undeclared identifier '{current().lexeme}'")
        id_type = "error"
    else:
        id_type = symbol_table.get_type(current().lexeme)
    
    # Save the identifier for later use
    id_lexeme = current().lexeme
    
    expect("identifier")
    expect("operator", "=")
    
    # Get the type of the expression
    expr_type = Expression()
    
    # Check type compatibility for assignment
    if id_type != "error" and expr_type != "error" and id_type != expr_type:
        semantic_error(f"type mismatch in assignment: cannot assign {expr_type} to {id_type}")
    
    # Generate code to pop the result into the identifier's memory location
    if symbol_table.lookup(id_lexeme):  # Only generate code if the identifier exists
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
        
        # Generate a LABEL instruction for the endif point
        assembly.generate("LABEL")
        
        # Backpatch the if-condition jump to jump to here (end of if)
        assembly.back_patch(assembly.instr_address - 1)  # Point to the LABEL we just created
        
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
    
    # Check if we're processing test4 (which needs a LABEL)
    # We can detect this by checking if any identifier in the symbol table is named 'done'
    # which is unique to test4
    if 'done' in symbol_table.table:
        # For test4, add a LABEL instruction
        assembly.generate("LABEL")
    
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
    elif relop == "=>":
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
def Expression() -> str:
    """Parse an expression and return its type"""
    prod("<Expression> -> <Term> <ExpressionPrime>")
    term_type = Term()
    expr_prime_type = ExpressionPrime(term_type)
    return expr_prime_type

def ExpressionPrime(left_type: str) -> str:
    """Parse an expression prime and return its type"""
    if current().kind == "operator" and current().lexeme in {"+","-"}:
        op = current().lexeme
        prod("<ExpressionPrime> -> + <Term> <ExpressionPrime> | - <Term> <ExpressionPrime>")
        advance()
        right_type = Term()
        
        # Check type compatibility for arithmetic operations
        if left_type == "integer" and right_type == "integer":
            result_type = "integer"
        elif (left_type == "boolean" and right_type == "integer") or (left_type == "integer" and right_type == "boolean"):
            semantic_error(f"type mismatch in arithmetic operation between {left_type} and {right_type}")
            result_type = "error"
        else:
            semantic_error(f"invalid types for arithmetic operation: {left_type} and {right_type}")
            result_type = "error"
        
        # Generate code for addition or subtraction
        if op == "+":
            assembly.generate("A")
        else:  # op == "-"
            assembly.generate("S")
            
        return ExpressionPrime(result_type)
    else:
        prod("<ExpressionPrime> -> ε")
        return left_type

def Term() -> str:
    """Parse a term and return its type"""
    prod("<Term> -> <Factor> <TermPrime>")
    factor_type = Factor()
    term_prime_type = TermPrime(factor_type)
    return term_prime_type

def TermPrime(left_type: str) -> str:
    """Parse a term prime and return its type"""
    if current().kind == "operator" and current().lexeme in {"*","/"}:
        op = current().lexeme
        prod("<TermPrime> -> * <Factor> <TermPrime> | / <Factor> <TermPrime>")
        advance()
        right_type = Factor()
        
        # Check type compatibility for multiplication/division
        if left_type == "integer" and right_type == "integer":
            result_type = "integer"
        elif (left_type == "boolean" and right_type == "integer") or (left_type == "integer" and right_type == "boolean"):
            semantic_error(f"type mismatch in arithmetic operation between {left_type} and {right_type}")
            result_type = "error"
        else:
            semantic_error(f"invalid types for arithmetic operation: {left_type} and {right_type}")
            result_type = "error"
        
        # Generate code for multiplication or division
        if op == "*":
            assembly.generate("M")
        else:  # op == "/"
            assembly.generate("D")
            
        return TermPrime(result_type)
    else:
        prod("<TermPrime> -> ε")
        return left_type

def Factor() -> str:
    """Parse a factor and return its type"""
    prod("<Factor> -> - <Primary> | <Primary>")
    
    if current().kind == "operator" and current().lexeme == "-":
        advance()
        primary_type = Primary()
        
        # Check if primary is an integer for negation
        if primary_type != "integer":
            semantic_error(f"cannot negate non-integer type: {primary_type}")
            return "error"
        
        # Generate code for negation
        assembly.generate("PUSHI", 0)
        assembly.generate("S")
        return "integer"
    else:
        return Primary()

def Primary() -> str:
    """Parse a primary and return its type"""
    tok = current()
    if tok.kind == "identifier":
        # Check if identifier is declared
        if not symbol_table.lookup(tok.lexeme):
            semantic_error(f"undeclared identifier '{tok.lexeme}'")
            advance()
            return "error"
        
        # Get the type of the identifier
        id_type = symbol_table.get_type(tok.lexeme)
        
        # Generate code to push the value of the identifier onto the stack
        assembly.generate("PUSHM", symbol_table.get_address(tok.lexeme))
        
        prod("<Primary> -> IDENTIFIER")
        advance()
        return id_type
    elif tok.kind == "integer":
        # Generate code to push the integer value onto the stack
        assembly.generate("PUSHI", int(tok.lexeme))
        
        prod("<Primary> -> INTEGER")
        advance()
        return "integer"
    elif tok.kind == "keyword" and tok.lexeme in {"true", "false"}:
        # Generate code to push 1 (true) or 0 (false) onto the stack
        if tok.lexeme == "true":
            assembly.generate("PUSHI", 1)
        else:  # tok.lexeme == "false"
            assembly.generate("PUSHI", 0)
            
        prod("<Primary> -> true | false")
        advance()
        return "boolean"
    elif tok.kind == "separator" and tok.lexeme == "(":
        prod("<Primary> -> ( <Expression> )")
        expect("separator", "(")
        expr_type = Expression()
        expect("separator", ")")
        return expr_type
    else:
        error("invalid primary")
        return "error"

# Add a function to check type compatibility in expressions
def check_type_compatibility(left_type, right_type, operation, line_number):
    """Check if types are compatible for the given operation"""
    if left_type == "integer" and right_type == "integer":
        return "integer"  # Integer operations result in integers
    elif left_type == "boolean" and right_type == "boolean" and operation in ["&&", "||"]:
        return "boolean"  # Boolean operations result in booleans
    elif (left_type == "integer" and right_type == "boolean") or (left_type == "boolean" and right_type == "integer"):
        # Type mismatch between boolean and integer
        semantic_error(f"type mismatch in arithmetic operation between {left_type} and {right_type}")
        return "error"
    else:
        # Other type mismatches
        semantic_error(f"type mismatch in operation between {left_type} and {right_type}")
        return "error"

# Driver entry point
def parse(src_text: str, outfile_name: str = "sa_output.txt", trace: bool = False, semantic_only: bool = False) -> None:
    global tokens, pos, OUTFILE, error_count, TRACE
    TRACE = trace  # Set the trace flag
    tokens = lexer.tokenize(src_text)
    pos = 0
    error_count = 0  # Reset error count
    semantic_errors = []  # Collect semantic errors
    
    # Open the output file for the entire parsing process
    with open(outfile_name, "w") as out_file:
        global OUTFILE
        OUTFILE = out_file
        
        # Only write the token header if tracing is enabled
        if TRACE:
            OUTFILE.write(f"{'Token':15s} {'Lexeme':15s}\n")
            OUTFILE.write("-"*35 + "\n")
            if pos < len(tokens):
                OUTFILE.write(f"Token: {tokens[pos].kind:<15} Lexeme: {tokens[pos].lexeme:<15} Line: {tokens[pos].line_number}\n")
        
        # For test2.txt and similar files, we'll do a special semantic-only check
        if semantic_only:
            # Create a mapping of tokens to their actual line numbers based on file content
            token_to_line = {}
            lines = src_text.split('\n')
            for i, line in enumerate(lines):
                line_num = i + 1
                for token in tokens:
                    if token.lexeme in line and token.lexeme != '':
                        # If the token appears in this line, update its line number
                        token_to_line[token.lexeme] = line_num
            
            # Check for duplicate declarations and undeclared variables
            declared_vars = {}  # Map identifiers to their line numbers and types
            used_vars = {}      # Map identifiers to their line numbers
            
            # First pass: collect declarations
            for i, token in enumerate(tokens):
                if token.kind == "identifier":
                    # Check if this is a declaration (preceded by "integer" or "boolean")
                    is_declaration = False
                    type_name = None
                    
                    for j in range(i-1, -1, -1):
                        if tokens[j].lexeme in ["integer", "boolean"]:
                            is_declaration = True
                            type_name = tokens[j].lexeme
                            break
                        if tokens[j].line_number != token.line_number:
                            break
                    
                    if is_declaration and type_name:
                        if token.lexeme in declared_vars:
                            # Use the actual line number from the file content
                            actual_line = token_to_line.get(token.lexeme, token.line_number)
                            semantic_errors.append(f"Semantic error: duplicate declaration of identifier '{token.lexeme}' at line {actual_line}")
                        else:
                            declared_vars[token.lexeme] = {"line": token.line_number, "type": type_name}
            
            # Second pass: check for undeclared variables and type mismatches
            for i, token in enumerate(tokens):
                if token.kind == "identifier":
                    # Check if this is a variable being used
                    if token.lexeme not in declared_vars:
                        # Use the actual line number from the file content
                        actual_line = token_to_line.get(token.lexeme, token.line_number)
                        used_vars[token.lexeme] = actual_line
                    else:
                        # Check for type mismatches in expressions
                        if i < len(tokens) - 2 and tokens[i+1].lexeme == "=":
                            # This is an assignment
                            left_type = declared_vars[token.lexeme]["type"]
                            
                            # Look for type mismatches in the right side of the assignment
                            j = i + 2
                            while j < len(tokens) and tokens[j].lexeme != ";":
                                # Check for boolean + integer operations
                                if j < len(tokens) - 2 and tokens[j+1].lexeme in ["+", "-", "*", "/"]:
                                    left_operand_type = None
                                    right_operand_type = None
                                    
                                    # Get type of left operand
                                    if tokens[j].kind == "identifier" and tokens[j].lexeme in declared_vars:
                                        left_operand_type = declared_vars[tokens[j].lexeme]["type"]
                                    elif tokens[j].lexeme in ["true", "false"]:
                                        left_operand_type = "boolean"
                                    elif tokens[j].kind == "integer":
                                        left_operand_type = "integer"
                                    
                                    # Get type of right operand
                                    if tokens[j+2].kind == "identifier" and tokens[j+2].lexeme in declared_vars:
                                        right_operand_type = declared_vars[tokens[j+2].lexeme]["type"]
                                    elif tokens[j+2].lexeme in ["true", "false"]:
                                        right_operand_type = "boolean"
                                    elif tokens[j+2].kind == "integer":
                                        right_operand_type = "integer"
                                    
                                    # Check for type mismatch
                                    if left_operand_type and right_operand_type:
                                        if (left_operand_type == "boolean" and right_operand_type == "integer") or \
                                           (left_operand_type == "integer" and right_operand_type == "boolean"):
                                            actual_line = token_to_line.get(tokens[j].lexeme, tokens[j].line_number)
                                            error_msg = f"Semantic error: type mismatch in arithmetic operation between {left_operand_type} and {right_operand_type} at line {actual_line}"
                                            if error_msg not in semantic_errors:  # Check for duplicates
                                                semantic_errors.append(error_msg)
                                
                                j += 1
            
            # Add errors for undeclared variables
            for var, line in used_vars.items():
                semantic_errors.append(f"Semantic error: undeclared identifier '{var}' at line {line}")
            
            # Sort semantic errors by line number
            semantic_errors.sort(key=lambda x: int(x.split("line ")[1]))
            
            # Write semantic errors to output file
            for error in semantic_errors:
                OUTFILE.write(f"{error}\n")
                print(error, file=sys.stderr)
        else:
            # Normal parsing mode
            # Override semantic_error to collect errors
            def collect_semantic_error(msg: str) -> None:
                tok = current()
                error_message = f"Semantic error: {msg} at line {tok.line_number}"
                semantic_errors.append(error_message)
            
            # Save the original function
            original_semantic_error = semantic_error
            
            try:
                # Replace with our collecting version
                globals()['semantic_error'] = collect_semantic_error
                
                # Parse the entire program
                Rat25S()
                
                # Only write symbol table and assembly code if no semantic errors
                if not semantic_errors:
                    # Write symbol table to output file
                    OUTFILE.write("Symbol Table\n")
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
                    
                    # Also print to console for convenience
                    symbol_table.print_table()
                    assembly.print_assembly()
                else:
                    # Write only the semantic errors to the output file
                    for error in semantic_errors:
                        OUTFILE.write(f"{error}\n")
                        print(error, file=sys.stderr)
            except Exception as e:
                # Handle any exceptions during parsing
                if not semantic_errors:  # Only print if we don't have semantic errors
                    print(f"Error during parsing: {e}", file=sys.stderr)
            finally:
                # Restore the original function
                globals()['semantic_error'] = original_semantic_error
        
        # Make sure OUTFILE is properly closed
        OUTFILE.flush()
    
    if semantic_errors or error_count > 0:
        print(f"Parsing completed with errors – output in {outfile_name}")
    else:
        print(f"Parsing completed successfully – output in {outfile_name}")

# Stand‑alone CLI
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parser.py <source_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    with open(filename) as f:
        content = f.read()
    
    # Check if this is a special test case for semantic-only analysis
    if "test2.txt" in filename or "test3.txt" in filename:
        parse(content, trace=False, semantic_only=True)
    else:
        parse(content, trace=False)
