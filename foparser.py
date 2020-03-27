from typing import List, Dict, Iterator, Tuple, Pattern, Optional, Match
import re
from enum import Enum, unique


@unique
class FOTokenType(Enum):
    EPSILON = -1
    OPEN_BRACKET = 0
    CLOSE_BRACKET = 1
    COMMA = 2
    # User-defined Sets
    VARIABLE = 3
    CONSTANT = 4
    PREDICATE = 5
    # Equality Set
    OP_EQUALS = 6
    # Connectives Set
    OP_AND = 7
    OP_OR = 8
    OP_IMPLIES = 9
    OP_IFF = 10
    OP_NOT = 11
    # Quantifiers Set
    OP_EXISTS = 12
    OP_FOR_ALL = 13


@unique
class FONonTerminal(Enum):
    constant = 0
    variable = 1
    predicate = 2
    complete_predicate = 3
    value = 4
    formula = 5


class Location:
    def __init__(self, line_index: int = -1, position: int = -1) -> None:
        self.line_index: int = line_index
        self.position_index: int = position

    def string(self) -> str:
        return "%s%s" % \
               (("line %d%s" % (self.line_index + 1, ", " if self.position_index > -1 else ""))
                if self.line_index > -1 else "",
                ("position %d" % (self.position_index + 1)) if self.position_index > -1 else "")

    def __str__(self) -> str:
        return "Location(%s)" % self.string()

    def __repr__(self) -> str:
        return self.__str__()


class FOFormula:
    def __init__(self) -> None:
        self.text: str = ""
        self.start_location = Location()

    def get_location(self, index: int) -> Location:
        assert index < len(self.text)
        line_index = self.text.count("\n", 0, index) + self.start_location.line_index
        position_index = len(self.text[:index].split("\n")[-1]) + (
            self.start_location.position_index if line_index == self.start_location.line_index else 0)
        return Location(line_index, position_index)


class InputSet(list):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.parsed = False
        self.name: str = name
        self.location: Location or None = None
        self.line_text: str = ""
        self.regex: Pattern = re.compile(r"[^\[\](),]+")

    def parse(self, line_index: Location, text: str) -> None:
        self.location = line_index
        self.line_text = text
        for element in self.line_text.replace("\t", " ").split(" "):
            if element in self:
                raise FOError(self.location, "Identifier '%s' already in %s set." % (element, self.name))
            if element != "":
                self.append(element)
        self.verify()
        self.parsed = True

    def verify(self) -> None:
        for element in self:
            if not self.regex.match(element):
                raise FOError(self.location, "Invalid %s syntax '%s'." % (self.name, element))

    def token_groups(self) -> Iterator[Tuple[Tuple[bool, str], FOTokenType]]:
        for element in self:
            yield (False, element), FOTokenType[self.name.upper()]

    def contains(self, item: str) -> bool:
        return item in self

    def get_elements(self) -> List:
        return self

    def __str__(self) -> str:
        return "%s(%s)" % \
               (self.name, ", ".join(s for s in
                                     [self.location.string() if self.location is not None else "",
                                      "{%s}" % ", ".join("'%s'" % str(element) for element in self)]
                                     if s not in [None, ""]))

    def __repr__(self) -> str:
        return self.__str__()


class LiteralSet(InputSet):
    def __init__(self, name: str):
        super().__init__(name)
        self.regex: Pattern = re.compile(r"[A-Za-z0-9_]+")

    def get_rules(self) -> List[List[FOTokenType or FONonTerminal or Tuple[FOTokenType, str]]]:
        return [[(FOTokenType[self.name.upper()], element)] for element in self]


class OperatorSet(InputSet):
    def __init__(self, name: str, operators: List[str]):
        super().__init__(name)
        self.operators: List[str] = operators
        self.operator_map: List[Tuple[str, str]] = []
        self.regex: Pattern = re.compile(r"[A-Za-z0-9_\\]+")

    def verify(self) -> None:
        super().verify()
        if len(self) != len(self.operators):
            raise FOError(self.location, "%s set must contain %d element%s (currently %d)." %
                          (self.name.capitalize(), len(self.operators),
                           "s" if len(self.operators) > 1 else "", len(self)))

    def parse(self, line_index: Location, text: str) -> None:
        super().parse(line_index, text)
        for i in range(len(self)):
            self.operator_map.append((self[i], self.operators[i]))

    def token_groups(self) -> Iterator[Tuple[str, str]]:
        for literal, operator_type in sorted(self.operator_map, key=lambda op: len(op[0]), reverse=True):
            yield (False, literal), FOTokenType[operator_type]


class VariableSet(LiteralSet):
    def __init__(self) -> None:
        super().__init__("variable")


class ConstantSet(LiteralSet):
    def __init__(self) -> None:
        super().__init__("constant")


class PredicateSet(LiteralSet):
    def __init__(self) -> None:
        super().__init__("predicate")
        self.predicates: List[Tuple[str, int]] = []
        self.regex: Pattern = re.compile(r"[^\[\](),]+\[[0-9]+\]")

    def parse(self, line_index: Location, text: str) -> None:
        super().parse(line_index, text)
        self.predicates: List[Tuple[str, int]] = \
            list(map(lambda pred: (pred.split("[")[0], int(pred.split("[")[1].split("]")[0])), self))
        for predicate, arity in self.predicates:
            if arity < 1:
                raise FOError(line_index, "Invalid predicate arity of %d in '%s'." % (arity, predicate))

    def contains(self, item: str) -> bool:
        return item in self.get_elements()

    def get_elements(self) -> List:
        return [predicate[0] for predicate in self.predicates]

    def token_groups(self) -> Iterator[Tuple[Tuple[bool, str], str]]:
        for predicate in self.predicates:
            yield (False, predicate[0]), FOTokenType[self.name.upper()]

    def get_rules(self) -> List[List[FOTokenType or FONonTerminal or Tuple[FOTokenType, str]]]:
        return [([(FOTokenType.PREDICATE, name), FOTokenType.OPEN_BRACKET] +
                 ([item for i in range(arity) for item in ([FONonTerminal.variable, FOTokenType.COMMA]
                                                           if i != arity - 1 else [FONonTerminal.variable])]) +
                 [FOTokenType.CLOSE_BRACKET]) for (name, arity) in self.predicates]


class EqualitySet(OperatorSet):
    def __init__(self) -> None:
        super().__init__("equality", ["OP_EQUALS"])
        self.regex: Pattern = re.compile(r"[A-Za-z0-9_\\=]+")


class ConnectiveSet(OperatorSet):
    def __init__(self) -> None:
        super().__init__("connective", ["OP_AND", "OP_OR", "OP_IMPLIES", "OP_IFF", "OP_NOT"])


class QuantifierSet(OperatorSet):
    def __init__(self) -> None:
        super().__init__("quantifier", ["OP_EXISTS", "OP_FOR_ALL"])


class FOError(Exception):
    def __init__(self, location: Location or None, message: str) -> None:
        self.location: Location or None = location
        self.message: str = message

    def __str__(self) -> str:
        return "Error%s%s" % \
               ((" (%s)" % self.location.string())
                if self.location is not None else "",
                (": %s" % self.message) if self.message != "" else "")

    def __repr__(self) -> str:
        return self.__str__()


class FOLexicalError(FOError):
    pass


class FOParseError(FOError):
    pass


class Token:
    def __init__(self, location: Location, token_type: FOTokenType, value: str) -> None:
        self.location: Location = location
        self.type: FOTokenType = token_type
        self.value: str = value

    def __str__(self) -> str:
        return "Token(%s)" % ", ".join(s for s in [self.type.name, self.location.string(), "'%s'" % self.value]
                                       if s not in [None, ""])

    def __repr__(self) -> str:
        return self.__str__()


class FOLexer:
    def __init__(self, input_sets: Dict[str, InputSet],
                 formula: FOFormula = FOFormula()) -> None:

        self.group_map: Dict[str, FOTokenType] = \
            {"GROUP_OB": FOTokenType.OPEN_BRACKET, "GROUP_CB": FOTokenType.CLOSE_BRACKET, "GROUP_CO": FOTokenType.COMMA}
        group_values: List[Tuple[str, Tuple[bool, str]]] = \
            [("GROUP_OB", (False, "(")), ("GROUP_CB", (False, ")")), ("GROUP_CO", (False, ","))]
        group_index: int = 0
        for literal_set in input_sets.values():
            for (literal_data, token_type) in literal_set.token_groups():
                group_id: str = "GROUP_%d" % group_index
                self.group_map[group_id] = token_type
                group_values.append((group_id, literal_data))
                group_index += 1

        regex_parts: List[str] = [r"(?P<%s>(%s))" % (group_id, literal_value if is_regex else re.escape(literal_value))
                                  for (group_id, (is_regex, literal_value))
                                  in sorted(group_values, key=lambda tv: len(tv[1][1]), reverse=True)]

        self.regex_rules = re.compile(r"|".join(regex_parts))
        self.input_formula: FOFormula = formula
        self.index: int = 0
        self.white_space_regex = re.compile(r"\S")

    def set_input(self, formula: FOFormula) -> None:
        self.input_formula: FOFormula = formula
        self.index: int = 0

    def reset(self) -> None:
        self.index = 0

    def next_token(self) -> Token or None:
        if self.index > len(self.input_formula.text) - 1:
            return None

        whitespace_match: Optional[Match[str]] = self.white_space_regex.search(self.input_formula.text, self.index)

        if not whitespace_match:
            return None

        self.index: int = whitespace_match.start()

        rule_match: Optional[Match[str]] = self.regex_rules.match(self.input_formula.text, self.index)

        if rule_match and rule_match.lastgroup:
            group: Optional[str] = rule_match.lastgroup
            token: Token = Token(self.input_formula.get_location(self.index),
                                 self.group_map[group],
                                 rule_match.group(group))
            self.index: int = rule_match.end()
            return token

        raise FOLexicalError(self.input_formula.get_location(self.index), "Could not match expression.")

    def tokens(self, reset: bool = False) -> Iterator[Token]:
        if reset:
            self.reset()
        token: Token = self.next_token()
        while token is not None:
            yield token
            token: Token = self.next_token()


class FOParser:
    def __init__(self, lexer: FOLexer, predicate_set: PredicateSet, token_type_map: Dict[FOTokenType, str]) -> None:
        self.lexer: FOLexer = lexer
        self.predicate_set = predicate_set
        self.current_token: Token = Token(Location(), FOTokenType.EPSILON, "")
        self.token_type_map: Dict[FOTokenType, str] = token_type_map
        self.parse_tree: Tuple[Dict[int, str], List[Tuple[int, int]]] = ({}, [])
        self.current_node: int = 0

    def parse(self) -> Tuple[Dict[int, str], List[Tuple[int, int]]]:
        self.parse_tree: Tuple[Dict[int, str], List[Tuple[int, int]]] = ({}, [])
        self.current_node: int = 0
        self.lexer.reset()
        self.next_token()
        self.formula()
        if self.current_token:
            raise FOParseError(self.current_token.location,
                               "Syntax Error: Reached end of production with tokens remaining")
        return self.parse_tree

    def create_node(self, upper_node: int or None, label: str) -> int:
        self.current_node += 1
        self.parse_tree[0][self.current_node] = label
        if upper_node:
            self.parse_tree[1].append((upper_node, self.current_node))
        return self.current_node

    def next_token(self) -> None:
        self.current_token: Token = self.lexer.next_token()

    def validate_token(self) -> None:
        if not self.current_token:
            raise FOParseError(None, "Syntax Error: Reached end of file")

    def accept(self, token_type: FOTokenType, upper_node: int) -> None:
        self.validate_token()
        if token_type == self.current_token.type:
            if token_type in [FOTokenType.VARIABLE, FOTokenType.CONSTANT, FOTokenType.PREDICATE]:
                new_node = self.create_node(upper_node, r"<%s>" % self.current_token.type.name.lower())
                self.create_node(new_node, self.current_token.value)
            else:
                self.create_node(upper_node, self.current_token.value.replace("\\", "\\\\"))
            self.next_token()
        else:
            raise FOParseError(self.current_token.location,
                               "Syntax error: expecting '%s'" % self.token_type_map[token_type])

    def formula(self, upper_node: int or None = None) -> None:
        new_node: int = self.create_node(upper_node, "<formula>")
        self.validate_token()
        if self.current_token.type == FOTokenType.OP_NOT:
            self.accept(FOTokenType.OP_NOT, new_node)
            self.formula(new_node)
        elif self.current_token.type == FOTokenType.OP_FOR_ALL:
            self.accept(FOTokenType.OP_FOR_ALL, new_node)
            self.accept(FOTokenType.VARIABLE, new_node)
            self.formula(new_node)
        elif self.current_token.type == FOTokenType.OP_EXISTS:
            self.accept(FOTokenType.OP_EXISTS, new_node)
            self.accept(FOTokenType.VARIABLE, new_node)
            self.formula(new_node)
        elif self.current_token.type == FOTokenType.OPEN_BRACKET:
            self.accept(FOTokenType.OPEN_BRACKET, new_node)
            self.validate_token()
            if self.current_token.type in [FOTokenType.VARIABLE, FOTokenType.CONSTANT]:
                if self.current_token.type in [FOTokenType.VARIABLE, FOTokenType.CONSTANT]:
                    self.accept(self.current_token.type, new_node)
                self.accept(FOTokenType.OP_EQUALS, new_node)
                if self.current_token.type in [FOTokenType.VARIABLE, FOTokenType.CONSTANT]:
                    self.accept(self.current_token.type, new_node)
            else:
                self.formula(new_node)
                self.validate_token()
                if self.current_token.type in [FOTokenType.OP_AND, FOTokenType.OP_OR,
                                               FOTokenType.OP_IMPLIES, FOTokenType.OP_IFF]:
                    self.accept(self.current_token.type, new_node)
                self.formula(new_node)
            self.accept(FOTokenType.CLOSE_BRACKET, new_node)
        elif self.current_token.type == FOTokenType.PREDICATE:
            self.complete_predicate(new_node)
        else:
            raise FOParseError(self.current_token.location,
                               "Syntax error: cloud not match '%s'" % self.current_token.value)

    def complete_predicate(self, upper_node: int):
        new_node = self.create_node(upper_node, "<complete_predicate>")
        arity = list(filter(lambda pred: pred[0] == self.current_token.value, self.predicate_set.predicates))[0][1]
        self.accept(FOTokenType.PREDICATE, new_node)
        self.accept(FOTokenType.OPEN_BRACKET, new_node)
        for i in range(arity):
            self.accept(FOTokenType.VARIABLE, new_node)
            if i != arity - 1:
                self.accept(FOTokenType.COMMA, new_node)
        self.accept(FOTokenType.CLOSE_BRACKET, new_node)

    def value(self, upper_node: int):
        new_node = self.create_node(upper_node, "<value>")
        self.validate_token()
        if self.current_token.type in [FOTokenType.VARIABLE, FOTokenType.CONSTANT]:
            self.accept(self.current_token.type, new_node)
        else:
            raise FOParseError(self.current_token.location,
                               "Syntax error: cloud not match to '%s' variable or constant." % self.current_token)


if __name__ == '__main__':
    import argparse
    import os
    from datetime import datetime

    try:
        from graphviz import Graph, nohtml
    except ImportError:
        Graph: None = None
        nohtml: None = None

    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Parse first-order logic files.")
    arg_parser.add_argument("filename")
    arg_parser.add_argument("-l", "--log", help="output to log file", action="store_true")
    arg_parser.add_argument("-o", "--log-file", help="filename for output log (default: 'log.txt')",
                            default="log.txt", type=str, action="store")
    arg_parser.add_argument("-d", "--dir", help="file output directory (default: input file directory)",
                            default="", type=str, action="store")
    arg_parser.add_argument("-n", "--name", help="logic name and prefix of output files",
                            default=None, type=str, action="store")
    arg_parser.add_argument("--graphviz", help="",
                            default=None, type=str, action="store")
    arg_parser.add_argument("-g", "--graph", help="output a graph representing the parse tree", action="store_true")
    arg_parser.add_argument("-c", "--grammar",
                            help="output the grammar corresponding to the parse tree", action="store_true")
    arg_parser.add_argument("-f", "--format-grammar",
                            help="add line breaks between productions and union productions", action="store_true")
    arg_parser.add_argument("-e", "--label-literals",
                            help="label literal values in the grammar with their token names", action="store_true")
    args = arg_parser.parse_args()

    USE_LOG: bool = args.log
    LOG_FILENAME: str = args.log_file
    OUTPUT_GRAPH: bool = args.graph
    OUTPUT_GRAMMAR: bool = args.grammar
    FORMAT_GRAMMAR: bool = args.format_grammar
    LABEL_LITERALS: bool = args.label_literals
    INPUT_FILENAME: str = args.filename
    OUTPUT_DIRECTORY: str = args.dir
    OUTPUT_PREFIX: str = args.name if args.name else ".".join(INPUT_FILENAME.split(".")[:-1])

    if args.graphviz:
        os.environ["PATH"] += "%s%s" % (os.pathsep, args.graphviz)

    if OUTPUT_DIRECTORY and not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    def log(output: str, end: str = "\n", include_date_time: bool = True, include_file: bool = True) -> None:
        text: str = "%s%s%s%s%s" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                                    if include_date_time else "",
                                    " <%s>" % OUTPUT_PREFIX if include_file else "",
                                    ": " if include_date_time or include_file else "", output, end)
        if USE_LOG:
            with open(LOG_FILENAME, "at") as log_file:
                log_file.write(text)
        else:
            print(text, end="")


    if OUTPUT_GRAPH and not Graph:
        log("Warning: graphviz could not be imported (cannot output graph).")


    def get_rule_string(grammar: Dict[FONonTerminal,
                                      List[List[FONonTerminal or FOTokenType or Tuple[FOTokenType, str]]]],
                        literal_sets: List[LiteralSet], token_type_map: Dict[FOTokenType, str]) -> str:
        return "%s\n%s" % ("\n".join([
            "Terminals: {%s}" % ", ".join(
                ["'%s'" % token_type for token_type in token_type_map.values()] +
                [("%s['%s']" % (element[0].name, element[1])) for literal_set in literal_sets
                 for rule in literal_set.get_rules()
                 for element in rule if type(element) is tuple]
            ),
            "Non-terminals: {%s}" % ", ".join(["<%s>" % non_terminal.name for non_terminal in FONonTerminal]),
            "", "Production Rules:"
        ]), ("\n\n" if FORMAT_GRAMMAR else "\n").join([
            "<%s> ::= %s" %
            (non_terminal.name,
             (" |%s" % (
                 ("\n" + (" " * len(
                     "<%s> ::= " % non_terminal.name))) if FORMAT_GRAMMAR else " ")).join(
                 [" ".join(
                     ((("'%s'" % token_type_map[
                         element]) if element in token_type_map else element.name)
                      if type(element) is FOTokenType else "<%s>" % element.name)
                     if type(element) is not tuple else ("%s['%s']" % (element[0].name, element[1])
                                                         if LABEL_LITERALS else "'%s'" % element[1])
                     for element in
                     elements
                 ) for elements in rule])) for non_terminal, rule in grammar.items()]))


    class FOFileError(FOError):
        pass


    def main() -> int:
        try:
            log("Loading '%s'..." % INPUT_FILENAME, include_file=False)
            with open(INPUT_FILENAME, "rt") as file:
                input_text: str = file.read().replace("\t", "    ")
        except FileNotFoundError:
            log("Could not find input file...")
            exit(1)

        lines: List[str] = input_text.split("\n")

        try:
            predicate_set = PredicateSet()
            literal_sets: Dict[str, LiteralSet] = {
                "variables": VariableSet(),
                "constants": ConstantSet(),
                "predicates": predicate_set
            }
            operator_sets: Dict[str, OperatorSet] = {
                "equality": EqualitySet(),
                "connectives": ConnectiveSet(),
                "quantifiers": QuantifierSet()
            }
            sets: Dict[str, InputSet] = {**literal_sets, **operator_sets}

            input_formula: FOFormula = FOFormula()
            in_formula: bool = False
            formula_found: bool = False
            for i in range(len(lines)):
                line_parts = lines[i].split(" ")
                current_set: str = line_parts[0][:-1]
                if current_set in sets.keys():
                    in_formula = False
                    sets[current_set].parse(Location(i), " ".join(line_parts[1:]))
                    for element in sets[current_set].get_elements():
                        for set_ in sets:
                            if set_ != current_set and sets[set_].contains(element):
                                raise FOError(Location(i),
                                              "Identifier '%s' already in %s set." % (element, sets[set_].name))
                elif current_set == "formula" or in_formula:
                    if not in_formula:
                        input_formula.start_location = Location(i, len("formula: "))
                    input_formula.text += " ".join(line_parts[0 if in_formula else 1:]) + "\n"
                    in_formula = formula_found = True
                elif current_set == "":
                    pass
                else:
                    raise FOError(Location(i, 0), "Unrecognised set '%s'." % current_set)

            for set_ in sets.values():
                if not set_.parsed:
                    raise FOError(None, "%s set could not be found." % set_.name.capitalize())

            if not formula_found:
                raise FOError(None, "Formula cloud not be found.")

            lexer: FOLexer = FOLexer(sets, input_formula)
            token_type_map: Dict[FOTokenType, str] = {
                FOTokenType.OPEN_BRACKET: "(",
                FOTokenType.CLOSE_BRACKET: ")",
                FOTokenType.COMMA: ","
            }
            for operator_set in operator_sets.values():
                for (literal, operator) in operator_set.operator_map:
                    token_type_map[FOTokenType[operator]] = literal

            for literal_set in literal_sets.values():
                token_type_map[FOTokenType[literal_set.name.upper()]] = literal_set.name

            log("Parsing formula...")
            parser: FOParser = FOParser(lexer, predicate_set, token_type_map)
            parse_tree_tuple: Tuple[Dict[int, str], List[Tuple[int, int]]] = parser.parse()
            log("Parse successful.")

            if OUTPUT_GRAPH and Graph and nohtml:
                log("Generating graph for parse tree...")
                parse_tree: Graph = Graph()
                nodes, edges = parse_tree_tuple
                for node, label in nodes.items():
                    parse_tree.node(str(node), nohtml(label.replace("\\", "\\\\")))
                parse_tree.edges([(str(edge[0]), str(edge[1])) for edge in edges])
                graph_output: str = os.path.join(OUTPUT_DIRECTORY, "%s.parse-tree" % OUTPUT_PREFIX)
                log("Outputting parse tree to '%s.pdf'..." % graph_output)
                parse_tree.format = "pdf"
                parse_tree.render(graph_output, cleanup=True)

            grammar: Dict[FONonTerminal, List[List[FONonTerminal or FOTokenType or Tuple[FOTokenType, str]]]] = {
                **{FONonTerminal[literal_set.name]: literal_set.get_rules() for literal_set in literal_sets.values()},
                FONonTerminal.value: [[FONonTerminal.constant], [FONonTerminal.variable]],
                FONonTerminal.formula: [
                    [FOTokenType.OPEN_BRACKET,
                     FONonTerminal.formula,
                     FOTokenType.OP_AND,
                     FONonTerminal.formula,
                     FOTokenType.CLOSE_BRACKET],
                    [FOTokenType.OPEN_BRACKET,
                     FONonTerminal.formula,
                     FOTokenType.OP_OR,
                     FONonTerminal.formula,
                     FOTokenType.CLOSE_BRACKET],
                    [FOTokenType.OPEN_BRACKET,
                     FONonTerminal.formula,
                     FOTokenType.OP_IMPLIES,
                     FONonTerminal.formula,
                     FOTokenType.CLOSE_BRACKET],
                    [FOTokenType.OPEN_BRACKET,
                     FONonTerminal.formula,
                     FOTokenType.OP_IFF,
                     FONonTerminal.formula,
                     FOTokenType.CLOSE_BRACKET],
                    [FOTokenType.OPEN_BRACKET,
                     FONonTerminal.value,
                     FOTokenType.OP_EQUALS,
                     FONonTerminal.value,
                     FOTokenType.CLOSE_BRACKET],
                    [FONonTerminal.complete_predicate],
                    [FOTokenType.OP_NOT, FONonTerminal.formula],
                    [FOTokenType.OP_EXISTS, FONonTerminal.variable, FONonTerminal.formula],
                    [FOTokenType.OP_FOR_ALL, FONonTerminal.variable, FONonTerminal.formula]
                ]
            }

            if OUTPUT_GRAMMAR:
                grammar_filename: str = os.path.join(OUTPUT_DIRECTORY, "%s.grammar.txt" % OUTPUT_PREFIX)
                log("Outputting grammar to '%s'..." % grammar_filename)
                with open(grammar_filename, "wt") as grammar_file:
                    grammar_file.write(("First Order Grammar for '%s':\n" % OUTPUT_PREFIX) +
                                       get_rule_string(grammar, list(literal_sets.values()), token_type_map))
                log("Done.")
            return 0
        except FOError as e:
            err_string: str = str(e)
            spacer_string: str = "-" * len(err_string) + "-"
            file_reference_string = ""
            if e.location is not None and e.location.line_index > -1:
                error_line: str = lines[e.location.line_index]
                spacer_string += "-" * max(0, len(error_line) - len(err_string) + 4) + "|"
                file_reference_string += \
                    ">>> %s" % error_line + (" " * (len(spacer_string) - len(error_line) - 5) + "|")
                if e.location.position_index > -1:
                    file_reference_string += "\n"
                    pointer_string: str = " " * (e.location.position_index + 4) + "^"
                    file_reference_string += \
                        pointer_string + (" " * (len(spacer_string) - len(pointer_string) - 1) + "|")
            err_string += (" " * (len(spacer_string) - len(err_string) - 1) + "|")
            err_output = "\n".join([spacer_string, err_string] +
                                   ([file_reference_string] if file_reference_string else []) +
                                   [spacer_string])
            log("An error occurred while %s:" %
                ("parsing the formula" if type(e) is FOParseError else
                 ("performing lexical analysis" if type(e) is FOLexicalError else "parsing the file")))
            log(err_output, include_file=False, include_date_time=False)
            return 1
        finally:
            log("", include_file=False, include_date_time=False)


    exit(main())
