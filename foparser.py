from typing import List, Dict, Iterator, Tuple, Pattern, Optional, Match, Set
import sys
import re
from enum import Enum, unique


@unique
class FOTerminal(Enum):
    DOLLAR = -2
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


# class FONonTerminal(Enum):
#     formula = 0
#     formula_ = 1
#     formula__ = 2
#     value = 3
#     predicate = 4
#     constant = 5
#     variable = 6
@unique
class FONonTerminal(Enum):
    constant = 0
    variable = 1
    predicate = 2
    value = 3
    atom = 4
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
                raise FOError(self.location, "Element '%s' already in %s set." % (element, self.name))
            if element != "":
                self.append(element)
        self.verify()
        self.parsed = True

    def verify(self) -> None:
        for element in self:
            if not self.regex.match(element):
                raise FOError(self.location, "Invalid %s syntax '%s'." % (self.name, element))

    def token_groups(self) -> Iterator[Tuple[Tuple[bool, str], FOTerminal]]:
        for element in self:
            yield (False, element), FOTerminal[self.name.upper()]

    def contains(self, item: str) -> bool:
        return item in self

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

    def get_rules(self) -> List[List[FOTerminal or FONonTerminal or Tuple[FOTerminal, str]]]:
        return [[(FOTerminal[self.name.upper()], element)] for element in self]


class OperatorSet(InputSet):
    def __init__(self, name: str, operators: List[str]):
        super().__init__(name)
        self.operators: List[str] = operators
        self.operator_map: List[Tuple[str, str]] = []
        self.regex: Pattern = re.compile(r"[A-Za-z0-9_\\]+")

    def verify(self) -> None:
        super().verify()
        if len(self) != len(self.operators):
            raise FOError(self.location, "%s set must contain one element (currently %d)." %
                          (self.name.capitalize(), len(self)))

    def parse(self, line_index: Location, text: str) -> None:
        super().parse(line_index, text)
        for i in range(len(self)):
            self.operator_map.append((self[i], self.operators[i]))

    def token_groups(self) -> Iterator[Tuple[str, str]]:
        for literal, operator_type in sorted(self.operator_map, key=lambda op: len(op[0]), reverse=True):
            yield (False, literal), FOTerminal[operator_type]


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

    def contains(self, item: str) -> bool:
        return item in [predicate[0] for predicate in self.predicates]

    def token_groups(self) -> Iterator[Tuple[Tuple[bool, str], str]]:
        for predicate in self.predicates:
            yield (False, predicate[0]), FOTerminal[self.name.upper()]

    def get_rules(self) -> List[List[FOTerminal or FONonTerminal or Tuple[FOTerminal, str]]]:
        return [([(FOTerminal.PREDICATE, name), FOTerminal.OPEN_BRACKET] +
                 ([item for i in range(arity) for item in ([FONonTerminal.variable, FOTerminal.COMMA]
                                                           if i != arity - 1 else [FONonTerminal.variable])]) +
                 [FOTerminal.CLOSE_BRACKET]) for (name, arity) in self.predicates]


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


class Token:
    def __init__(self, location: Location, token_type: FOTerminal, value: str) -> None:
        self.location: Location = location
        self.type: FOTerminal = token_type
        self.value: str = value

    def __str__(self) -> str:
        return "Token(%s)" % ", ".join(s for s in [self.type.name, self.location.string(), "'%s'" % self.value]
                                       if s not in [None, ""])

    def __repr__(self) -> str:
        return self.__str__()


class FOLexer:
    def __init__(self, input_sets: Dict[str, InputSet],
                 formula: FOFormula = FOFormula()) -> None:

        self.group_map: Dict[str, FOTerminal] = \
            {"GROUP_OB": FOTerminal.OPEN_BRACKET, "GROUP_CB": FOTerminal.CLOSE_BRACKET, "GROUP_CO": FOTerminal.COMMA}
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

        raise FOError(self.input_formula.get_location(self.index), "Could not match expression.")

    def tokens(self, reset: bool = False) -> Iterator[Token]:
        if reset:
            self.reset()
        token: Token = self.next_token()
        while token is not None:
            yield token
            token: Token = self.next_token()


class FOParser:
    def __init__(self, lexer: FOLexer, predicate_set: PredicateSet):
        self.lexer: FOLexer = lexer
        self.predicate_set = predicate_set
        self.lexer.reset()
        self.current_token: Token = lexer.next_token()
        self.formula()

    def next_token(self):
        self.current_token: Token = self.lexer.next_token()

    def accept(self, terminal: FOTerminal):
        if terminal == self.current_token.type:
            self.next_token()
            print("Matched %s" % terminal.name)
        else:
            raise FOError(self.current_token.location, "Syntax error: cloud not match %s" % terminal.name)

    def formula(self):
        if self.current_token.type == FOTerminal.OP_NOT:
            self.accept(FOTerminal.OP_NOT)
            self.formula()
        elif self.current_token.type == FOTerminal.OP_FOR_ALL:
            self.accept(FOTerminal.OP_FOR_ALL)
            self.accept(FOTerminal.VARIABLE)
            self.formula()
        elif self.current_token.type == FOTerminal.OP_EXISTS:
            self.accept(FOTerminal.OP_EXISTS)
            self.accept(FOTerminal.VARIABLE)
            self.formula()
        elif self.current_token.type == FOTerminal.OPEN_BRACKET:
            self.accept(FOTerminal.OPEN_BRACKET)
            if self.current_token.type in [FOTerminal.VARIABLE, FOTerminal.CONSTANT]:
                self.value()
                self.accept(FOTerminal.OP_EQUALS)
                self.value()
            else:
                self.formula()
                if self.current_token.type in [FOTerminal.OP_AND, FOTerminal.OP_OR,
                                               FOTerminal.OP_IMPLIES, FOTerminal.OP_IFF]:
                    self.accept(self.current_token.type)
                self.formula()
            self.accept(FOTerminal.CLOSE_BRACKET)
        elif self.current_token.type == FOTerminal.PREDICATE:
            arity = list(filter(lambda pred: pred[0] == self.current_token.value, self.predicate_set.predicates))[0][1]
            self.accept(FOTerminal.PREDICATE)
            self.accept(FOTerminal.OPEN_BRACKET)
            for i in range(arity):
                self.accept(FOTerminal.VARIABLE)
                if i != arity - 1:
                    self.accept(FOTerminal.COMMA)
            self.accept(FOTerminal.CLOSE_BRACKET)
        else:
            raise FOError(self.current_token.location, "Syntax error: formula")

    def value(self):
        if self.current_token.type in [FOTerminal.VARIABLE, FOTerminal.CONSTANT]:
            self.accept(self.current_token.type)
        else:
            raise FOError(self.current_token.location, "Syntax error: value")


if __name__ == '__main__':
    from datetime import datetime


    def get_rule_string(rules: Dict[FONonTerminal, List[List[FONonTerminal or FOTerminal or Tuple[FOTerminal, str]]]],
                        literal_sets: List[LiteralSet], terminal_dict: Dict[FOTerminal, str]) -> str:
        return "\n".join(
            [
                "Terminals: {%s}" % ", ".join(
                    ["'%s'" % terminal for terminal in terminal_dict.values()] +
                    [("%s['%s']" % (element[0].name, element[1])) for literal_set in literal_sets
                     for rule in literal_set.get_rules()
                     for element in rule if type(element) is tuple]
                ),
                "Non-terminals: {%s}" % ", ".join(["<%s>" % non_terminal.name for non_terminal in FONonTerminal]),
                "", "Production Rules:"
            ] + [
                "<%s> ::= %s" %
                (non_terminal.name, (" |\n%s" % (" " * len("<%s> ::= " % non_terminal.name))).join([
                    " ".join(
                        ((("'%s'" % terminal_dict[element]) if element in terminal_dict else element.name)
                         if type(element) is FOTerminal else "<%s>" % element.name)
                        if type(element) is not tuple else ("%s['%s']" % (element[0].name, element[1])) for element in
                        elements
                    ) for elements in rule
                ])) for non_terminal, rule in rules.items()
            ])


    def main() -> int:
        use_log: bool = "--no-log" not in sys.argv

        def log(output, end="\n"):
            if use_log:
                with open("log.txt", "at") as log_file:
                    log_file.write(output + end)

        with open(sys.argv[1], "rt") as file:
            input_text: str = file.read().replace("\t", "    ")

        log("%s: Parsing '%s'..." % (datetime.now().strftime("%Y/%m/%d %H:%M:%S"), sys.argv[1]))
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
                    for element in sets[current_set]:
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
            # tokens: List[Token] = list(lexer.tokens())
            log("Lexical analysis successful...")
            # print("\n".join(str(token) for token in tokens))
            parser: FOParser = FOParser(lexer, predicate_set)
            # terminal_dict: Dict[FOTerminal, str] = {
            #     FOTerminal.OPEN_BRACKET: "(",
            #     FOTerminal.CLOSE_BRACKET: ")",
            #     FOTerminal.COMMA: ","
            # }
            # for operator_set in operator_sets.values():
            #     for (literal, operator) in operator_set.operator_map:
            #         terminal_dict[FOTerminal[operator]] = literal
            # log("Grammar:\n" + get_rule_string(parser.rules, list(literal_sets.values()), terminal_dict))
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
            err_output = "\n".join([spacer_string, err_string, file_reference_string, spacer_string])
            log(err_output + "\n\n")
            print(err_output)
            return 1


    exit(main())
