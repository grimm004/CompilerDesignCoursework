import sys
import re
from typing import List, Set, Dict, Iterator, Tuple, Pattern


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


class FirstOrderFormula:
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
                raise FirstOrderError(self.location, "Element '%s' already in %s set." % (element, self.name))
            if element != "":
                self.append(element)
        self.verify()
        self.parsed = True

    def verify(self) -> None:
        for element in self:
            if not self.regex.match(element):
                raise FirstOrderError(self.location, "Invalid %s syntax '%s'." % (self.name, element))

    def token_groups(self) -> Iterator[Tuple[Tuple[bool, str], str]]:
        for element in self:
            yield (False, element), self.name.upper()

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


class OperatorSet(InputSet):
    def __init__(self, name: str, operators: List[str]):
        super().__init__(name)
        self.operators: List[str] = operators
        self.operator_map: List[Tuple[str, str]] = []
        self.regex: Pattern = re.compile(r"[A-Za-z0-9_\\]+")

    def verify(self) -> None:
        super().verify()
        if len(self) != len(self.operators):
            raise FirstOrderError(self.location, "%s set must contain one element (currently %d)." %
                                  (self.name.capitalize(), len(self)))

    def parse(self, line_index: Location, text: str) -> None:
        super().parse(line_index, text)
        for i in range(len(self)):
            self.operator_map.append((self[i], self.operators[i]))

    def token_groups(self) -> Iterator[Tuple[str, str]]:
        for literal, operator_type in sorted(self.operator_map, key=lambda op: len(op[0]), reverse=True):
            yield (False, literal), operator_type


class VariableSet(LiteralSet):
    def __init__(self) -> None:
        super().__init__("variable")


class ConstantSet(LiteralSet):
    def __init__(self) -> None:
        super().__init__("constant")


class PredicateSet(LiteralSet):
    def __init__(self) -> None:
        super().__init__("predicate")
        self.data: List[Tuple[str, int]] = []
        self.regex: Pattern = re.compile(r"[^\[\](),]+\[[0-9]+\]")

    def parse(self, line_index: Location, text: str) -> None:
        super().parse(line_index, text)
        self.data: List[Tuple[str, int]] = []
        for predicate_string in self:
            self.data.append((predicate_string.split("[")[0], int(predicate_string.split("[")[1].split("]")[0])))

    def contains(self, item: str) -> bool:
        return item in [predicate[0] for predicate in self.data]

    def token_groups(self) -> Iterator[Tuple[Tuple[bool, str], str]]:
        for element in self.data:
            yield (False, element[0]), self.name.upper()


class EqualitySet(OperatorSet):
    def __init__(self) -> None:
        super().__init__("equality", ["EQUALS"])
        self.regex: Pattern = re.compile(r"[A-Za-z0-9_\\=]+")


class ConnectiveSet(OperatorSet):
    def __init__(self) -> None:
        super().__init__("connective", ["AND", "OR", "IMPLIES", "IFF", "NOT"])


class QuantifierSet(OperatorSet):
    def __init__(self) -> None:
        super().__init__("quantifier", ["EXISTS", "FOR_ALL"])


class FirstOrderError(Exception):
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
    def __init__(self, location: Location, token_type: str, value: str) -> None:
        self.location: Location = location
        self.type: str = token_type.upper()
        self.value: str = value

    def __str__(self) -> str:
        return "Token(%s)" % ", ".join(s for s in [self.type, self.location.string(), "'%s'" % self.value]
                                       if s not in [None, ""])

    def __repr__(self) -> str:
        return self.__str__()


class FirstOrderLexer:
    def __init__(self, input_sets: Dict[str, InputSet],
                 formula: FirstOrderFormula = FirstOrderFormula()) -> None:

        self.group_map: Dict[str, str] =\
            {"GROUP_OB": "OPEN_BRACKET", "GROUP_CB": "CLOSE_BRACKET", "GROUP_CO": "COMMA"}
        group_values: List[Tuple[str, Tuple[bool, str]]] =\
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
        self.input_formula: FirstOrderFormula = formula
        self.index: int = 0
        self.white_space_regex = re.compile(r"\S")

    def set_input(self, formula: FirstOrderFormula) -> None:
        self.input_formula: FirstOrderFormula = formula
        self.index: int = 0

    def reset(self) -> None:
        self.index = 0

    def next_token(self) -> Token or None:
        if self.index > len(self.input_formula.text) - 1:
            return None

        whitespace_match = self.white_space_regex.search(self.input_formula.text, self.index)

        if not whitespace_match:
            return None

        self.index = whitespace_match.start()

        rule_match = self.regex_rules.match(self.input_formula.text, self.index)

        if rule_match:
            group = rule_match.lastgroup
            token = Token(self.input_formula.get_location(self.index), self.group_map[group], rule_match.group(group))
            self.index = rule_match.end()
            return token

        raise FirstOrderError(self.input_formula.get_location(self.index), "Could not match expression.")

    def tokens(self, reset=False) -> Iterator[Token]:
        if reset:
            self.reset()
        token = self.next_token()
        while token is not None:
            yield token
            token = self.next_token()


class FirstOrderParser:
    def __init__(self, first_order_lexer: FirstOrderLexer) -> None:
        self.lexer: FirstOrderLexer = first_order_lexer

    def parse(self):
        pass


if __name__ == '__main__':
    def main():
        with open(sys.argv[1], "rt") as file:
            input_text: str = file.read().replace("\t", "    ")

        lines: List[str] = input_text.split("\n")

        try:
            sets: Dict[str, InputSet] = {
                "variables": VariableSet(),
                "constants": ConstantSet(),
                "predicates": PredicateSet(),
                "equality": EqualitySet(),
                "connectives": ConnectiveSet(),
                "quantifiers": QuantifierSet()
            }
            input_formula: FirstOrderFormula = FirstOrderFormula()

            in_formula: bool = False
            formula_found: bool = False
            for i_ in range(len(lines)):
                line_parts = lines[i_].split(" ")
                current_set: str = line_parts[0][:-1]
                if current_set in sets.keys():
                    in_formula = False
                    sets[current_set].parse(Location(i_), " ".join(line_parts[1:]))
                    for element_ in sets[current_set]:
                        for set_ in sets:
                            if set_ != current_set and sets[set_].contains(element_):
                                raise FirstOrderError(Location(i_),
                                                      "Identifier '%s' already in %s set." %
                                                      (element_, sets[set_].name))
                elif current_set == "formula" or in_formula:
                    if not in_formula:
                        input_formula.start_location = Location(i_, len("formula: "))
                    input_formula.text += " ".join(line_parts[0 if in_formula else 1:]) + "\n"
                    in_formula = formula_found = True
                elif current_set == "":
                    pass
                else:
                    raise FirstOrderError(Location(i_, 0), "Unrecognised set '%s'." % current_set)

            for set_ in sets.values():
                if not set_.parsed:
                    raise FirstOrderError(None, "%s set could not be found." % set_.name.capitalize())

            if not formula_found:
                raise FirstOrderError(None, "Formula cloud not be found.")

            lexer: FirstOrderLexer = FirstOrderLexer(sets, input_formula)
            tokens: List[Token] = list(lexer.tokens())
            print("\n".join(str(token) for token in tokens))
            parser: FirstOrderParser = FirstOrderParser(lexer)
        except FirstOrderError as e:
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
            print("\n".join([spacer_string, err_string, file_reference_string, spacer_string]))
            exit(1)


    main()
