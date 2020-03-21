import sys
import re
from typing import List, Set, Dict, Iterator, Tuple


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


class InputLiteralSet(list):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name
        self.location: Location or None = None
        self.line_text: str = ""

    def parse(self, line_index: Location, text: str) -> None:
        self.location = line_index
        self.line_text = text
        for element in self.line_text.replace("\t", " ").replace("\\", "\\\\").split(" "):
            if element in self:
                raise FirstOrderError(self.location, "Element '%s' already in %s set." % (element, self.name))
            if element != "":
                self.append(element)
        assert self.verify()

    def verify(self) -> bool:
        return True

    def get_regex(self) -> str:
        return r"|".join(self)

    def __str__(self) -> str:
        return "%s(%s)" % \
               (self.name, ", ".join(s for s in
                                     [self.location.string(), "{%s}" % ", ".join("'%s'" % element for element in self)]
                                     if s not in [None, ""]))

    def __repr__(self) -> str:
        return self.__str__()


class VariableLiteralSet(InputLiteralSet):
    def __init__(self) -> None:
        super().__init__("variables")


class ConstantLiteralSet(InputLiteralSet):
    def __init__(self) -> None:
        super().__init__("constants")


class PredicateLiteralSet(InputLiteralSet):
    def __init__(self) -> None:
        super().__init__("predicates")
        self.predicate_data: List[Tuple[str, int]] = []
        self.regex = re.compile(r"[^\[\]]*\[[0-9]+\]")

    def parse(self, line_index: Location, text: str) -> None:
        super().parse(line_index, text)
        self.predicate_data: List[Tuple[str, int]] = []
        for element in self:
            self.predicate_data.append((element.split("[")[0], int(element.split("[")[1].split("]")[0])))

    def verify(self) -> bool:
        for element in self:
            if not self.regex.match(element):
                raise FirstOrderError(self.location, "Invalid predicate syntax '%s'." % element)
        return True

    def get_regex(self) -> str:
        return "|".join([r"(%s)" % element.split("[")[0] for element in self])


class EqualityLiteralSet(InputLiteralSet):
    def __init__(self) -> None:
        super().__init__("equality")

    def verify(self) -> bool:
        if len(self) != 1:
            raise FirstOrderError(self.location, "Equality set must contain one element (currently %d)." % len(self))
        return True


class ConnectiveLiteralSet(InputLiteralSet):
    def __init__(self) -> None:
        super().__init__("connectives")

    def verify(self) -> bool:
        if len(self) != 5:
            raise FirstOrderError(self.location,
                                  "Connective set must contain five elements (currently %d)." % len(self))
        return True


class QuantifierLiteralSet(InputLiteralSet):
    def __init__(self) -> None:
        super().__init__("quantifiers")

    def verify(self) -> bool:
        if len(self) != 2:
            raise FirstOrderError(self.location, "Quantifier set must contain two elements (currently %d)." % len(self))
        return True


class FirstOrderError(Exception):
    def __init__(self, location: Location or None, message: str) -> None:
        self.location: Location or None = location
        self.message: str = message

    def __str__(self) -> str:
        return "Error%s%s" % ((" (%s)" % self.location.string()) if self.location is not None else "",
                              (": %s" % self.message) if self.message != "" else "")

    def __repr__(self) -> str:
        return self.__str__()


class Token:
    def __init__(self, location, token_type, value) -> None:
        self.location: Location = location
        self.type: str = token_type
        self.value: str = value

    def __str__(self) -> str:
        return "Token(%s)" % ", ".join(s for s in [self.type, self.location.string(), "'%s'" % self.value]
                                       if s not in [None, ""])

    def __repr__(self) -> str:
        return self.__str__()


class FirstOrderLexer:
    def __init__(self, input_sets: Dict[str, InputLiteralSet],
                 formula: FirstOrderFormula = FirstOrderFormula()) -> None:
        regex_parts: List[str] = [
            "(?P<open_bracket>[(])",
            "(?P<close_bracket>[)])",
            "(?P<comma>[,])"
        ]
        self.token_types: Set[str] = set()
        for token_type, literal_set in input_sets.items():
            self.token_types.add(token_type)
            regex_parts.append(r"(?P<%s>(%s))" % (token_type, literal_set.get_regex()))

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

        if whitespace_match:
            self.index = whitespace_match.start()
        else:
            return None

        rule_match = self.regex_rules.match(self.input_formula.text, self.index)

        if rule_match:
            group = rule_match.lastgroup
            token = Token(self.input_formula.get_location(self.index), group, rule_match.group(group))
            self.index = rule_match.end()
            return token

        raise FirstOrderError(self.input_formula.get_location(self.index), "Could not match expression.")

    def tokens(self) -> Iterator[Token]:
        token = self.next_token()
        while token is not None:
            yield token
            print(token)
            token = self.next_token()


if __name__ == '__main__':
    with open(sys.argv[1], "rt") as file:
        input_text = file.read()

    try:
        sets: Dict[str, InputLiteralSet] = {
            "variables": VariableLiteralSet(),
            "constants": ConstantLiteralSet(),
            "predicates": PredicateLiteralSet(),
            "equality": EqualityLiteralSet(),
            "connectives": ConnectiveLiteralSet(),
            "quantifiers": QuantifierLiteralSet()
        }
        input_formula: FirstOrderFormula = FirstOrderFormula()

        current_set: str = ""
        in_formula: bool = False
        lines: List[str] = input_text.split("\n")
        for i in range(len(lines)):
            line_parts = lines[i].split(" ")
            current_set = line_parts[0][:-1]
            if current_set in sets.keys():
                in_formula = False
                sets[current_set].parse(Location(i, -1), " ".join(line_parts[1:]))
            elif current_set == "formula" or in_formula:
                if not in_formula:
                    input_formula.start_location = Location(i, len("formula: "))
                input_formula.text += " ".join(line_parts[0 if in_formula else 1:]) + "\n"
                in_formula = True
            elif current_set == "":
                pass
            else:
                raise FirstOrderError(Location(i, 0), "Unrecognised set '%s'." % current_set)
        input_formula.text.replace("\t", "    ")

        # print(sets)
        lexer: FirstOrderLexer = FirstOrderLexer(sets, input_formula)
        tokens: List[Token] = list(lexer.tokens())
        # print(tokens)
    except FirstOrderError as e:
        print(e)
        exit(1)
