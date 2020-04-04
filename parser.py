from typing import Set, Dict, List, Iterator, Tuple


class Grammar:
    def __init__(self,
                 terminals: Set[str],
                 non_terminals: Set[str],
                 production_rules: Dict[str, List[List[str]]],
                 start_symbol: str) -> None:

        assert "" not in terminals
        assert not (terminals & non_terminals)
        assert start_symbol in non_terminals
        assert "$" not in non_terminals
        assert set(production_rules.keys()) == non_terminals

        self.terminals: Set[str] = terminals
        self.non_terminals: Set[str] = non_terminals
        self.production_rules = production_rules
        self.start_symbol: str = start_symbol

    def rules(self) -> List[List[str]]:
        return [rule for _, rule in self]

    def __iter__(self) -> Iterator[Tuple[str, List[str]]]:
        return [(non_terminal, rule) for non_terminal, rules in self.production_rules.items() for rule in rules] \
            .__iter__()

    def __getitem__(self, item: str or Tuple[str, int]) -> List[List[str]] or List[str]:
        return self.production_rules[item] if type(item) is str else self.production_rules[item[0]][item[1]]

    def make_ll1(self) -> None:
        changed = True
        while changed:
            e: bool = self.epsilon_derivation_elimination()
            lr: bool = self.left_recursion_elimination()
            lf: bool = self.left_factorisation()
            changed = e or lr or lf
        print(self.production_rules)

    def epsilon_derivation_elimination(self) -> bool:
        changed = False

        null_productions = [(non_term, i)
                            for non_term, rules in self.production_rules.items()
                            for i in range(len(rules)) if not rules[i]]

        for null_prod in null_productions:
            changed = True
            self.production_rules[null_prod[0]].pop(null_prod[1])
            for rules in self.production_rules.values():
                for rule in rules:
                    for symbol in rule:
                        if symbol == null_prod[0]:
                            new_rule = rule.copy()
                            new_rule.remove(symbol)
                            if new_rule:
                                rules.append(new_rule)

        return changed

    def left_recursion_elimination(self) -> bool:
        changed = False

        non_terminals: List[str] = list(self.non_terminals)

        for i in range(len(non_terminals)):
            a_i = non_terminals[i]
            for j in range(i):
                a_j = non_terminals[j]
                for k in range(len(self.production_rules[a_i])):
                    rule = self.production_rules[a_i][k]
                    if len(rule) > 0 and rule[0] == a_j:
                        changed = True
                        rule = self.production_rules[a_i].pop(k)
                        for prod in self.production_rules[a_j]:
                            self.production_rules[a_i].append(prod + rule[1:])

            left_recursive = []
            non_left_recursive = []
            for rule in self.production_rules[a_i]:
                if len(rule) > 0 and rule[0] == a_i:
                    left_recursive.append(rule[1:])
                else:
                    non_left_recursive.append(rule.copy())

            if len(left_recursive) > 0:
                changed = True
                a_i_prime = "%s'" % a_i
                self.non_terminals.add(a_i_prime)
                self.production_rules[a_i] = [rule + [a_i_prime] for rule in non_left_recursive]
                self.production_rules[a_i_prime] = [rule + [a_i_prime] for rule in left_recursive]

        return changed

    def left_factorisation(self) -> bool:
        changed = False

        for non_terminal, rules in self.production_rules.items():
            if len(rules) < 2:
                continue

            longest_prefix = rules[0]
            for i in range(1, len(rules)):
                match = []
                if len(longest_prefix) == 0:
                    break
                for j in range(len(rules[i])):
                    if j < len(longest_prefix) and longest_prefix[j] == rules[i][j]:
                        match += longest_prefix[j]
                    else:
                        break
                longest_prefix = match

        return changed

    def first_and_follow(self):
        # first & follow sets, epsilon-productions
        first: Dict[str, Set[str]] = {i: set() for i in self.non_terminals}
        first.update((i, {i}) for i in self.terminals)
        follow: Dict[str, Set[str]] = {i: set() for i in self.non_terminals}
        epsilon: Set[str] = set()

        while True:
            updated = False

            for nt, expression in self:
                # FIRST set w.r.t epsilon-productions
                for symbol in expression:
                    updated |= self.union(first[nt], first[symbol])
                    if symbol not in epsilon:
                        break
                else:
                    updated |= self.union(epsilon, {nt})

                # FOLLOW set w.r.t epsilon-productions
                aux = follow[nt]
                for symbol in reversed(expression):
                    if symbol in follow:
                        updated |= self.union(follow[symbol], aux)
                    if symbol in epsilon:
                        aux = aux.union(first[symbol])
                    else:
                        aux = first[symbol]

            if not updated:
                return first, follow, epsilon

    @staticmethod
    def union(first, begins):
        n = len(first)
        first |= begins
        return len(first) != n

    def parse_table(self) -> Dict[str, Dict[str, int]]:
        first, follow, _ = self.first_and_follow()
        print(first, follow)

        parse_table: Dict[str, Dict[str, int]] = \
            {non_terminal: {terminal: -1 for terminal in (self.terminals | {"$"})}
             for non_terminal in self.non_terminals}

        rules: List[Tuple[str, List[str]]] = list(self)
        for i in range(len(rules)):
            non_terminal, rule = rules[i]
            for a in first[non_terminal]:
                if a == "":
                    for b in follow[non_terminal]:
                        parse_table[non_terminal][b] = i
                else:
                    if parse_table[non_terminal][a] > -1:
                        print("Rule '%s' already in table at (%s, '%s') (found while adding '%s')." %
                              (non_terminal + " -> " + str(self[non_terminal, parse_table[non_terminal][a]]),
                               non_terminal, a,
                               non_terminal + " -> " + str(self[non_terminal, i])))
                        # exit(1)
                    parse_table[non_terminal][a] = i

        print("\n\n".join(["\n".join(["%s[%d] -> %s" % (rules[i][0], i, str(rules[i][1]))
                                      for i in range(len(rules))])]))
        print("\n".join(["\t".join([""] + [item for item in list(parse_table.values())[0]])] +
                        ["\t".join([key] + [("%d" % item if item > -1 else "-") for item in row.values()])
                         for key, row in parse_table.items()]))
        return parse_table


class Parser:
    def __init__(self, grammar: Grammar) -> None:
        self.grammar: Grammar = grammar

    def parse(self, tokens: List[str]) -> None:
        parse_table: Dict[str, Dict[str, int]] = self.grammar.parse_table()

        if not tokens:
            print("No tokens.")
            return

        finished = False
        stack: List[str] = ["$", self.grammar.start_symbol]
        t_pointer = 0
        while stack != ["$"] and t_pointer < len(tokens):
            a: str = tokens[t_pointer]
            x = stack[-1]

            if x in self.grammar.terminals:
                if x == a:
                    print("Match '%s'." % a)
                    stack.pop()
                    t_pointer += 1
                else:
                    print("Syntax error: expecting '%s', got '%s'" % (x, a))
                    return
            else:
                production: int = parse_table[x][a]
                if production > -1:
                    stack.pop()
                    stack += reversed(self.grammar[x, production])
                    print("Production: %s[%d]" % (x, production))
                else:
                    print("Syntax error")
                    return

        if finished and stack.pop() == "$":
            print("Success")
        else:
            print("Failure")


if __name__ == "__main__":
    def main() -> None:
        production_rules: Dict[str, List[List[str]]] = {
            "formula": ["( formula'".split(),
                        "PREDICATE[1] ( VAR )".split(),
                        "PREDICATE[2] ( VAR , VAR )".split(),
                        "NOT formula".split(),
                        "FOR_ALL VAR formula".split(),
                        "EXISTS VAR formula".split()],
            "formula'": ["formula bin_op formula )".split(),
                         "VAR EQUALS VAR )".split(),
                         "CONST EQUALS CONST )".split()],
            "bin_op": [[bin_op] for bin_op in "AND OR IMPLIES IFF".split()],
        }

        non_terminals: Set[str] = set(production_rules.keys())
        terminals: Set[str] = set([symbol
                                   for rules in production_rules.values()
                                   for rule in rules
                                   for symbol in rule if symbol not in non_terminals])

        grammar: Grammar = Grammar(terminals, non_terminals, production_rules, "formula")
        parser: Parser = Parser(grammar)
        parser.parse("FOR_ALL VAR1 ( ( VAR1 EQUALS VAR1 ) IFF PREDICATE[2] ( VAR1 , VAR1 ) )".split())


    main()

"""
formula -> ( formula bin_op formula )
formula -> ( variable EQUALS variable )
formula -> PREDICATE[1] ( variable )
formula -> PREDICATE[2] ( variable , variable )
formula -> NOT formula
formula -> FOR_ALL variable formula
formula -> EXISTS variable formula

bin_op -> AND
bin_op -> OR
bin_op -> IMPLIES
bin_op -> IFF

variable -> VAR1
variable -> VAR2
constant -> CONST1
constant -> CONST2
"""
