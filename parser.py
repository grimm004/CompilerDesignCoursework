from typing import Set, Dict, List
import copy


class Grammar:
    def __init__(self,
                 terminals: Set[str],
                 non_terminals: Set[str],
                 production_rules: Dict[str, List[List[str]]],
                 start_symbol: str) -> None:

        assert not (terminals & non_terminals)
        assert start_symbol in non_terminals
        assert set(production_rules.keys()) == non_terminals

        self.terminals: Set[str] = terminals
        self.non_terminals: Set[str] = non_terminals
        self.production_rules = production_rules
        self.start_symbol: str = start_symbol

    def make_ll1(self) -> None:
        production_rules: Dict[str, List[List[str]]] = {}
        non_terminals: Set[str] = set()
        while production_rules != self.production_rules or non_terminals != self.non_terminals:
            production_rules = copy.deepcopy(self.production_rules)
            non_terminals = self.non_terminals.copy()

            self.epsilon_derivation_elimination()
            self.left_recursion_elimination()
            self.left_factorisation()
            print(self.production_rules)

    def epsilon_derivation_elimination(self) -> None:
        null_productions = [(non_term, i)
                            for non_term, rules in self.production_rules.items()
                            for i in range(len(rules)) if not rules[i]]

        for null_prod in null_productions:
            self.production_rules[null_prod[0]].pop(null_prod[1])
            for rules in self.production_rules.values():
                for rule in rules:
                    for symbol in rule:
                        if symbol == null_prod[0]:
                            new_rule = rule.copy()
                            new_rule.remove(symbol)
                            if new_rule:
                                rules.append(new_rule)

    def left_recursion_elimination(self) -> None:
        non_terminals: List[str] = list(self.non_terminals)

        for i in range(len(non_terminals)):
            a_i = non_terminals[i]
            for j in range(i):
                a_j = non_terminals[j]
                for k in range(len(self.production_rules[a_i])):
                    rule = self.production_rules[a_i][k]
                    if len(rule) > 0 and rule[0] == a_j:
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
                a_i_prime = "%s'" % a_i
                self.non_terminals.add(a_i_prime)
                self.production_rules[a_i] = [rule + [a_i_prime] for rule in non_left_recursive]
                self.production_rules[a_i_prime] = [rule + [a_i_prime] for rule in left_recursive]

    def left_factorisation(self) -> None:
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

            if longest_prefix:
                print("I give up")

class Parser:
    def __init__(self, grammar) -> None:
        pass


if __name__ == "__main__":
    def main() -> None:
        terminals: Set[str] = {
            "a", "b", "c"
        }

        non_terminals: Set[str] = {
            "S", "A", "B", "C"
        }

        production_rules: Dict[str, List[List[str]]] = {
            "S": [["A", "B"]],
            "A": [["a"], []],
            "B": [["C", "C"], ["A"], ["b"]],
            "C": [["c"]],
        }

        grammar: Grammar = Grammar(terminals, non_terminals, production_rules, "S")
        grammar.make_ll1()
        parser: Parser = Parser(grammar)


    main()
