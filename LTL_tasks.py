formulas = []

items = ['pickaxe', 'lava', 'door', 'gem', 'empty' ]

#PATTERNS INSPIRED FROM LTL2action
formulas.append(("(F c0) & (F c1)", 2, "task1: visit({0}, {1})".format(*items)))
formulas.append(("(F c0) & (F c1) & (F c2)", 3, "task2: visit({0}, {1}, {2})".format(*items)))
formulas.append(("F(c0 & F(c1))", 2, "task3: seq_visit({0}, {1})".format(*items)))
formulas.append(("F(c0 & F(c1)) & (F c2)", 3, "task5: seq_visit({0}, {1}) + visit({2})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2))", 3, "task7: visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2)) & (G(! c3))", 4, "task8: visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2)", 3, "task9: seq_visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2) & G(! c3)", 4, "task10: seq_visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))
