formulas = []

items = ['pickaxe', 'lava', 'door', 'gem', 'empty' ]

#PATTERNS INSPIRED FROM LTL2action
formulas.append(("(F c0) & (F c1)", 2, "task1: visit({0}, {1})".format(*items)))
formulas.append(("(F c0) & (F c1) & (F c2)", 3, "task2: visit({0}, {1}, {2})".format(*items)))
formulas.append(("F(c0 & F(c1))", 2, "task3: seq_visit({0}, {1})".format(*items)))
formulas.append(("F(c0 & F(c1)) & F(c2 & F(c3))", 4, "task4: seq_visit({0}, {1}) + seq_visit({2}, {3})".format(*items)))
formulas.append(("F(c0 & F(c1)) & (F c2)", 4, "task5: seq_visit({0}, {1}) + visit({2})".format(*items)))
formulas.append(("F(c0 & F(c1)) & (F c2) & (F c3)", 4, "task6: seq_visit({0}, {1}) + visit({2}, {3})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2))", 3, "task7: visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2)) & (G(! c3))", 4, "task8: visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2)", 3, "task9: seq_visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2) & G(! c3)", 4, "task10: seq_visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))

#TODO:
#RM+A2C symb_env            (V)
#RNN+A2C symb_env           (entro 28/12)
#grounding+A2C symb_env     (entro 28/12)

#RM+DDQN symb_env           (entro 28/12)
#grounding+A2C symb_env     (entro 04/01)

#RM+A2C image_env           (entro 04/01)
#RNN+A2C image_env          (entro 04/01)
#grounding+A2C image_env    (entro 11/01)

#RM+DDQN image_env           (entro 11/01)
#grounding+A2C image_env     (entro 11/01)
