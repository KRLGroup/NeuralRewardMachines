import random
from graphviz import Source
import sys
from pythomata import SimpleDFA
from flloat.parser.ltlf import LTLfParser
import numpy as np
from itertools import product

class DFA:
  def __init__(self, arg1, arg2, arg3, dictionary_symbols = None):
      if dictionary_symbols == None:
          self.dictionary_symbols = list(range(self.num_of_symbols))
      else:
          self.dictionary_symbols = dictionary_symbols
      if isinstance(arg1, str):
          self.init_from_ltl(arg1, arg2, arg3, dictionary_symbols)
      elif isinstance(arg1, int):
          self.random_init(arg1, arg2)
      elif isinstance(arg1, dict):
          self.init_from_transacc(arg1, arg2)
      else:
          raise Exception("Uncorrect type for the argument initializing th DFA: {}".format(type(arg1)))


  def init_from_ltl(self, ltl_formula, num_symbols, formula_name, dictionary_symbols):

      #From LTL to DFA
      parser = LTLfParser()
      ltl_formula_parsed = parser(ltl_formula)
      dfa = ltl_formula_parsed.to_automaton()
      # print the automaton
    #   graph = dfa.to_graphviz()
    #   graph.render("symbolicDFAs/"+formula_name)

      #From symbolic DFA to simple DFA
      print(dfa.__dict__)
      self.alphabet = ["c" + str(i) for i in range(num_symbols)]
      self.transitions = self.reduce_dfa(dfa)
      print(self.transitions)
      self.num_of_states = len(self.transitions)
      self.acceptance = []
      for s in range(self.num_of_states):
          if s in dfa._final_states:
              self.acceptance.append(True)
          else:
              self.acceptance.append(False)
      #print(self.acceptance)

      #Complete the transition function with the symbols of the environment that ARE NOT in the formula
      self.num_of_symbols = len(dictionary_symbols)
      self.alphabet = []
      for a in range(self.num_of_symbols):
        self.alphabet.append( a )
      if len(self.transitions[0]) < self.num_of_symbols:
          for s in range(self.num_of_states):
              for sym in  self.alphabet:
                  if sym not in self.transitions[s].keys():
                      self.transitions[s][sym] = s
      #print("Complete transition function")
      #print(self.transitions)
    #   self.write_dot_file("simpleDFAs/{}.dot".format(formula_name))

  def reduce_dfa(self, pythomata_dfa):
      dfa = pythomata_dfa

      admissible_transitions = []
      for true_sym in self.alphabet:
          trans = {}
          for i, sym in enumerate(self.alphabet):
              trans[sym] = False
          trans[true_sym] = True
          admissible_transitions.append(trans)
      red_trans_funct = {}
      for s0 in dfa._states:
          red_trans_funct[s0] = {}
          transitions_from_s0 = dfa._transition_function[s0]
          for key in transitions_from_s0:
              label = transitions_from_s0[key]
              for sym, at in enumerate(admissible_transitions):
                  if label.subs(at):
                      red_trans_funct[s0][sym] = key

      return red_trans_funct
  def init_from_transacc(self, trans, acc):
      self.num_of_states = len(acc)
      self.num_of_symbols = len(trans[0])
      self.transitions = trans
      self.acceptance = acc

      self.alphabet = []
      for a in range(self.num_of_symbols):
        self.alphabet.append( a )

  def random_init(self, numb_of_states, numb_of_symbols):
      self.num_of_states = numb_of_states
      self.num_of_symbols = numb_of_symbols
      transitions= {}
      acceptance = []
      for s in range(numb_of_states):
          trans_from_s = {}
          #Each state is equiprobably set to be accepting or rejecting
          acceptance.append(bool(random.randrange(2)))
          #evenly choose another state from [i + 1; N ] and adds a random-labeled transition
          if s < numb_of_states - 1:
              s_prime = random.randrange(s + 1 , numb_of_states)
              a_start = random.randrange(numb_of_symbols)

              trans_from_s[a_start] = s_prime
          else:
              a_start = None
          for a in range(numb_of_symbols):
              #a = str(a)
              if a != a_start:
                  trans_from_s[a] = random.randrange(numb_of_states)
          transitions[s] = trans_from_s.copy()

      self.transitions = transitions
      self.acceptance = acceptance
      self.alphabet = ""
      for a in range(numb_of_symbols):
        self.alphabet += str(a)

  def accepts(self, string):
    if string == '':
      return self.acceptance[0]
    return self.accepts_from_state(0, string)

  def accepts_from_state(self, state,string):
    assert string != ''

    a = string[0]
    next_state = self.transitions[state][a]

    if len(string) == 1:
      return self.acceptance[next_state]

    return self.accepts_from_state(next_state, string[1:])

  def to_pythomata(self):
      trans = self.transitions
      acc = self.acceptance
      #print("acceptance:", acc)
      accepting_states = set()
      for i in range(len(acc)):
              if acc[i]:
                  accepting_states.add(i)

      automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

      return automaton

  def write_dot_file(self, file_name):
      with open(file_name, "w") as f:
          f.write(
              "digraph MONA_DFA {\nrankdir = LR;\ncenter = true;\nsize = \"7.5,10.5\";\nedge [fontname = Courier];\nnode [height = .5, width = .5];\nnode [shape = doublecircle];")
          for i, rew in enumerate(self.acceptance):
                  if rew:
                      f.write(str(i) + ";")
          f.write("\nnode [shape = circle]; 0;\ninit [shape = plaintext, label = \"\"];\ninit -> 0;\n")

          for s in range(self.num_of_states):
              for a in range(self.num_of_symbols):
                  s_prime = self.transitions[s][a]
                  f.write("{} -> {} [label=\"{}\"];\n".format(s, s_prime, self.dictionary_symbols[a]))
          f.write("}\n")

      s = Source.from_file(file_name)
      s.view()

class MooreMachine(DFA):
    def __init__(self, arg1, arg2, arg3, reward="acceptance", dictionary_symbols=None):
        # Initialize the base DFA first
        super().__init__(arg1, arg2, arg3, dictionary_symbols)
        # Initialize rewards and compute them according to the chosen scheme
        self.rewards = [100 for _ in range(self.num_of_states)]
        self.calculate_absorbing_states()
        if reward == "distance":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 0
            #print(self.rewards)
            old_rew = self.rewards.copy()
            termination = False
            while not termination:
                termination = True
                for s in range(self.num_of_states):
                    if not self.acceptance[s]:
                        next = [ self.rewards[self.transitions[s][sym]] for sym in self.alphabet if self.transitions[s][sym] != s]
                        if len(next) > 0:
                            self.rewards[s] = 1 + min(next)

                termination = (str(self.rewards) == str(old_rew))
                old_rew = self.rewards.copy()

            for i in range(len(self.rewards)):
                self.rewards[i] *= -1
            minimum = min([r for r in self.rewards if r != -100])
            for i, r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = (r - minimum)

            maximum = max(self.rewards )
            #max : 100 = rew : x
            #x = 100 * rew / max
            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = 100 * r/ maximum
            print("REWARDS:", self.rewards)
            #assert False
        elif reward == "acceptance":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 1
                else:
                    self.rewards[s] = 0
        elif reward == "three_value_acceptance":
            for q in range(self.num_of_states):
                #neutral states
                if q not in self.absorbing_states:
                    self.rewards[q] = 0
                else:
                    #winning state
                    if self.acceptance[q]:
                        self.rewards[q] = 1
                    #failure state
                    else:
                        self.rewards[q] = -1
        else:
            raise Exception("Reward based on '{}' NOT IMPLEMENTED, choose between ['acceptance', 'three_value_acceptance', 'distance']".format(reward))
        
    def calculate_absorbing_states(self):
        self.absorbing_states = []
        for q in range(self.num_of_states):
            absorbing = True
            for s in self.transitions[q].keys():
                    absorbing = absorbing & (self.transitions[q][s] == q)
            if absorbing:
                self.absorbing_states.append(q)