import nltk
from nltk import grammar, parse
from nltk import load_parser

grammar = nltk.CFG.fromstring("""
  O -> Suj Pred
  Suj -> Det Sust
  Det -> "el"
  Sust -> "perro" | "gato"
  Pred -> Verbo
  Verbo -> VT | VI
  VT -> "come"   
  VI -> "duerme"
""")

try:
    text = "el gato duerme"
    tokens = text.split()
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for tree in rd_parser.parse(tokens):
        print(tree)
        tree.pretty_print()
except ValueError:
    print("No se reconoce como oración del lenguaje")