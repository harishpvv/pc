import argparse
import numpy as np

try:
    from ase.io import read, write
except:
    pass

parser = argparse.ArgumentParser(description='Does basic math in python style')
parser.add_argument('expr', nargs='+', help='Expression')
args = parser.parse_args()

expr = args.expr
ixi=0
if expr[0]=='s':
    try:
        from sympy import *
        x, y, z = symbols('x y z')
        ixi = 1
    except:
        print('Sympy is not installed\n')

from math import *

def ar(l):
    return np.array(l)

for i in expr[ixi:]:
    try:
        print(eval(i))
    except:
        exec(i)
