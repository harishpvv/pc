# A simple terminal based scientific calculator

A simple terminal based scientific calculator is somewhat dream of mine, because whenever I want to do some calculations I have to open a calculator, which I am not happy with. There is an option called bc in linux but it can only perform very simple calculations. So I started working to create terminal based projectfor scientific calculator, it took 20 lines of code and 1/2 hour of time.

Here I am sharing simple but powerful scientific calculator which runs using python modules namely `sympy` and `math`. Here goes the application,

Only module we need to install is sympy. Which can be installed using the command,

```pip install sympy```

Sympy stands for symbolic python, as name suggests it works as hand written mathematics, e.g.

```python
> diff(x**2)
> output:  2*x

> x**0.5
> output: x**0.5
```
Let's look at the code,
```python
#! /usr/bin/python

from math import *
import argparse
from sympy import *
from sympy.plotting import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('expr', nargs='+', help='Expression')
args = parser.parse_args()

expr = args.expr

index=0
if expr[0]=='s':
    x, y, z, t = symbols('x y z t')
    f, g, h = symbols('f g h', cls=Function)
    index = 1

for i in expr[index:]:
    try:
        print(eval(i))
    except:
        exec(i)
```
To make it a terminal command, save this code as `pc`, and copy it to bin as `cp /path/to/pc /usr/bin/` in linux, or you have to add path of this file `PATH` variable in Windows.

Let's see what it can do ...

## Simple arithmetics

```
> pc 2*8
> 16
>
> pc 2/4
> 0.5
>
> pc 2**2
> 4
>
> pc 9**0.5
> 3
```
And it can do multiple calculations with one command

```
> pc 6*4 7*2 9/3
> 24
> 14
> 3.0
```
And we can define local variables
```
> pc x=5 y=6 x+y x*y x/y
> 11
> 30
> 0.8333333333333334
```
As you understood, it executes each argument as a line in python code.

## Simple trigonometry

When ever () are used, we have to put expression in `" "` 

```
> pc "sin(pi/2)"
> 1.0
>
> pc "sin(1.52)"
> 0.99999
>
> pc "sinh(1)"
> 1.17520119
>
> pc "exp(2)"
> 7.38905609
```
All the functions available in `math` module can be used, functions are listed below
```
math.acos(       math.cos(        math.factorial(  math.isclose(    math.log2(       math.tan(
math.acosh(      math.cosh(       math.floor(      math.isfinite(   math.modf(       math.tanh(
math.asin(       math.degrees(    math.fmod(       math.isinf(      math.nan         math.tau
math.asinh(      math.e           math.frexp(      math.isnan(      math.pi          math.trunc(
math.atan(       math.erf(        math.fsum(       math.ldexp(      math.pow(        
math.atan2(      math.erfc(       math.gamma(      math.lgamma(     math.radians(    
math.atanh(      math.exp(        math.gcd(        math.log(        math.sin(        
math.ceil(       math.expm1(      math.hypot(      math.log10(      math.sinh(       
math.copysign(   math.fabs(       math.inf         math.log1p(      math.sqrt(
```

## Symbolic mathematics

For doing symbolic mathematics we have to give 's' as the first argument, e.g,

```
> pc s x*x
> x**2
```
We can do pretty much every thing which we can do with [Sympy](https://docs.sympy.org/latest/tutorial/index.html) 


In sympy variables and functions have to be defined apriori, in this code `x, y, z, t` are defined as variables and `f, g, h` are defined as functions, we can use them without defining. 

By default, in sympy equations will not be solved for numerical answers, as
```
> pc s "sqrt(2)"
> sqrt(2)
```
In order to get numerical value we have to use `N()`

```
> pc s "N(sqrt(2))"
> 1.41421356
```
That is why it is better not to use sympy (do not put first argument as 's') for numerical calculations.



### Expressions
```
> pc s "exp=(x**2+y**2+2*x*y)" "exp.subs([(x,2),(y,6)])"
> 64
```
In above command subs() is for substitution.

### Simplification

```
> pc s "simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1))"
> x - 1
>
> pc s "expand((x+y)**2)"
> x**2 + 2*x*y + y**2
>
> pc s "factor(x**3 - x**2 + x - 1)"
> (x - 1)*(x**2 + 1)
>
> pc s "cancel((x**2 + 2*x + 1)/(x**2 + x))"
> (x + 1)/x
>
> pc s "apart((x-1)/((x-4)*(x+2)))"
> 1/(2*(x + 2)) + 1/(2*(x - 4))
```
apart() is for partial fractions.

### Calculus
```
> pc s "limit(sin(x)/x, x, 0)"
> 1
>
> pc s "limit(exp(-x), x, oo)"
> 0
```
'oo' is infinity in sympy.

```
> pc s "diff(x**2)"
> 2*x
>
> pc s "diff(x**2).subs(x,1)"
> 2
```

```
> pc s "integrate(x**2)"
> x**3/3
>
> pc s "integrate(x**2,(x, 3, 5))"
> 98/3
```
```
> pc s "sin(x).series(x, 0, 4)"
> x - x**3/6 + O(x**4)
```
### Solving equations

In solve method first argument is function (=0 is implied), and second is target variable 

```
> pc s "solve(x**2-5*x+6,x)"
> [2, 3]
```
### Differential equations

To solve an equation like f''(x) - 2f'(x) + f(x) = sin(x)
```
> pc s "dsolve(f(x).diff(x, x) - 2*f(x).diff(x) + f(x)-sin(x),f(x))"
> Eq(f(x), (C1 + C2*x)*exp(x) + cos(x)/2)
```
The result should be read as f(x) = second argument

### Matrices

Matrices can be created and manipulated as 
```
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m**2"
> Matrix([[-5, 12], [-8, 3]])
>
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m**-1"
> Matrix([[1/3, -1/3], [2/9, 1/9]])
>
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m.T"
> Matrix([[1, -2], [3, 3]])
>
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m.det()"
> 9
>
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m.rref()"
> (Matrix([
[1, 0],
[0, 1]]), (0, 1))
>
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m.eigenvals()"
> {2 - sqrt(5)*I: 1, 2 + sqrt(5)*I: 1}
>
> pc s "m = Matrix([[1, 3], [-2, 3]])" "m.eigenvects()"
> [(2 - sqrt(5)*I, 1, [Matrix([
[-3/(-1 + sqrt(5)*I)],
[                  1]])]), (2 + sqrt(5)*I, 1, [Matrix([
[-3/(-1 - sqrt(5)*I)],
[                  1]])])]
```

### Plotting
We can you sympy for interactive plotting

```
> pc s "plot(x**2)"
```
This will show this

![plot](https://github.com/harishpvv/pc/blob/main/x2.png)

```
> pc s "plot3d(x**2+y**2)"
```
![plot](https://github.com/harishpvv/pc/blob/main/3d.png)
