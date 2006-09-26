"""
dmath v0.9.1

Python math module for Decimal numbers.  All functions should return Decimal
numbers.  Probably only works with real numbers.

pi, exp, cos, sin from Decimal recipes at http://docs.python.org/lib/decimal-recipes.html

float_to_decimal from Decimal FAQ at http://docs.python.org/lib/decimal-faq.html

Copyright (c) 2006 Brian Beck <exogen@gmail.com> and Christopher Hesse <christopher.hesse@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# TODO all inputs should be converted using convert_other to Decimal, and all results should be returned as Decimal (don't bother matching input types)
# TODO context should be taken as an argument when appropriate, especially when throwing an error, look at decimal.py for hints
# TODO should use custom convert_other that has the option of converting floats (using float_to_decimal) if an option is set in advance (just not by default)
# TODO try implementing something, say pi, in pyrex to compare the speed

import math
import decimal
from decimal import Decimal, getcontext, setcontext
from decimal import _convert_other

D = Decimal

#
# utility functions
#

def float_to_decimal(f):
    "Convert a floating point number to a Decimal with no loss of information"
    # Transform (exactly) a float to a mantissa (0.5 <= abs(m) < 1.0) and an
    # exponent.  Double the mantissa until it is an integer.  Use the integer
    # mantissa and exponent to compute an equivalent Decimal.  If this cannot
    # be done exactly, then retry with more precision.

    mantissa, exponent = math.frexp(f)
    while mantissa != int(mantissa):
        mantissa *= 2.0
        exponent -= 1
    mantissa = int(mantissa)

    oldcontext = getcontext()
    setcontext(Context(traps=[Inexact]))
    try:
        while True:
            try:
               return mantissa * Decimal(2) ** exponent
            except Inexact:
                getcontext().prec += 1
    finally:
        setcontext(oldcontext)

#
# constants
#

def pi():
    """Compute Pi to the current precision."""
    getcontext().prec += 2
    lasts = 0; t = D(3); s = 3; n = 1; na = 0; d = 0; da = 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s

def e():
    """Compute the base of the natural logarithm to the current precision."""
    return exp(D(1))

def golden_ratio():
    """Calculate the golden ratio to the current precision."""
    return  (1 + D(5).sqrt()) / 2

#
# transcendental functions
#

def exp(x):
    """Return e raised to the power of x."""
    getcontext().prec += 2
    i = 0; lasts = 0; s = 1; fact = 1; num = 1
    while s != lasts:
        lasts = s    
        i += 1
        fact *= i
        num *= x     
        s += num / fact   
    getcontext().prec -= 2        
    return +s

def log(x, base=None):
    """Return the logarithm of x to the given base.
    If the base not specified, returns the natural logarithm (base e) of x.
    """
    # TODO make sure log(e) = 1
    
    if x < 0:
        return D('NaN')
    elif base == 1:
        raise ValueError("Base was 1!")
    elif x == base:
        return D(1)
    elif x == 0:
        return D('-Inf')
    
    getcontext().prec += 2    
    
    if base is None:
        log_base = 1
        approx = math.log(x)
    else:
        log_base = log(base)
        approx = math.log(x, base)

    lasts, s = 0, D(repr(approx))
    while lasts != s:
        lasts = s
        s -=  1 - x / exp(s)
    s /= log_base
    getcontext().prec -= 2
    return +s

def log10(x):
    """Return the base 10 logarithm of x."""
    return log(x, D(10))

#
# trigonometric functions
#

def sin(x):
    """Return the sine of x in radians."""
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s    
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s

def cos(x):
    """Return the cosine of x in radians."""
    # uses the series definition of cos, see
    # http://en.wikipedia.org/wiki/Trigonometric_function#Series_definitions
    getcontext().prec += 2
    i = 0; lasts = 0; s = 1; fact = 1; num = 1; sign = 1
    while s != lasts:
        lasts = s    
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign = -sign
        s += num / fact * sign 
    getcontext().prec -= 2        
    return +s

def tan(x):
    """Return the tangent of x in radians."""
    return sin(x) / cos(x)

#
# inverse trigonometric functions
#

# The version below is actually overwritten by the version using atan2 below
# it, since it is much faster. If possible, I'd like to write a fast version
# independent of atan2.
#def asin(x):
#    """Return the arc sine (measured in radians) of Decimal x."""
#    if abs(x) > 1:
#        raise ValueError("Domain error: asin accepts -1 <= x <= 1")
#    
#    if x == -1:
#        return pi() / -2
#    elif x == 0:
#        return D(0)
#    elif x == 1:
#        return pi() / 2
#    
#    getcontext().prec += 2
#    one_half = D('0.5')
#    i, lasts, s, gamma, fact, num = D(0), 0, x, 1, 1, x
#    while s != lasts:
#        lasts = s
#        i += 1
#        fact *= i
#        num *= x * x
#        gamma *= i - one_half
#        coeff = gamma / ((2 * i + 1) * fact)
#        s += coeff * num
#    getcontext().prec -= 2
#    return +s

# This is way faster, I wonder if there's a downside?
def asin(x):
    """Return the arcsine of x in radians."""
    if abs(x) > 1:
        raise ValueError("Domain error: asin accepts -1 <= x <= 1")
    
    if x == -1:
        return pi() / -2
    elif x == 0:
        return D(0)
    elif x == 1:
        return pi() / 2
    
    return atan2(x, D.sqrt(1 - x ** 2))

# The version below is actually overwritten by the version using atan2 below
# it, since it is much faster. If possible, I'd like to write a fast version
# independent of atan2.
#def acos(x):
#    """Return the arc cosine (measured in radians) of Decimal x."""
#    if abs(x) > 1:
#        raise ValueError("Domain error: acos accepts -1 <= x <= 1")
#    
#    if x == -1:
#        return pi()
#    elif x == 0:
#        return pi() / 2
#    elif x == 1:
#        return D(0)
#    
#    getcontext().prec += 2
#    one_half = D('0.5')
#    i, lasts, s, gamma, fact, num = D(0), 0, pi() / 2 - x, 1, 1, x
#    while s != lasts:
#        lasts = s
#        i += 1
#        fact *= i
#        num *= x * x
#        gamma *= i - one_half
#        coeff = gamma / ((2 * i + 1) * fact)
#        s -= coeff * num
#    getcontext().prec -= 2
#    return +s

# This is way faster, I wonder if there's a downside?
def acos(x):
    """Return the arccosine of x in radians."""
    if abs(x) > 1:
        raise ValueError("Domain error: acos accepts -1 <= x <= 1")

    if x == -1:
        return pi()
    elif x == 0:
        return pi() / 2
    elif x == 1:
        return D(0)
    
    return pi() / 2 - atan2(x, D.sqrt(1 - x ** 2))

def atan(x):
    """Return the arctangent of x in radians."""
    if x == D('-Inf'):
        return pi() / -2
    elif x == 0:
        return D(0)
    elif x == D('Inf'):
        return pi() / 2
    
    if x < -1:
        c = pi() / -2
        x = 1 / x
    elif x > 1:
        c = pi() / 2
        x = 1 / x
    else:
        c = 0
    
    getcontext().prec += 2
    x_squared = x ** 2
    y = x_squared / (1 + x_squared)
    y_over_x = y / x
    i, lasts, s, coeff, num = D(0), 0, y_over_x, 1, y_over_x
    while s != lasts:
        lasts = s 
        i += 2
        coeff *= i / (i + 1)
        num *= y
        s += coeff * num
    if c:
        s = c - s
    getcontext().prec -= 2
    return +s

def atan2(y, x):
    """Return the arctangent of y/x in radians.
    Unlike atan(y/x), the signs of both x and y are considered.
    """
# TODO check the sign function make sure this still works
# decimal zero has a sign
    abs_y = abs(y)
    abs_x = abs(x)
    y_is_real = (abs_y != D('Inf'))
    
    if x != 0:
        if y_is_real:
            a = y and atan(y / x) or D(0)
            if x < 0:
                a += sign(y) * pi()
            return a
        elif abs_y == abs_x:
            x = sign(x)
            y = sign(y)
            return pi() * (D(2) * abs(x) - x) / (D(4) * y)

    if y != 0:
        return atan(sign(y) * D('Inf'))
    elif x < 0:
        return sign(y) * pi()
    else:
        return D(0)

#
# hyperbolic trigonometric functions
#

def sinh(x):
    """Return the hyperbolic sine of x."""
    if x == 0:
        return D(0)
    
    # uses the taylor series expansion of sinh, see
    # http://en.wikipedia.org/wiki/Hyperbolic_function#Taylor_series_expressions
    getcontext().prec += 2
    i, lasts, s, fact, num = 1, 0, x, 1, x
    while s != lasts:
        lasts = s
        i += 2
        num *= x * x
        fact *= i * (i - 1)
        s += num / fact
    getcontext().prec -= 2
    return +s

def cosh(x):
    """Return the hyperbolic cosine of x."""
    if x == 0:
        return D(1)
    
    # uses the taylor series expansion of cosh, see
    # http://en.wikipedia.org/wiki/Hyperbolic_function#Taylor_series_expressions
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        num *= x * x
        fact *= i * (i - 1)
        s += num / fact
    getcontext().prec -= 2
    return +s

def tanh(x):
    """Return the hyperbolic tangent of x."""
    return +(sinh(x) / cosh(x))

#
# miscellaneous functions
#

def sgn(x):
    """Return -1 for negative numbers, 1 for positive numbers and 0 for zero."""
    # the signum function, see:
    # http://en.wikipedia.org/wiki/Sign_function
    if x > 0:
        return D(1)
    elif x < 0:
        return D(-1)
    else:
        return D(0)

def degrees(x):
    """Return angle x converted from radians to degrees."""
    return x * 180 / pi()

def radians(x):
    """Return angle x converted from degrees to radians."""
    return x * pi() / 180

def ceil(x):
    """Return the smallest integral value >= x."""
    return x.to_integral(rounding=decimal.ROUND_CEILING)

def floor(x):
    """Return the largest integral value <= x."""
    return x.to_integral(rounding=decimal.ROUND_FLOOR)

def hypot(x, y):
    """Return the Euclidean distance, sqrt(x**2 + y**2)."""
    return sqrt(x * x + y * y)

def modf(x):
    """Return the fractional and integer parts of x."""
    int_part = x.to_integral(rounding=decimal.ROUND_FLOOR)
    frac_part = x-int_part
    return frac_part,int_part

def ldexp(s, e):
    """Return s*(10**e), the value of a decimal floating point number with
    significand s and exponent e.  This function is the inverse of frexp. Note
    that this is different from math.ldexp, which uses 2**e instead of
    10**e."""
    return s*(10**e)

def frexp(x):
    """Return s and e where s*(10**e) == x.  s and e are the significand and
    exponent, respectively of x.  This function is the inverse of ldexp. Note
    that this is different from math.frexp, which uses 2**e instead of
    10**e."""
    e = D(x.adjusted())
    s = D(10)**(-x.adjusted())*x
    return s, e

def pow(x, y, context=None):
    context, x, y = _initialize(context, x, y)
    # if y is an integer, just call regular pow
    if y._isinteger():
        return x**y
    # if x is negative, the result is complex
    if x < 0:
        return context._raise_error(decimal.InvalidOperation, 'x (negative) ** y (fractional)')
    return exp(y * log(x))

def tetrate(x, y, context=None):
    """returns x recursively raised to the power of x, y times.
    y must be a natural number
    
    ;)"""
    context, x, y = _initialize(context, x, y)

    if not y._isinteger():
        return context._raise_error(decimal.InvalidOperation, 'x *** (non-integer)')

    def _tetrate(x,y):
        if y == -1:
            return D(-1)
        if y == 0:
            return D(1)
        if y == 1:
            return x
        return x**_tetrate(x,y-1)

    return _tetrate(x,y)

#
# internal functions
#

def _initialize(context, *args):
    if context is None:
        context = getcontext()
    
    r = [context]
    for arg in args:
# TODO should something else be seeing NotImplemented?
        e = _convert_other(arg)
        if e is NotImplemented:
            raise TypeError, "unsupported operand type: '%s' (if it's a float, try the float_to_decimal function)" % (type(e).__name__,)
        r.append(e)
    
    return r

def _sign(x):
    """Return -1 for negative numbers and 1 for positive numbers."""
    # brian's sign function
    if x._sign == 0:
        return D(1)
    else:
        return D(-1)

sqrt = D.sqrt
fabs = abs
fmod = D.__mod__

__all__ = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'degrees',
'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'golden_ratio', 'hypot', 'ldexp',
'log', 'log10', 'modf', 'pi', 'pow', 'radians', 'sgn', 'sin', 'sinh', 'sqrt',
'tan', 'tanh', 'tetrate']

if __name__ == '__main__':
    # TODO put some test functions down here
    pass
