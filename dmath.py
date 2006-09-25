import decimal
from decimal import Decimal, getcontext

D = Decimal


def pi():
    """Compute Pi to the current precision."""
    getcontext().prec += 2
    lasts, t, s, n, na, d, da = 0, D(3), 3, 1, 0, 0, 24
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
    return  +((1 + D(5).sqrt()) / 2)

def exp(x):
    """Return e raised to the power of x.  Result type matches input type.

    >>> print exp(Decimal(1))
    2.718281828459045235360287471
    >>> print exp(Decimal(2))
    7.389056098930650227230427461
    >>> print exp(2.0)
    7.38905609893
    >>> print exp(2+0j)
    (7.38905609893+0j)
    
    """
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s    
        i += 1
        fact *= i
        num *= x     
        s += num / fact   
    getcontext().prec -= 2        
    return +s

def cos(x):
    """Return the cosine of x as measured in radians.

    >>> print cos(Decimal('0.5'))
    0.8775825618903727161162815826
    >>> print cos(0.5)
    0.87758256189
    >>> print cos(0.5+0j)
    (0.87758256189+0j)
    
    """
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s    
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign 
    getcontext().prec -= 2        
    return +s

def sin(x):
    """Return the sine of x as measured in radians.

    >>> print sin(Decimal('0.5'))
    0.4794255386042030002732879352
    >>> print sin(0.5)
    0.479425538604
    >>> print sin(0.5+0j)
    (0.479425538604+0j)
    
    """
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

def cosh(x):
    """Return the hyperbolic cosine of Decimal x."""
    if x == 0:
        return D(1)
    
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

def sinh(x):
    """Return the hyperbolic sine of Decimal x."""
    if x == 0:
        return D(0)
    
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

# The version below is actually overwritten by the version using atan2 below
# it, since it is much faster. If possible, I'd like to write a fast version
# independent of atan2.
def asin(x):
    """Return the arc sine (measured in radians) of Decimal x."""
    if abs(x) > 1:
        raise ValueError("Domain error: asin accepts -1 <= x <= 1")
    
    if x == -1:
        return pi() / -2
    elif x == 0:
        return D(0)
    elif x == 1:
        return pi() / 2
    
    getcontext().prec += 2
    one_half = D('0.5')
    i, lasts, s, gamma, fact, num = D(0), 0, x, 1, 1, x
    while s != lasts:
        lasts = s
        i += 1
        fact *= i
        num *= x * x
        gamma *= i - one_half
        coeff = gamma / ((2 * i + 1) * fact)
        s += coeff * num
    getcontext().prec -= 2
    return +s

# This is way faster, I wonder if there's a downside?
def asin(x):
    """Return the arc sine (measured in radians) of Decimal x."""
    if abs(x) > 1:
        raise ValueError("Domain error: asin accepts -1 <= x <= 1")
    
    return atan2(x, D.sqrt(1 - x ** 2))

# The version below is actually overwritten by the version using atan2 below
# it, since it is much faster. If possible, I'd like to write a fast version
# independent of atan2.
def acos(x):
    """Return the arc cosine (measured in radians) of Decimal x."""
    if abs(x) > 1:
        raise ValueError("Domain error: acos accepts -1 <= x <= 1")
    
    if x == -1:
        return pi()
    elif x == 0:
        return pi() / 2
    elif x == 1:
        return D(0)
    
    getcontext().prec += 2
    one_half = D('0.5')
    i, lasts, s, gamma, fact, num = D(0), 0, pi() / 2 - x, 1, 1, x
    while s != lasts:
        lasts = s
        i += 1
        fact *= i
        num *= x * x
        gamma *= i - one_half
        coeff = gamma / ((2 * i + 1) * fact)
        s -= coeff * num
    getcontext().prec -= 2
    return +s

# This is way faster, I wonder if there's a downside?
def acos(x):
    """Return the arc cosine (measured in radians) of Decimal x."""
    if abs(x) > 1:
        raise ValueError("Domain error: acos accepts -1 <= x <= 1")
    
    getcontext().prec += 1
    a =  pi() / 2 - atan2(x, D.sqrt(1 - x ** 2))
    getcontext().prec -= 1
    return +a

def tan(x):
    """Return the tangent of Decimal x (measured in radians)."""
    t = sin(x) / cos(x)
    return +t

def tanh(x):
    """Return the hyperbolic tangent of Decimal x."""
    t = sinh(x) / cosh(x)
    return +t

def atan(x):
    """Return the arc tangent (measured in radians) of Decimal x."""
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

def sign(x):
    """Return -1 for negative numbers and 1 for positive numbers."""
    return 2 * D(x >= 0) - 1

def atan2(y, x):
    """Return the arc tangent (measured in radians) of y/x.
    Unlike atan(y/x), the signs of both x and y are considered.
    """
    abs_y = abs(y)
    abs_x = abs(x)
    y_is_real = abs_y != D('Inf')
    
    if x:
        if y_is_real:
            a = y and atan(y / x) or D(0)
            if x < 0:
                a += sign(y) * pi()
            return a
        elif abs_y == abs_x:
            x = sign(x)
            y = sign(y)
            return pi() * (D(2) * abs(x) - x) / (D(4) * y)
    if y:
        return atan(sign(y) * D('Inf'))
    elif x < 0:
        return sign(y) * pi()
    else:
        return D(0)

def log(x, base=None):
    """log(x[, base]) -> the logarithm of Decimal x to the given Decimal base.
    If the base not specified, returns the natural logarithm (base e) of x.
    """
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

    lasts, s = 0, Decimal(repr(approx))
    while lasts != s:
        lasts = s
        s = s - 1 + x / exp(s)
    s /= log_base
    getcontext().prec -= 2
    return +s

def log10(x):
    """log10(x) -> the base 10 logarithm of Decimal x."""
    return log(x, D(10))
