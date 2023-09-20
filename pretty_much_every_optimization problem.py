import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import stats as st
from scipy import optimize
import time          

T = 9
beta = 0.9
kv = np.zeros(T+1,float)
cv = np.zeros(T+1,float)
uv = np.zeros(T+1,float)
kv[0] = 100  # k0
cv[0] = (1.0-beta)/(1.0-beta**(T+1)) * kv[0]  # c0
uv[0] = np.log(cv[0])

for i in range(1,T+1):
    #print "i=" + str(i)
    cv[i] = beta * cv[i-1]
    kv[i] = kv[i-1] - cv[i-1]

    # Period utility with discounting
    uv[i] = beta**(i-1)*np.log(cv[i])

np.sum(uv)  # total utility

print("cv = " + str(cv))
print("kv = " + str(kv))

fig, ax = plt.subplots(2,1)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
#
ax[0].plot(cv, '-o')
ax[0].set_ylabel(r'$c^*_t$')
ax[0].set_xlabel('Period t')
ax[0].set_title('Optimal Consumption')
#
ax[1].plot(kv, '-o')
ax[1].set_ylabel(r'$k_t$')
ax[1].set_xlabel('Period t')
ax[1].set_title('Cake size')
#
plt.show()



def func1(cv):
    T = len(cv)
    uv= np.zeros(T,float)
    for i in range(T):
        beta = 0.9
        # Period utility with discounting
        uv[i] = (beta**i) * np.log(cv[i])

    # We want to maximize this welfare,
    # so we need to 'negate' the result
    return (-np.sum(uv))



def constr1(cv):
    k0 = 100
    z1 = np.sum(cv) - k0
    return np.array([z1])





T = 10

# Starting guesses for the optimal consumption vector
c0v = np.ones(T,float)*0.1

coptv = optimize.fmin_slsqp(func1, c0v, f_eqcons = constr1)

print(coptv)

fig, ax = plt.subplots()
# Plot analytical and numerical solution
ax.plot(np.arange(0,T), cv, 'b-o', np.arange(0,T), coptv, 'r--o')
ax.set_title("Optimal consumption")
ax.set_xlabel("Period t")
ax.set_ylabel("c_t")
# Create a legend
ax.legend(['analytical', 'numerical'], loc='best', shadow=True)
print('-----------------------------')
print('Analytic solution')
print('cv = {}'.format(cv))
print('-----------------------------')
print(' ')
print('-----------------------------')
print('Numeric solution')
print('cv = {}'.format(coptv))
print('-----------------------------')




#defingn profit


def f_profit(x):
    if (x < 0):
        return (0)
    if (x == 0):
        return (np.nan)
    y = np.exp(-2*x)
    return (4 * x**2 * y)




import matplotlib.pyplot as plt
#Newton Method


xmin = -1.0
xmax = 1.0
xv = np.linspace(xmin, xmax, 200)
fx = np.zeros(len(xv),float) # define column vector
for i in range(len(xv)):
    fx[i] = f_profit(xv[i])

fig, ax = plt.subplots()
ax.plot(xv, fx)
ax.plot(xv, f_profit(1)*np.ones(len(xv)))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Profit Function')
plt.show()



def f_profit_plus_deriv(x):
    # gamma(2,3) density
    if (x < 0):
        return np.array([0, 0, 0])
    if (x == 0):
        return np.array([0, 0, np.nan])
    y = np.exp(-2.0*x)
    return np.array([4.0 * x**2.0 * np.exp(-2.0*x), \
      8.0 * np.exp(-2.0*x) * x*(1.0-x), \
      8.0 * np.exp(-2.0*x) * (1 - 4*x + 2 * x**2)])



xmin = -1.0
xmax = 1.0
xv = np.linspace(xmin, xmax, 200)
dfx = np.zeros(len(xv),float) # define column vector
for i in range(len(xv)):

    # The derivate value is in the second position
    dfx[i] = f_profit_plus_deriv(xv[i])[1]

fig, ax = plt.subplots()
ax.plot(xv, dfx)
ax.plot(xv, 0*np.ones(len(xv)))
ax.set_xlabel('x')
ax.set_ylabel('f\'(x)')
ax.set_title('Derivative of Profit Function')
plt.show()


def newton(f3, x0, tol = 1e-9, nmax = 100):
    # Newton's method for optimization, starting at x0
    # f3 is a function that given x returns the vector
    # (f(x), f'(x), f''(x)), for some f
    x = x0
    f3v = f3(x)
    n = 0
    while ((abs(f3v[1]) > tol) and (n < nmax)):
        # 1st derivative value is the second return value
        # 2nd derivative value is the third return value
        x = x - f3v[1]/f3v[2]
        f3v = f3(x)
        n = n + 1
    if (n == nmax):
        print("newton failed to converge")
    else:
        return(x)


print(" -----------------------------------")
print(" Newton results ")
print(" -----------------------------------")
print(newton(f_profit_plus_deriv, 0.25))
print(newton(f_profit_plus_deriv, 0.5))
print(newton(f_profit_plus_deriv, 0.75))
print(newton(f_profit_plus_deriv, 1.75))



def gsection(ftn, xl, xm, xr, tol = 1e-9):
    # applies the golden-section algorithm to maximise ftn
    # we assume that ftn is a function of a single variable
    # and that x.l < x.m < x.r and ftn(x.l), ftn(x.r) <= ftn(x.m)
    #
    # the algorithm iteratively refines x.l, x.r, and x.m and
    # terminates when x.r - x.l <= tol, then returns x.m
    # golden ratio plus one
    gr1 = 1 + (1 + np.sqrt(5))/2
    #
    # successively refine x.l, x.r, and x.m
    fl = ftn(xl)
    fr = ftn(xr)
    fm = ftn(xm)
    while ((xr - xl) > tol):
        if ((xr - xm) > (xm - xl)):
            y = xm + (xr - xm)/gr1
            fy = ftn(y)
            if (fy >= fm):
                xl = xm
                fl = fm
                xm = y
                fm = fy
            else:
                xr = y
                fr = fy
        else:
            y = xm - (xm - xl)/gr1
            fy = ftn(y)
            if (fy >= fm):
                xr = xm
                fr = fm
                xm = y
                fm = fy
            else:
                xl = y
                fl = fy
    return(xm)




print(" -----------------------------------")
print(" Golden section results ")
print(" -----------------------------------")
myOpt = gsection(f_profit, 0.1, 0.25, 1.3)
print(gsection(f_profit, 0.1, 0.25, 1.3))
print(gsection(f_profit, 0.25, 0.5, 1.7))
print(gsection(f_profit, 0.6, 0.75, 1.8))
print(gsection(f_profit, 0.0, 2.75, 5.0))


def f_profitNeg(x):
    # gamma(2,3) density
    if (x < 0):
        return (0)
    if (x == 0):
        return (np.nan)
    y = np.exp(-2*x)
    return (-(4 * x**2 * y))





from scipy.optimize import fmax

print(" -----------------------------------")
print(" fmax results ")
print(" -----------------------------------")
print(fmin(f_profitNeg, 0.25))
print(fmin(f_profitNeg, 0.5))
print(fmin(f_profitNeg, 0.75))
print(fmin(f_profitNeg, 1.75))



#minimization

def func(x):
    s = np.log(x) - np.exp(-x)  # function: f(x)
    return s


from scipy.optimize import root

guess = 2
result = root(func, guess) # starting from x = 2
print(" ")
print(" -------------- Root ------------")
myroot = result.x  # Grab number from result dictionary
print("The root of d_func is at {}".format(myroot))
print("The max value of the function is {}".format(myOpt))




#### Root Min
def func_root_min(x):
    s = np.log(x) - np.exp(-x)  # function: f(x)
    return (s**2)



from scipy.optimize import minimize

guess = 2 # starting guess x = 2
result = minimize(func_root_min, guess, method='Nelder-Mead')

print(" ")
print("-------------- Root ------------")
myroot = result.x  # Grab number from result dictionary
print("The root of func is at {}".format(myroot))



from scipy.optimize import minimize

guess = 2 # starting guess x = 2
result = minimize(func_root_min, guess, method='Nelder-Mead')

print(" ")
print("-------------- Root ------------")
myroot = result.x  # Grab number from result dictionary
print("The root of func is at {}".format(myroot))


#Multivariate optimization


def f3simple(x):
    a = x[0]**2/2.0 - x[1]**2/4.0
    b = 2*x[0] - np.exp(x[1])
    f = np.sin(a)*np.cos(b)
    return(f)


def f3simpleNeg(x):
    a = x[0]**2/2.0 - x[1]**2/4.0
    b = 2*x[0] - np.exp(x[1])
    f = -np.sin(a)*np.cos(b)
    return(f)


def f3(x):
    a = x[0]**2/2.0 - x[1]**2/4.0
    b = 2*x[0] - np.exp(x[1])
    f = np.sin(a)*np.cos(b)
    f1 = np.cos(a)*np.cos(b)*x[0] - np.sin(a)*np.sin(b)*2
    f2 = -np.cos(a)*np.cos(b)*x[1]/2 + np.sin(a)*np.sin(b)*np.exp(x[1])
    f11 = -np.sin(a)*np.cos(b)*(4 + x[0]**2) + np.cos(a)*np.cos(b) \
        - np.cos(a)*np.sin(b)*4*x[0]
    f12 = np.sin(a)*np.cos(b)*(x[0]*x[1]/2.0 + 2*np.exp(x[1])) \
        + np.cos(a)*np.sin(b)*(x[0]*np.exp(x[1]) + x[1])
    f22 = -np.sin(a)*np.cos(b)*(x[1]**2/4.0 + np.exp(2*x[1])) \
        - np.cos(a)*np.cos(b)/2.0 - np.cos(a)*np.sin(b)*x[1]*np.exp(x[1]) \
        + np.sin(a)*np.sin(b)*np.exp(x[1])
    # Function f3 returns: f(x), f'(x), and f''(x)
    return (f, np.array([f1, f2]), np.array([[f11, f12], [f12, f22]]))



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 16))
ax = plt.gca(projection='3d')

X = np.arange(-3, 3, .1)
Y = np.arange(-3, 3, .1)
X, Y = np.meshgrid(X, Y)

Z = np.zeros((len(X),len(Y)),float)
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i][j] = f3simple([X[i][j],Y[i][j]])

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, \
    cmap=plt.cm.jet, linewidth=0, antialiased=False)
plt.show()




ax = plt.gca(projection='3d')


### Multi-Variate newton method


def newtonMult(f3, x0, tol = 1e-9, nmax = 100):
    # Newton's method for optimization, starting at x0
    # f3 is a function that given x returns the list
    # {f(x), grad f(x), Hessian f(x)}, for some f
    x = x0
    f3x = f3(x)
    n = 0
    while ((max(abs(f3x[1])) > tol) and (n < nmax)):
        x = x - np.linalg.solve(f3x[2], f3x[1])
        f3x = f3(x)
        n = n + 1
    if (n == nmax):
        print("newton failed to converge")
    else:
        return(x)




from scipy.optimize import fmin

for x0 in np.arange(1.4, 1.6, 0.1):
    for y0 in np.arange(0.4, 0.7, 0.1):
        # This algorithm requires f(x), f'(x), and f''(x)
        print("Newton: f3  " + str([x0,y0]) + ' --> ' + str(newtonMult(f3, \
            np. array([x0,y0]))))

        print("fmin: f3 " + str([x0,y0]) + ' --> ' \
            + str(fmin(f3simpleNeg, np.array([x0,y0]))))

        print(" ----------------------------------------- ")



# Formal definition of the model

# Set parameter values
N_y     = 1.0
N_o     = 1.0
alpha   = 0.3
A       = 1
beta    = 0.9
delta   = 0.0
tau_L   = 0.2
tau_K   = 0.15
t_y     = 0.0
t_o     = 0.0
#
L       = 1

# -------------------------------------------------------------
# Method 1: Root finding
# -------------------------------------------------------------
# Find x so that f(x) = 0

# Define function of capital K so that func(K) = 0

def func(K):
    s = - K + N_y\
    *((beta*(1+(1-tau_K)*(alpha*A*K**(alpha-1) - delta))* \
    ((1-tau_L)*((1-alpha)*A*K**alpha) + t_y) - t_o) \
    /((1+beta)*(1. + (1-tau_K)*(alpha*A*K**(alpha-1) - delta))))

    return s

# Plot the function to see whether it has a root-point
Kmin = 0.0001
Kmax = 0.3

# Span grid with gridpoints between Kmin and Kmax
Kv = np.linspace(Kmin, Kmax, 200)

# Output vector prefilled with zeros
fKv = np.zeros(len(Kv),float) # define column vector

for i,K in enumerate(Kv):
    fKv[i] = func(K)

#print("fK=", fK)



fig, ax = plt.subplots()
ax.plot(Kv, fKv)
# Plot horizontal line at zero in red
ax.plot(Kv, np.zeros(len(Kv)), 'r')
ax.set_title('Capital')
plt.show()




from scipy.optimize import fsolve

# Use built in 'fsolve'
print(" ")
print(" -------------- Fsolve ------------")

k_guess = 2  # our starting guess
solutionK = fsolve(func, k_guess) # starting from K = 2

# Kstar is a numpy array which does not print well
# We therefore change it into a 'pure' number
# so we can use the print format to create a nice
# looking output
Kstar = solutionK[0]

Ystar = A*Kstar**alpha*L**(1-alpha)
qstar = alpha*A*Kstar**(alpha-1)
rstar = qstar - delta
Rstar = 1. + (1-tau_K)*(qstar - delta)
wstar = (1.-alpha)*A*Kstar**alpha

# Back out solutions for the rest of the Economy
# ----------------------------------------------
# Household values
sstar = Kstar/N_y
cystar= (1.-tau_L)*wstar + t_y - sstar
costar= Rstar*sstar + t_o

# Residual gov't consumption, thrown in the ocean
Gstar = N_y*tau_L*wstar + N_o*tau_K*rstar*sstar

# Aggregate consumption
Cstar = N_y*cystar + N_o*costar

# Check the goods market condition or Aggregate resource constraint
ARC = Ystar - delta*Kstar - Cstar - Gstar

# Print results
print(" -------------------------------------")
print(" Root finding ")
print(" -------------------------------------")
print("K* = {:6.4f}".format(Kstar))
print("Y* = {:6.4f}".format(Ystar))
print("q* = {:6.4f}".format(qstar))
print("r* = {:6.4f}".format(rstar))
print("R* = {:6.4f}".format(Rstar))
print("w* = {:6.4f}".format(wstar))
print(" -------------------------------------")
print("ARC = {:6.4f}".format(ARC))



#Gauss-Seidl Algorithm


glamda  = 0.5   # updating parameter
Kold    = 0.4
jerror  = 100
iter    = 1
while (iter<200) or (jerror>0.001):
    # Solve for prices using expressions for w(K) and q(K)
    q = alpha*A*Kold**(alpha-1)
    w = (1-alpha)*A*Kold**alpha
    R = 1 + (1-tau_K)*(q - delta)
    Knew = N_y* (beta*R*((1-tau_L)*w + t_y) - t_o)/((1+beta)*R)
    # Calculate discrepancy between old and new capital stock
    jerror = abs(Kold-Knew)/Kold
    # Update capital stock
    Kold    = glamda*Knew + (1-glamda)*Kold
    iter = iter +1

# Print results
Kstar = Knew
Ystar = A*Kstar**alpha*L**(1-alpha)
wstar = w
qstar = q
Rstar = R
rstar = qstar - delta

# ------------------------------------
# Back out solutions for the rest of the Economy

# Household values
sstar = Kstar/N_y
cystar= (1-tau_L)*wstar + t_y - sstar
costar= Rstar*sstar + t_o

# Residual gov't consumption, thrown in the ocean
Gstar = N_y*tau_L*wstar + N_o*tau_K*rstar*sstar

# Aggregate consumption
Cstar = N_y*cystar + N_o*costar

# Check the goods market condition or Aggregate resource constraint
ARC = Ystar - delta*Kstar - Cstar - Gstar

print(" -------------------------------------")
print(" Gauss-Seidl ")
print(" -------------------------------------")
print("Nr. of iterations = " +str(iter))
print("K* = {:6.4f}".format(Kstar))
print("Y* = {:6.4f}".format(Ystar))
print("q* = {:6.4f}".format(qstar))
print("r* = {:6.4f}".format(rstar))
print("R* = {:6.4f}".format(Rstar))
print("w* = {:6.4f}".format(wstar))
print(" -------------------------------------")
print("ARC = {:6.4f}".format(ARC))



# Equilibrium definiton


import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import stats as st
import time            # Imports system time module to time your script

plt.close('all')  # close all open figures



# Set parameter values
N_y     = 1.0
N_o     = 1.0
alpha   = 0.3
A       = 1
beta    = 0.9
delta   = 0.0
tau_L   = 0.2
tau_K   = 0.15
t_y     = 0.0
t_o     = 0.0
#
L       = 1

# -------------------------------------------------------------
# Method 1: Root finding
# -------------------------------------------------------------
# Find x so that f(x) = 0

# Define function of capital K so that func(K) = 0

def func(K):
    s = - K + N_y\
    *((beta*(1+(1-tau_K)*(alpha*A*K**(alpha-1) - delta))* \
    ((1-tau_L)*((1-alpha)*A*K**alpha) + t_y) - t_o) \
    /((1+beta)*(1. + (1-tau_K)*(alpha*A*K**(alpha-1) - delta))))

    return s

# Plot the function to see whether it has a root-point
Kmin = 0.0001
Kmax = 0.3

# Span grid with gridpoints between Kmin and Kmax
Kv = np.linspace(Kmin, Kmax, 200)

# Output vector prefilled with zeros
fKv = np.zeros(len(Kv),float) # define column vector

for i,K in enumerate(Kv):
    fKv[i] = func(K)

#print("fK=", fK)



fig, ax = plt.subplots()
ax.plot(Kv, fKv)
# Plot horizontal line at zero in red
ax.plot(Kv, np.zeros(len(Kv)), 'r')
ax.set_title('Capital')
plt.show()



