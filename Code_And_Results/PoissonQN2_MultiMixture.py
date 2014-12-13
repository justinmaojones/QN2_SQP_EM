# Written by Justin Mao-Jones 12/13/14, jmj418@nyu.edu


import numpy as np
import numpy.linalg as la
import math
import scipy.misc
import time
from functools import wraps

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mymsg = ""
        for key in kwargs.keys():
            if key == "msg":
                mymsg = kwargs[key] + ":"
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        #print func.__name__ + ":",mymsg,round(end-start,2),"seconds"
        return result,end-start
    return wrapper

class inputdata():
    def __init__(self, c):
        self.c = c
        self.array = np.array(c)
        self.count = len(c)
        J = np.arange(len(c))
        self.J = J
        self.maxJ = np.max(J)
        a = np.arange(1,np.max(J)+1).reshape(-1,1)
        b = a*(a <= J)
        self.sum_log_factorial = np.sum(np.log(np.maximum(np.ones(b.shape),b)),axis=0)
        self.sum = sum(c)
        

def printlines(line,printing):
    if printing:
        for item in line:
            print item,
        print '\n',
    
def theta_matrix(gammas,thetas):
    return np.matrix(np.hstack((np.array(gammas),np.array(thetas)))).T

def theta_list(theta):
    m,n = theta.shape
    arr = np.array(theta).ravel().tolist()
    gammas = arr[:m/2]
    thetas = arr[m/2:]
    return gammas,thetas
    
def theta_array(theta):
    m,n = theta.shape
    arr = np.array(theta).ravel()
    gammas = arr[:m/2]
    thetas = arr[m/2:]
    return gammas,thetas
    
def f_(j,theta):
    # f(x|theta)    
    gammas,thetas = theta_array(theta)
    return np.sum(gammas*np.exp(-thetas)*thetas**j/math.factorial(j))

    
def f_tilda_array2d(C,theta):
    gammas,thetas = theta_array(theta)
    gammas = gammas.reshape(-1,1)
    thetas = thetas.reshape(-1,1)
    J = C.J
    sum_log_factorial = C.sum_log_factorial
    return gammas*np.exp(-thetas+J*np.log(thetas)-sum_log_factorial)
    
def w_(C,theta):
    # gamma*exp(-t1)t1^j/j!/f(j|theta)
    f_tilda = f_tilda_array2d(C,theta)
    return f_tilda/np.sum(f_tilda,axis=0)   
    
def M_(C,theta):
    # M(theta) = argmax{theta}{Q(theta|theta_k)}
    
    w = w_(C,theta)
    wc = w*C.array
    wc_sum = np.sum(wc,axis=1)
    wcJ_sum = np.sum(wc*C.J,axis=1)
    c_sum = sum(C.c)
    
    gammas = wc_sum/c_sum
    thetas = wcJ_sum/wc_sum
    return theta_matrix(gammas,thetas)
       
def dQ_(C,theta):
    # gradient of Q(theta,theta)
    gammas,thetas = theta_array(theta)
    
    J = C.J
    w = w_(C,theta)
    wc = w*C.array
    wc_sum = np.sum(wc,axis=1)
    wcJ_sum = np.sum(wc*J,axis=1)
    c_sum = C.sum

    dgammas = wc_sum/gammas - c_sum #c_sum = lambda*
    dthetas = wcJ_sum/thetas - wc_sum
    return theta_matrix(dgammas,dthetas)

def ll_(C,theta):
    # log likelihood function
    ll = 0
    for j in range(C.count):
        ll += C.c[j]*math.log(f_(j,theta))
    return ll

def merit_rg(C,theta,g):
    # Convergence criteria
    ll_abs = np.abs(ll_(C,theta))
    lmax = max(ll_abs,1)
    g_abs = np.abs(g)
    theta_abs = np.abs(theta)
    theta_max = np.maximum(theta_abs,1)
    rg = np.max(np.multiply(g_abs,theta_max)/lmax)
    return rg    
    
def merit_dq(C,theta,g):
    # Convergence criteria
    return la.norm(g)   

def merit_EM(C,theta_prev,theta_next):
    # merit function
    return la.norm(theta_prev-theta_next)


@timethis    
def EM_(c,gammas0,thetas0,maxit=5,ftol=1e-6,printing=False,merit_type='em',returntype=0):
    C = inputdata(c)
    theta_prev = theta_matrix(gammas0,thetas0)*1.0
    theta_next = M_(C,theta_prev)
    
    
    if merit_type == 'em':
        merit_func = merit_EM
    elif merit_type == 'rg':
        merit_func = merit_rg
    else:
        merit_func = merit_dq
    
    merit = merit_func(C,theta_prev,theta_next)
    t = 0
    
    Log = []
    line = t,theta_prev.T,merit
    Log.append(line)
    printlines(line,printing)
    
    while t < maxit and merit >= ftol:
        t = t+1
        if t % 10000 == 0:
            print "EM: ",t,merit
            
        theta_prev = theta_next
        theta_next = M_(C,theta_prev)
        merit = merit_func(C,theta_prev,theta_next)
        
        line = t,theta_prev.T,merit
        Log.append(line)
        printlines(line,printing)
        
        if la.norm(theta_next-theta_prev) < 1e-18:
            print theta_next
            
    if returntype == 0:
        gammas,thetas = theta_array(theta_next)
        converged = merit < ftol
        return gammas,thetas,t,merit,Log,converged,ll_(C,theta_next)
    else:
        return theta_next


def gt_(C,theta):
    # M(theta) - theta
    return M_(C,theta) - theta

def g_(C,theta):
    # grad of log(L) = grad of Q
    return dQ_(C,theta)

def dS_(dg,dtheta,dtstar):
    # BFGS update for S
    C1 = float((1.0+(dg.T*dtstar)/(dg.T*dtheta))/(dg.T*dtheta))
    C2 = float(1.0/(dg.T*dtheta))
    return C1*dtheta*dtheta.T - C2*(dtstar*dtheta.T + (dtstar*dtheta.T).T)

def testdS_(dg,dgt,dtheta,S):
    # tests dS_
    C1_test = (1+float(dg.T*(-dgt+S*dg))/float(dg.T*dtheta))/float(dg.T*dtheta)
    C2_test = float(dg.T*dtheta)
    testdS = C1_test*dtheta*dtheta.T - (dtheta*(-dgt+S*dg).T+(-dgt+S*dg)*dtheta.T)/C2_test
    return testdS
    
    
def linesearch(C,theta,d,alpha=1.0):
    success = True
    maxJ = 10
    j = 0
    g = g_(C,theta)
    
    def armijo(C,theta,d,alpha):
        eta_c = 1e-4
        return ll_(C,theta + alpha*d) - ll_(C,theta) >= eta_c*alpha*float(g.T*d)
    def wolfe(C,theta,d,alpha):
        eta_w = 0.9999
        g1 = abs(float(g_(C,theta+alpha*d).T*d))
        g2 = abs(float(g_(C,theta).T*d))
        return g1 <= eta_w*g2
       
    while (armijo(C,theta,d,alpha) and wolfe(C,theta,d,alpha))==False and j < maxJ:
        alpha = 0.5*alpha
        j = j+1
    if j >= maxJ:
        success = False
    return success,alpha,j

def paramconstraint(theta,d,alpha=1.0):
    gammas,thetas = theta_array(theta + alpha*d)
    maxit = 100
    j = 0
    while (any(gammas > 1) or any(gammas < 0) or any(thetas < 0)) and j < maxit:
        alpha = alpha / 2.0
        gammas,thetas = theta_array(theta + alpha*d)
        j = j+1
    if j >= maxit:
        print "paramconstraint broken"
    return alpha
    
def linesearch_secant(C,theta,d,alpha=1.0):
    success = False
    a0 = 0
    a1 = alpha
    n = 0
    
    def F_(C,theta,d,alpha):
        g = g_(C,theta+alpha*d)
        return float(d.T*g)
    def sign(x):
        if x == 0:
            return 0
        elif x < 0:
            return -1
        else:
            return 1
        
    F00 = F_(C,theta,d,a0)
    F0 = F00
    F1 = F_(C,theta,d,a1)
    n = n+1
    while n < 10:
        if n != 1 and abs(F1) < 0.1*F00:
            success = True
            break
        else:
            if sign(a1-a0)*(F0-F1)/(abs(F0)+abs(F1)) < 1e-5:
                break
            else:
                a = (a1*F0 - a0*F1)/(F0-F1)
                a0 = a1
                F0 = F1
                a1 = a  
        F1 = F_(C,theta,d,a1)
        n = n+1
    return success, a1, n
    

@timethis
def QN2(c,gammas0,thetas0,maxit=100,ftol=1e-6,merit_type='rg',mod=False,printing=False):
    C = inputdata(c)
    numparams = len(gammas0+thetas0)
    theta = theta_matrix(gammas0,thetas0)*1.0
    theta = EM_(c,gammas0,thetas0,maxit=5,returntype=1)[0]
    S = np.matrix(np.zeros((numparams,numparams)))
    g = g_(C,theta)
    gt = gt_(C,theta)
    
    if merit_type == 'rg':
        merit_func = merit_rg
    else:
        merit_func = merit_dq
    
    merit = merit_func(C,theta,g)
    
    linesearchiterations = 0
    
    t = 0
    Log = []
    line = t,theta.T
    Log.append(line)
    printlines(line,printing)
    foundalpha = True
    alpha = 1
    countEMusage = 0
    while t < maxit and merit >= ftol:
        t=t+1
        if t % 1000 == 0:
            print "QN2: ",t,linesearchiterations,merit
        
        
        # step a)
        d = gt - S*g
        
        # step b)
        alpha = paramconstraint(theta,d)
        #if t % 1000 == 0 and False:
        #    if foundalpha == False:
        #        print 'alpha =',alpha,' paramconstraint'
        foundalphaprev = foundalpha
        foundalpha, alpha, iters = linesearch(C,theta,d,alpha)
        #if t % 1000 == 0 and False:
        #    if foundalpha == False:
        #        print 'alpha =',alpha,' linesearch'
        #        print la.norm(gt)
        #foundalpha, alpha, iters = linesearch_secant(C,theta,d,alpha)
        linesearchiterations += iters
        if foundalpha == False:
            if foundalphaprev == False:
                if mod == True:
                    numEMiters = 10
                    if countEMusage > 10:
                        numEMiters = 500
                    countEMusage += 1
                    gammas,thetas = theta_list(theta)
                    theta = EM_(c,gammas,thetas,maxit=numEMiters,returntype=1)[0]
                    linesearchiterations += 1
                else:
                    break
            S = np.matrix(np.zeros((numparams,numparams)))
            #g = g_(C,theta)
            #gt = gt_(C,theta)
            #rg = conv(C,theta,g)
        else:
            dtheta = alpha*d
            
            # step c)
            dg = g_(C,theta+dtheta) - g
            dgt = gt_(C,theta+dtheta) - gt
            
            # step d)
            dtstar = -dgt + S*dg
            dS = dS_(dg,dtheta,dtstar)
            
            # step e)
            thetaprev = theta.copy()
            theta = theta + dtheta
            g = g + dg
            gt = gt + dgt
            S = S + dS
            
            merit = merit_func(C,theta,g)
            if la.norm(thetaprev-theta) < 1e-18:
                print theta
        
        #gammas,thetas = theta_array(theta)
        #print np.round(S,2)
        line = t,theta.T,alpha,foundalpha,merit
        Log.append(line)
        printlines(line,printing)
    line = "total linesearch iterations:",linesearchiterations
    Log.append(line)
    printlines(line,printing)
    gammas,thetas = theta_array(theta)
    converged = merit < ftol
    return gammas,thetas,t,merit,Log,converged,ll_(C,theta)


def example1_params():
    c = [552,703,454,180,84,23,4]
    gammas0 = [0.3,0.7]
    thetas0 = [1.0,1.5]
    return c,gammas0,thetas0
    
def example1():
    QN2(*example1_params())