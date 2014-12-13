# Written by Justin Mao-Jones 12/13/14, jmj418@nyu.edu


import PoissonQN2_MultiMixture as qn2
from utils import AlgRuns
from utils import Problem
from utils import pickleIt
import gc

import numpy.random as random
import numpy as np

##############################################

NUM_MIXTURES = 2
NUM_RUNS = 100

##############################################

def problem_generator(num_mixtures=2,num_samples=3000,exp_lambda=2.0,inittype=1):
    lambdas = random.exponential(exp_lambda,num_mixtures)
    samples = random.poisson(lam=lambdas,size=(num_samples,num_mixtures))
    
    pvals = random.uniform(0,1,num_mixtures)
    pvals = pvals/np.sum(pvals)
    selector = random.multinomial(n=1,pvals=pvals,size=num_samples)
    
    selected = np.sum(selector*samples,axis=1)
    
    maxval = max(selected)
    J = np.arange(maxval+1)
    c = np.sum(selected.reshape(-1,1) == J,axis=0)
    
    return c,lambdas,pvals

def initialize(pvals,lambdas,inittype=1):
    num_mixtures = len(pvals)
    if inittype == 1:
        gammas0 = [1.0*(1.0+i)/sum(range(1,num_mixtures+1)) for i in range(num_mixtures)]
        thetas0 = range(1,num_mixtures+1)
    elif inittype ==2:
        gammas0 = pvals.tolist()
        thetas0 = lambdas.tolist()
    else:
        gammas0 = [1.0/num_mixtures]*num_mixtures
        thetas0 = range(1,num_mixtures+1)
    return gammas0,thetas0

def printlast20(Log):
    last20 = Log[-20:]
    for line in last20:
        for item in line:
            print item,
        print '\n',
    


num_mixtures = NUM_MIXTURES
num_samples = 3000
exp_lambda = 10.0
T = NUM_RUNS
numruns = 3
RunData = [[] for i in range(numruns)]

for t in range(T):
    print "Begin t =",t
    
    params_init = problem_generator(
                        num_mixtures=num_mixtures,
                        num_samples=num_samples,
                        exp_lambda=exp_lambda,
                        inittype=1)
    c,lambdas,pvals = params_init
    
    for r in range(numruns):    
        print "   RUN",r,
        
        gammas0,thetas0 = initialize(lambdas,pvals,inittype=r+1)
        problem = Problem(params_init,gammas0,thetas0) 
        
        params_EM = qn2.EM_(
                        c,gammas0,thetas0,
                        maxit=1e6,
                        ftol = 1e-6,
                        merit_type = 'em',
                        printing=False)
        print "EM",
        problem.init_EM(params_EM)
        params_EM = None
        gc.collect()
        params_QN2 = qn2.QN2(
                        c,gammas0,thetas0,
                        maxit=1e4,
                        ftol = 1e-6,
                        merit_type = 'rg',
                        mod = False,
                        printing=False)
        print "QN2"
        problem.init_QN2(params_QN2)
        RunData[r].append(problem)
        print "   EM",problem.EM.converged,problem.EM.k,problem.EM.time
        print "   QN",problem.QN2.converged,problem.QN2.k,problem.QN2.time
        params_QN2 = None
        gc.collect()
    





