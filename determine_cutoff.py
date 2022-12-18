################# Use this code to determine the cutoffs for known data generation


import numpy as np
import math 
import itertools as itertools
from itertools import product
import statsmodels.stats.multitest
from scipy.stats import norm

def locFDRS_GWAS_thread(Z,cov,B,b,pi,tau):
  K = np.size(Z);
  locFDR = np.zeros(K);
  for i in range(K):
    Z_sub = Z[max(i-B,0):(min(i+B,K-1)+1)];
    cov_sub = cov[max(i-B,0):(min(i+B,K-1)+1),max(i-B,0):(min(i+B,K-1)+1)];
    l = min(i+B,K-1) - max(i-B,0) + 1;
    H = product([0,1], repeat=l);                                           
    sum1 = sum2 = 0;
    for s in H:
      if(s[min(B,i)] == 0): 
        #t = (multivariate_normal.pdf(Z_sub,mean=np.multiply(s,b),cov = cov_sub+np.multiply(np.diag(s),pow(tau,2)))*pow(pi,sum(s)) * pow(1-pi,l-sum(s)))
        t = ((np.exp(-(1/2)*np.dot(np.inner((Z_sub - np.multiply(s,b)), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
          pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
        sum1 = sum1 + t;
        sum2 = sum2 + t;
      else:
        #sum2 = sum2 + (multivariate_normal.pdf(Z_sub,mean=np.multiply(s,b),cov = cov_sub+np.multiply(np.diag(s),pow(tau,2)))*pow(pi,sum(s)) * pow(1-pi,l-sum(s)))
        sum2 = sum2 + ((np.exp(-(1/2)*np.inner(np.dot(Z_sub - np.multiply(s,b), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
          pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
    locFDR[i]= sum1/sum2;
  
  
  return(locFDR);



def bootsdmfdr(V,R,rep,boot):
  stat = np.zeros(boot);
  for b in range(boot):
    boot_sample = np.random.choice(range(rep),size=rep,replace=True);
    V_boot = V[boot_sample];
    R_boot = R[boot_sample];
    stat[b]=np.mean(V_boot)/np.mean(R_boot);
  return(np.std(stat));


def rejected(p_val,level):
  oo = np.argsort(p_val);
  ss = np.sort(p_val);
  stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
  collection = np.where(stat <= level);
  return(oo[collection]);

def rejected_cutoff(lcfdr,cutoff):
    return(np.where(lcfdr<cutoff));

################# Choose the covariance as choice ########


### AR(1) covariance
rho = 0.5;
#B = 1;
K = 1000;
diagonal = np.concatenate((1/(1-pow(rho,2)),np.repeat(((1+pow(rho,2))/(1-pow(rho,2))),K-2),1/(1-pow(rho,2))),axis=None);
off_diag = -rho/(1-pow(rho,2));
prec = np.diag(diagonal);
for d in range(K-1):
  prec[d,d+1]=prec[d+1,d] = off_diag;
cov = np.linalg.inv(prec);

'''### Equicorrelated Covariance
rho = 0.5
K = 1000
cov = np.repeat(rho,K*K).reshape(K,K);
for i in range(K):
    cov[i,i] = 1;'''

'''### FGN Covariance
import numpy as np
H = 0.7;
K = 1000;
cov = np.repeat(0.1,(K*K)).reshape(K,K)
for i in range(K):
  for j in range(K):
    cov[i,j] = (0.5)*(pow(abs(abs(i-j)+1),(2*H)) - (2*pow(abs(i-j),(2*H))) + pow(abs(abs(i-j)-1),(2*H)));'''

########## Set the parameters of choice ##########
pi = 0.3;
tau = np.sqrt(4);
b = 0.;


zip = [pi,cov,b,tau];
import random
from random import randint
alpha = 0.05;
#N = 1;
level = alpha;
K = np.shape(cov)[1];
H = np.random.binomial(1,pi,size=K);
Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
#####################################
p_val = locFDRS_GWAS_thread(Z,cov,0,b,pi,tau);
oo = np.argsort(p_val);
ss = np.sort(p_val);
stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
collection = np.max(np.where(stat <= level));
#t_0 = np.max(p_val[oo[collection]]);
t_0 = ss[collection];
print(t_0)
#####################################
p_val = locFDRS_GWAS_thread(Z,cov,1,b,pi,tau);
oo = np.argsort(p_val);
ss = np.sort(p_val);
stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
collection = np.max(np.where(stat <= level));
#t_0 = np.max(p_val[oo[collection]]);
t_1 = ss[collection];
print(t_1)
######################################
p_val = locFDRS_GWAS_thread(Z,cov,2,b,pi,tau);
oo = np.argsort(p_val);
ss = np.sort(p_val);
stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
collection = np.max(np.where(stat <= level));
#t_0 = np.max(p_val[oo[collection]]);
t_2 = ss[collection];
print(t_2)


def cut_off(zip,ndraws,N,t_0):
    pi = zip[0]; cov = zip[1]; b=zip[2]; tau = zip[3];
    avg_V = 0;
    avg_R = 0;
    for rep in range(ndraws):
        Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
        lcfdr = locFDRS_GWAS_thread(Z,cov,N,b,pi,tau);
        idx = np.where(lcfdr <= t_0);
        avg_V = avg_V + np.sum(lcfdr[idx]);
        avg_R = avg_R + np.size(idx);
    return(avg_V/avg_R);


ndraws = 10;     ###### Number of draws B for bootstrapping

jump = 0.005;
########################################
print("N = 0")
mFDR_level = cut_off(zip,ndraws,0,t_0);
print(mFDR_level)
for _ in range(40):
    if mFDR_level<level:
        t_0 = t_0+jump;
        mFDR_level = cut_off(zip,ndraws,0,t_0);
        print(mFDR_level)
        if mFDR_level>level:
            t_0 = t_0-jump;
            break;
    else:
        t_0 = t_0-jump;
        mFDR_level = cut_off(zip,ndraws,0,t_0);
        print(mFDR_level)
        if mFDR_level<level:
            break;
        
    
#t_0 = t_0 - 0.001;
print(t_0)        ###### Cutoff for $T_0$ statistics

##############################################
print("N = 1")

mFDR_level = cut_off(zip,ndraws,1,t_1);
print(mFDR_level)
for _ in range(20):
    if mFDR_level<level:
        t_1 = t_1+jump;
        mFDR_level = cut_off(zip,ndraws,1,t_1);
        print(mFDR_level)
        if mFDR_level>level:
            t_1 = t_1-jump;
            break;
    else:
        t_1 = t_1-jump;
        mFDR_level = cut_off(zip,ndraws,1,t_1);
        print(mFDR_level)
        if mFDR_level<level:
            break;
        
    
#t_0 = t_0 - 0.001;
print(t_1)    ###### Cutoff for $T_1$ statistics

###############################################
print("N = 2")

mFDR_level = cut_off(zip,ndraws,2,t_2);
print(mFDR_level)
for _ in range(20):
    if mFDR_level<level:
        t_2 = t_2+jump;
        mFDR_level = cut_off(zip,ndraws,2,t_2);
        print(mFDR_level)
        if mFDR_level>level:
            t_2 = t_2-jump;
            break;
    else:
        t_2 = t_2-jump;
        mFDR_level = cut_off(zip,ndraws,2,t_2);
        print(mFDR_level)
        if mFDR_level<level:
            break;
        
    
#t_0 = t_0 - 0.001;
print(t_2)   ###### Cutoff for $T_2$ statistics
