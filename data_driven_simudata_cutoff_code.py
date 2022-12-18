######### Test Results for all data driven methods #########


import numpy as np
import math 
import torch
import itertools as itertools
from itertools import product
import statsmodels.stats.multitest
from scipy.stats import norm
# Using R inside python
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import FloatVector
# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('sun_cai(2007)_est.R')

# Loading the functions for sun and cai method defined in R.
est_sun_cai_r = robjects.globalenv['epsest.func']   #### function for estimating $\pi$

rej_sun_cai_r = robjects.globalenv['adaptZ.funcnull']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


######### Function for calculating $locFDR_N$ statistics given the parameters of the model 
#   b, $\pi$, $\tau$ and the z-scores $Z$, covariance of the z-scores cov. 
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



##### Function for calculating the bootstrap standard error of the mFDR given
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


######## Function for determining the rejections using $locFDR_N$ statistics
def rejected_cutoff(lcfdr,cutoff):
    return(np.where(lcfdr<cutoff));

def mFDR_level_t(data_set,t_0,N,draws):
    avg_V = 0;
    avg_R = 0;
    for r in range(draws):
        lcfdr = data_set[r][N]
        idx = np.where(lcfdr <= t_0);
        avg_V = avg_V + np.sum(lcfdr[idx]);
        avg_R = avg_R + np.size(idx);
    return(avg_V/avg_R);




########### Choose one of the covariances below ###############
'''### AR(1) covariance
rho = 0.5;
#B = 1;
K = 4000;
diagonal = np.concatenate((1/(1-pow(rho,2)),np.repeat(((1+pow(rho,2))/(1-pow(rho,2))),K-2),1/(1-pow(rho,2))),axis=None);
off_diag = -rho/(1-pow(rho,2));
prec = np.diag(diagonal);
for d in range(K-1):
  prec[d,d+1]=prec[d+1,d] = off_diag;
cov = np.linalg.inv(prec);'''


'''### FGN Covariance
import numpy as np
H = 0.7;
K = 10000;
cov = np.repeat(0.1,(K*K)).reshape(K,K)
for i in range(K):
  for j in range(K):
    cov[i,j] = (0.5)*(pow(abs(abs(i-j)+1),(2*H)) - (2*pow(abs(i-j),(2*H))) + pow(abs(abs(i-j)-1),(2*H)));'''


### Banded Covariance with bandwidth 1
rho = 0.5
K = 2000
cov = np.diag(np.ones(K));
for i in range(K-1):
  cov[i,i+1] = cov[i+1,i] = rho;


'''import pandas as pd
import numpy as np
cov = pd.read_csv("corr3.5kjump.csv")
cov = np.array(cov)
#print(np.shape(cov))
cov = cov[:,1:3581]'''




############## Set the parameters of the model for simulation
pi = 0.3;
tau = np.sqrt(4);
b = 0.;


zip = [pi,cov,b,tau];
import random
from random import randint
alpha = 0.05; #significance level

############### Function returning the number of false rejections by $locFDR_N$ statistics
#
def parallel_EM(zip):
  np.random.seed(randint(0, pow(2,31)));
  pi = zip[0]; cov = zip[1]; b=zip[2]; tau = zip[3];
  print("run");
  K = np.shape(cov)[1];
  H = np.random.binomial(1,pi,size=K);
  H_0 = np.zeros(K);H_1 = np.zeros(K);H_2 = np.zeros(K);
  HS = np.zeros(K); HB = np.zeros(K); HAB = np.zeros(K);
  Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
  #########################################################################################################
  ### EM Algorithm implementation
  ### Change est_mat, z, pi_ini, b_ini, tausq_ini depending on the estimation method "Est-S&C" and       "Est-EM".
  maxiter = 5000;
  ##est_mat = np.zeros(3*maxiter).reshape(maxiter ,3)
  est_mat = np.zeros(2*maxiter).reshape(maxiter ,2)
  #z = Z[((4*(1+np.arange(1000)))-1)];
  #z = Z[((3*(1+np.arange(1150)))-1)];
  z = Z[((2*(1+np.arange(1000)))-1)]
  ##pi_ini = max(est_sun_cai_r(FloatVector(z),0.,1.)+0.);
  pi_est = max(est_sun_cai_r(FloatVector(Z),0.,1.)+0.);
  ##b_ini = np.mean(z)/pi_ini; tausq_ini = np.amax([0.,(np.var(z)-1-(pow(b_ini,2)*(1-pow(pi_ini,2))))/pi_ini]);
  b_ini = np.mean(z)/pi_est; tausq_ini = np.amax([0.,(np.var(z)-1-(pow(b_ini,2)*(1-pow(pi_est,2))))/pi_est]); 
  ##est_mat[0,0] = pi_ini; est_mat[0,0] = b_ini; est_mat[0,1] = tausq_ini;
  est_mat[0,0] = b_ini; est_mat[0,1] = tausq_ini;
  for iter in 1+np.arange(maxiter-1):
    p_tau = 1-locFDRS_GWAS_thread(z,cov,0,b_ini,pi_est,np.sqrt(tausq_ini));
    ##p_tau = 1-locFDRS_GWAS_thread(z,cov,0,b_ini,pi_ini,np.sqrt(tausq_ini));
    ##pi_next = np.mean(p_tau);
    b_next = (np.sum((p_tau)*z)/np.sum((p_tau)));
    tausq_next = np.amax([0,(np.sum((p_tau)*pow(z-b_next,2))
    /np.sum((p_tau)))-1]);
    b_ini = b_next; tausq_ini = tausq_next;
    ##pi_ini = pi_next; b_ini = b_next; tausq_ini = tausq_next;
    ##est_mat[iter,0] = pi_ini; est_mat[iter,1] = b_ini; est_mat[iter,2] = tausq_ini;
    est_mat[iter,0] = b_ini; est_mat[iter,1] = tausq_ini;
    if np.amax(abs(est_mat[iter,:]-est_mat[iter-1,:])) < pow(10,-15):
      break;
   
  pi_ini = pi_est;
    ### Estimated Parameters pi_ini, b_ini, tausq_ini
  data_set =[]
  draws = 5
  for _ in range(draws):
      H_d = np.random.binomial(1,pi_ini,size=K);
      Z_d = np.multiply(H_d,b_ini) +                               np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H_d),tausq_ini)),np.random.standard_normal(K));
      lcfdr0 = locFDRS_GWAS_thread(Z_d,cov,0,b_ini,pi_ini,np.sqrt(tausq_ini));
      lcfdr1 = locFDRS_GWAS_thread(Z_d,cov,1,b_ini,pi_ini,np.sqrt(tausq_ini));
      lcfdr2 = locFDRS_GWAS_thread(Z_d,cov,2,b_ini,pi_ini,np.sqrt(tausq_ini));
      data_set.append([lcfdr0,lcfdr1,lcfdr2]);
        


  ### Determine the cutoff $t_{\alpha, N}$ for given $alpha$ and $N$
  level = alpha;
  Z_d = Z;
  p_val = locFDRS_GWAS_thread(Z_d,cov,0,b_ini,pi_ini,np.sqrt(tausq_ini));
  oo = np.argsort(p_val);
  ss = np.sort(p_val);
  stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
  collection = np.max(np.where(stat <= level));
  t_0 = ss[collection];
  #####################################
  p_val = locFDRS_GWAS_thread(Z_d,cov,1,b_ini,pi_ini,np.sqrt(tausq_ini));
  oo = np.argsort(p_val); 
  ss = np.sort(p_val);
  stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
  collection = np.max(np.where(stat <= level));
  t_1 = ss[collection];
  ######################################
  p_val = locFDRS_GWAS_thread(Z_d,cov,2,b_ini,pi_ini,np.sqrt(tausq_ini));
  oo = np.argsort(p_val);
  ss = np.sort(p_val);
  stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
  collection = np.max(np.where(stat <= level));
  t_2 = ss[collection];
    

  

  jump = 0.0005;
  mFDR_level = mFDR_level_t(data_set,t_0,0,draws);
  for _ in range(200):
      if mFDR_level<level:
          t_0 = t_0+jump;
          mFDR_level = mFDR_level_t(data_set,t_0,0,draws);
          if mFDR_level>level:
              t_0 = t_0-jump;
              break;
      else:
          t_0 = t_0-jump;
          mFDR_level = mFDR_level_t(data_set,t_0,0,draws);
          if mFDR_level<level:
              break;
        





  mFDR_level = mFDR_level_t(data_set,t_1,1,draws);

  for _ in range(200):
      if mFDR_level<level:
          t_1 = t_1+jump;
          mFDR_level = mFDR_level_t(data_set,t_1,1,draws);
          if mFDR_level>level:
              t_1 = t_1-jump;
              break;
      else:
          t_1 = t_1-jump;
          mFDR_level = mFDR_level_t(data_set,t_1,1,draws);
          if mFDR_level<level:
              break;
        
    


  mFDR_level = mFDR_level_t(data_set,t_2,2,draws);

  for _ in range(200):
      if mFDR_level<level:
          t_2 = t_2+jump;
          mFDR_level = mFDR_level_t(data_set,t_2,2,draws);
          if mFDR_level>level:
              t_2 = t_2-jump;
              break;
      else:
          t_2 = t_2-jump;
          mFDR_level = mFDR_level_t(data_set,t_2,2,draws);
          if mFDR_level<level:
              break;
        
    
  del data_set
  #########################################################################################################
  H_0[rejected_cutoff(locFDRS_GWAS_thread(Z,cov,0,b_ini,pi_ini,np.sqrt(tausq_ini)),t_0)] = 1;
  H_1[rejected_cutoff(locFDRS_GWAS_thread(Z,cov,1,b_ini,pi_ini,np.sqrt(tausq_ini)),t_1)] = 1;
  H_2[rejected_cutoff(locFDRS_GWAS_thread(Z,cov,2,b_ini,pi_ini,np.sqrt(tausq_ini)),t_2)] = 1;
  HS[np.array(rej_sun_cai_r(FloatVector(Z),level))-1] = 1;
  HB[np.where(statsmodels.stats.multitest.multipletests(2*(1-norm.cdf(abs(Z))), alpha=level, method='fdr_bh', is_sorted=False, returnsorted=False)[0]==True)[0]] =1;
  HAB[np.where(statsmodels.stats.multitest.multipletests(2*(1-norm.cdf(abs(Z))), alpha=(level/(1-pi_ini)), method='fdr_bh', is_sorted=False, returnsorted=False)[0]==True)[0]] =1;
  mat = np.array([H,H_0,H_1,H_2,HS,HB,HAB]);
  V_0 = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[1,]==1)));R_0 = np.size(np.where(mat[1,:]==1));
  V_1 = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[2,]==1)));R_1 = np.size(np.where(mat[2,:]==1));
  V_2 = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[3,]==1)));R_2 = np.size(np.where(mat[3,:]==1));
  V_S = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[4,]==1)));R_S = np.size(np.where(mat[4,:]==1));
  V_B = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[5,]==1)));R_B = np.size(np.where(mat[5,:]==1));
  V_AB = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[6,]==1)));R_AB = np.size(np.where(mat[6,:]==1));
  return([np.array([[V_0,R_0],[V_1,R_1],[V_2,R_2],[V_S,R_S],[V_B,R_B],[V_AB,R_AB]]),[t_0,t_1,t_2]])

import time
import concurrent.futures
start = time.perf_counter()
rep = 10;
zip = [pi,cov,b,tau];
list = []
for _ in range(rep):
  list.append(zip)
from multiprocessing import Pool

if __name__ =="__main__":
  p=Pool()
  results = p.map(parallel_EM,list)
  print(results)


finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')


re = []
for result in results:
  re.append(result)
  
  
V_0 = np.zeros(rep);R_0 = np.zeros(rep);V_1 = np.zeros(rep);
R_1 = np.zeros(rep);V_2 = np.zeros(rep);R_2 = np.zeros(rep);
V_S = np.zeros(rep);R_S = np.zeros(rep);
V_B = np.zeros(rep);R_B = np.zeros(rep); V_AB = np.zeros(rep);R_AB = np.zeros(rep);

cutoffs = [];
test_results = [];
for r in range(rep):
    [[V_0[r],R_0[r]],[V_1[r],R_1[r]],[V_2[r],R_2[r]],[V_S[r],R_S[r]],[V_B[r],R_B[r]],[V_AB[r],R_AB[r]]] = re[r][0];
    cutoffs.append(re[r][1]);
    test_results.append(np.ravel(re[r][0]));

    

a=np.array([np.mean(V_0)/np.mean(R_0),bootsdmfdr(V_0,R_0,rep,rep),np.mean(V_1)/np.mean(R_1),bootsdmfdr(V_1,R_1,rep,rep),np.mean(V_2)/np.mean(R_2),bootsdmfdr(V_2,R_2,rep,rep),np.mean(V_S)/np.mean(R_S),bootsdmfdr(V_S,R_S,rep,rep),np.mean(V_B)/np.mean(R_B),bootsdmfdr(V_B,R_B,rep,rep),np.mean(V_AB)/np.mean(R_AB),bootsdmfdr(V_AB,R_AB,rep,rep),np.mean(np.divide(V_0,np.maximum(R_0,np.ones(rep)))), 
 np.std(np.divide(V_0,np.maximum(R_0,np.ones(rep))))/np.sqrt(rep),
np.mean(np.divide(V_1,np.maximum(R_1,np.ones(rep)))),np.std(np.divide(V_1,np.maximum(R_1,np.ones(rep))))/np.sqrt(rep),
np.mean(np.divide(V_2,np.maximum(R_2,np.ones(rep)))),np.std(np.divide(V_2,np.maximum(R_2,np.ones(rep))))/np.sqrt(rep),
 
 np.mean(np.divide(V_S,np.maximum(R_S,np.ones(rep)))),np.std(np.divide(V_S,np.maximum(R_S,np.ones(rep))))/np.sqrt(rep),
 np.mean(np.divide(V_B,np.maximum(R_B,np.ones(rep)))),np.std(np.divide(V_B,np.maximum(R_B,np.ones(rep))))/np.sqrt(rep),
 np.mean(np.divide(V_AB,np.maximum(R_AB,np.ones(rep)))),np.std(np.divide(V_AB,np.maximum(R_AB,np.ones(rep))))/np.sqrt(rep),
np.mean(R_0 - V_0),np.std(R_0 - V_0)/np.sqrt(rep),np.mean(R_1 - V_1),np.std(R_1 - V_1)/np.sqrt(rep),
 np.mean(R_2 - V_2),np.std(R_2 - V_2)/np.sqrt(rep),
 np.mean(R_S - V_S),np.std(R_S - V_S)/np.sqrt(rep),np.mean(R_B - V_B),
 np.std(R_AB - V_AB)/np.sqrt(rep),np.mean(R_AB - V_AB),np.std(R_AB - V_AB)/np.sqrt(rep)])

np.savetxt('result_trial.csv', a.reshape(3,12), delimiter=',') #### mFDR level, FDR level and True positives by all the datadriven methods
print(a.reshape(3,12));

np.savetxt('cutoffs_trial.csv',cutoffs,delimiter = ',');   #### Cutoffs in all replications

np.savetxt('test_results_trial.csv',test_results,delimiter = ','); #### rejections and false rejections by all the methods
