######## Use this code for test results when we know the parameters of the distribution ####


import numpy as np
import math 
import torch
import itertools as itertools
from itertools import product



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
        t = ((np.exp(-(1/2)*np.dot(np.inner((Z_sub - np.multiply(s,b)), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
          pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
        sum1 = sum1 + t;
        sum2 = sum2 + t;
      else:
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





pi = 0.3;
tau = np.sqrt(4);
b = 0.;
zip = [pi,cov,b,tau];
import random
from random import randint
alpha = 0.05;
def parallel_EM(zip):
  np.random.seed(randint(0, pow(2,31)));
  pi = zip[0]; cov = zip[1]; b=zip[2]; tau = zip[3];
  print("run");
  K = np.shape(cov)[1];
  H = np.random.binomial(1,pi,size=K);
  H_0 = np.zeros(K);H_1 = np.zeros(K);H_2 = np.zeros(K);
  Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
  #########################################################################################################
  pi_ini = pi; b_ini = b; tausq_ini = pow(tau,2);
  t_0 = 0.13891; t_1 = 0.14356; t_2 = 0.14667;
  level = 0.05;
  #########################################################################################################
  H_0[rejected_cutoff(locFDRS_GWAS_thread(Z,cov,0,b_ini,pi_ini,np.sqrt(tausq_ini)),t_0)] = 1;
  H_1[rejected_cutoff(locFDRS_GWAS_thread(Z,cov,1,b_ini,pi_ini,np.sqrt(tausq_ini)),t_1)] = 1;
  H_2[rejected_cutoff(locFDRS_GWAS_thread(Z,cov,2,b_ini,pi_ini,np.sqrt(tausq_ini)),t_2)] = 1;
  mat = np.array([H,H_0,H_1,H_2]);
  V_0 = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[1,]==1)));R_0 = np.size(np.where(mat[1,:]==1));
  V_1 = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[2,]==1)));R_1 = np.size(np.where(mat[2,:]==1));
  V_2 = np.size(np.intersect1d(np.where(mat[0,]==0),np.where(mat[3,]==1)));R_2 = np.size(np.where(mat[3,:]==1));
  return(np.array([[V_0,R_0],[V_1,R_1],[V_2,R_2]]))

import time
import concurrent.futures
start = time.perf_counter()
rep = 10;     ##### Number of replicates for determining estimates
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
    [[V_0[r],R_0[r]],[V_1[r],R_1[r]],[V_2[r],R_2[r]]] = re[r];

    

a = np.array([np.mean(V_0)/np.mean(R_0),bootsdmfdr(V_0,R_0,rep,rep),np.mean(V_1)/np.mean(R_1),bootsdmfdr(V_1,R_1,rep,rep),
 np.mean(V_2)/np.mean(R_2),bootsdmfdr(V_2,R_2,rep,rep),np.mean(np.divide(V_0,np.maximum(R_0,np.ones(rep)))), 
       np.std(np.divide(V_0,np.maximum(R_0,np.ones(rep))))/np.sqrt(rep),
np.mean(np.divide(V_1,np.maximum(R_1,np.ones(rep)))),np.std(np.divide(V_1,np.maximum(R_1,np.ones(rep))))/np.sqrt(rep),
np.mean(np.divide(V_2,np.maximum(R_2,np.ones(rep)))),np.std(np.divide(V_2,np.maximum(R_2,np.ones(rep))))/np.sqrt(rep),
np.mean(R_0 - V_0),np.std(R_0 - V_0)/np.sqrt(rep),np.mean(R_1 - V_1),np.std(R_1 - V_1)/np.sqrt(rep),
 np.mean(R_2 - V_2),np.std(R_2 - V_2)/np.sqrt(rep)])
print(a);
np.savetxt('result_1.csv', a.reshape(3,6), delimiter=',')   ##### mFDR level, FDR level and True positives by locFDR_N statistics