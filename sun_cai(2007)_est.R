lin.itp<-function(x, X, Y){
  ## x: the coordinates of points where the density needs to be interpolated
  ## X: the coordinates of the estimated densities
  ## Y: the values of the estimated densities
  ## the output is the interpolated densities
  x.N<-length(x)
  X.N<-length(X)
  y<-rep(0, x.N)
  for (k in 1:x.N){
    i<-max(which((x[k]-X)>=0))
    if (i<X.N)
      y[k]<-Y[i]+(Y[i+1]-Y[i])/(X[i+1]-X[i])*(x[k]-X[i])
    else 
      y[k]<-Y[i]
  }
  return(y)
}



EstNull.func<-function (x,gamma=0.1)
{
  # x is a vector of z-values
  # gamma is a parameter, default is 0.1
  # output the estimated mean and standard deviation
  
  n = length(x)
  t = c(1:1000)/200
  
  gan    = n^(-gamma)
  that   = 0 
  shat   = 0
  uhat   = 0
  epshat = 0
  
  phiplus   = rep(1,1000)
  phiminus  = rep(1,1000)
  dphiplus  = rep(1,1000)
  dphiminus = rep(1,1000)
  phi       = rep(1,1000)
  dphi      = rep(1,1000)
  
  for (i in 1:1000) {
    s = t[i]
    phiplus[i]   = mean(cos(s*x))
    phiminus[i]  = mean(sin(s*x))
    dphiplus[i]  = -mean(x*sin(s*x))
    dphiminus[i] = mean(x*cos(s*x))
    phi[i]       = sqrt(phiplus[i]^2 + phiminus[i]^2)
  }
  
  ind = min(c(1:1000)[(phi - gan) <= 0])
  tt = t[ind]
  a  = phiplus[ind]
  b  = phiminus[ind]
  da = dphiplus[ind]
  db = dphiminus[ind]
  c  = phi[ind]
  
  that   = tt
  shat   = -(a*da + b*db)/(tt*c*c)
  shat   = sqrt(shat) 
  uhat   = -(da*b - db*a)/(c*c)
  epshat = 1 - c*exp((tt*shat)^2/2)
  
  return(musigma=list(mu=uhat,s=shat))
}


epsest.func <- function(x,u,sigma)
{
  # x is a vector
  # u is the mean
  # sigma is the standard deviation
  
  z  = (x - u)/sigma
  xi = c(0:100)/100
  tmax=sqrt(log(length(x)))
  tt=seq(0,tmax,0.1)
  
  epsest=NULL
  
  for (j in 1:length(tt)) { 
    
    t=tt[j]
    f  = t*xi
    f  = exp(f^2/2)
    w  = (1 - abs(xi))
    co  = 0*xi
    
    for (i in 1:101) {
      co[i] = mean(cos(t*xi[i]*z));
    } 
    epshat = 1 - sum(w*f*co)/sum(w)
    epsest=c(epsest,epshat)
  }
  return(epsest=max(epsest))
}


adpt.cutz<-function(lfdr, q)
{
  # the input
  # lfdr the vector of local fdr statistics
  # q the desired FDR level
  # the output is a list with
  # the first element (st.lfdr) the sorted local fdr values
  # the second element (k) the number of hypotheses to be rejected
  # the third element (lfdrk) the threshold for the local fdr values
  # the fourth element (reject) the set of indices of the rejected hypotheses
  # the fifth element (accept) the set of indices of the accepted hypotheses
  
  m=length(lfdr)
  st.lfdr<-sort(lfdr)
  k=1
  while(k<m && (1/k)*sum(st.lfdr[1:k])<q){
    k=k+1
  }
  k<-k-1
  lfdrk<-st.lfdr[k]
  reject<-which(lfdr<=lfdrk)
  accept<-which(lfdr>lfdrk)
  y<-list(sf=st.lfdr, nr=k, thr=lfdrk, re=reject, ac=accept)
  return (y)
}



adaptZ.func<-function(zv, q)
{
  # the input
  # zv is the z-values transformed from m tests
  # q is the desired FDR level
  # the output is a list with
  # the first element (st.lfdr) the sorted local fdr values
  # the second element (k) the number of hypotheses to be rejected
  # the third element (lfdrk) the threshold for the local fdr values
  # the fourth element (reject) the set of indices of the rejected hypotheses
  # the fifth element (accept) the set of indices of the accepted hypotheses
  ## the estimates for the local fdr statistics
  # density estimates 
  zv.ds<-density(zv, from=min(zv)-10, to=max(zv)+10, n=1000)
  # linear interpolation
  zv.ds<-lin.itp(zv, zv.ds$x, zv.ds$y)
  # estimating the null distribution
  zv.MuSigma<-EstNull.func(zv)
  mu<-zv.MuSigma$mu
  s<-zv.MuSigma$s
  #mu<-0; s<-1
  zv.p0<-1-epsest.func(zv, mu, s)
  zv.lfdr<-zv.p0*dnorm(zv, mu, s)/zv.ds
  y<-adpt.cutz(zv.lfdr, q)
  return (y)
}








adaptZ.funcnull<-function(zv, q)
{
  # the input
  # zv is the z-values transformed from m tests
  # q is the desired FDR level
  # the output is a list with
  # the first element (st.lfdr) the sorted local fdr values
  # the second element (k) the number of hypotheses to be rejected
  # the third element (lfdrk) the threshold for the local fdr values
  # the fourth element (reject) the set of indices of the rejected hypotheses
  # the fifth element (accept) the set of indices of the accepted hypotheses
  ## the estimates for the local fdr statistics
  # density estimates 
  zv.ds<-density(zv, from=min(zv)-10, to=max(zv)+10, n=1000)
  # linear interpolation
  zv.ds<-lin.itp(zv, zv.ds$x, zv.ds$y)
  # estimating the null distribution
  #zv.MuSigma<-EstNull.func(zv)
  #mu<-zv.MuSigma$mu
  #s<-zv.MuSigma$s
  mu<-0; s<-1
  zv.p0<-1-epsest.func(zv, mu, s)
  zv.lfdr<-zv.p0*dnorm(zv, mu, s)/zv.ds
  y<-adpt.cutz(zv.lfdr, q)
  return (y$re)
}
