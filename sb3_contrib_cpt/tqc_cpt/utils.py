import numpy as np

rho1, rho2 = .58, .58
alpha, lmbd =  .77, 1.5
B = 1.0

def compute_cdf(k, q, pos):
    # compute cdf of k-th q quantile: 1 <= k <= q;
    
    if pos == True:
        return 1 - k/q
    else:
        return k/q

def prelec_weighting(taus, a=1.0):
    """
    :param quantiles: 1D tensor of sorted quantiles
    :param a: the parameter a of the prelec weighting function
    :return: the prelec-weighted quantiles
    """

    weights = np.exp(-(-np.log(taus))**a)
    return weights

def cpt_weighting(F, pos, rho1 = 0.58, rho2 = 0.58): 
    # @Prash: rho1 = .61, rho2 = .69 \in[0.3, 1]; assert w(F) monotonic in F;
    # @Tversky: rho1 = .61, rho2 = .69 lambda=2.25;
    # @Barberis: rho1 = rho2 = .5 

    if pos == True:
        return F**rho1 / ((F**rho1 + (1-F)**rho1)**(1/rho1))
    else:
        return F**rho2 / ((F**rho2 + (1-F)**rho2)**(1/rho2))

def utility(x, pos, alpha = 0.77, lmbd = 1.5, B = 1.0):
    # @Prash: alpha = .88 \in [0, 1], lmbd = 2.25 \in [1, 4];
    # @Barberis: alpha = .95, lmbd = 1.5 
    
    # compute u^{+/-}(x), where x is a quantile value: \in support(X);
    # X: accumulated reward r.v.;
    
    if pos == True:
        return (x-B)**alpha
    else:
        return lmbd * (B-x)**alpha 
        
def compute_CPT(XList, sort = False, B = B):
    
    if sort:
        XList = sorted(XList)
    
    m = len(XList)
    
    CPT_val_pos = 0
    CPT_val_neg = 0
    
    for id_, x in enumerate(XList):
        
        i = id_ + 1
        
        if x >= B:
            dF_i_pos = prob_weight(compute_cdf(i-1, m, True), True) - prob_weight(compute_cdf(i, m, True), True)
            CPT_val_pos += utility(x, True) * dF_i_pos # CPT_val_neg += 0
        
        elif x < B:
            dF_i_neg = prob_weight(compute_cdf(i, m, False), False) - prob_weight(compute_cdf(i-1, m, False), False)
            CPT_val_neg += utility(x, False) * dF_i_neg # CPT_val_pos += 0
        
        else:
            print('@compute_CPT, x invalid!!!')
            raise ValueError
        
    # ver: u^{-} non-decreasing, formula @compute_CPT() changed to C_n^{+} + C_n^{-}
    return CPT_val_pos + CPT_val_neg
