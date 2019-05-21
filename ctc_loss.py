import numpy as  np
import pdb
np.seterr(divide='raise',invalid='raise')
class CTC:
    '''
    The class implements forward backward algorithm along with CTC loss fucntion
    '''
    def __init__(self):
        pass

    def ctc_loss_mass(self,params, seq,blank=0,is_prob=True):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions over m frames.
        seq - sequence of phone id's for given example.
        is_prob - whether params have already passed through a softmax
        Returns objective and gradient.
        """
 	grad = np.zeros_like(params)
		
        try:
            seqLen = seq.shape[0] # Length of label sequence (# phones)
            numphones = params.shape[0] # Number of labels
            L = 2*seqLen + 1 # Length of label sequence with blanks
            T = params.shape[1] # Length of utterance (time)

            alphas = np.zeros((L,T))
            betas = np.zeros((L,T))
	    #grads = np.zeros_like(params)	
            # Keep for gradcheck move this, assume NN outputs probs
            
            if not is_prob:
                params = params - np.max(params,axis=0)
                params = np.exp(params)
                params = params / np.sum(params,axis=0)

        
            
            # Initialize alphas and forward pass 
            alphas[0,0] = params[blank,0]
            alphas[1,0] = params[seq[0],0]
            c = np.sum(alphas[:,0])
            alphas[:,0] = alphas[:,0] / c
            llForward = np.log(c)
            for t in xrange(1,T):
                start = max(0,L-2*(T-t)) 
                end = min(2*t+2,L)
                for s in xrange(start,L):
                    l = (s-1)/2
                    # blank
                    if s%2 == 0:
                        if s==0:
                            alphas[s,t] = alphas[s,t-1] * params[blank,t]
                        else:
                            alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
                    # same label twice
                    elif s == 1 or seq[l] == seq[l-1]:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                                      * params[seq[l],t]
                    
                # normalize at current time (prevent underflow)
                c = np.sum(alphas[start:end,t])
                alphas[start:end,t] = alphas[start:end,t] / c
                llForward += np.log(c)

            # Initialize betas and backwards pass
            betas[-1,-1] = params[blank,-1]
            betas[-2,-1] = params[seq[-1],-1]
            c = np.sum(betas[:,-1])
            betas[:,-1] = betas[:,-1] / c
            llBackward = np.log(c)
            for t in xrange(T-2,-1,-1):
                start = max(0,L-2*(T-t)) 
                end = min(2*t+2,L)
                for s in xrange(end-1,-1,-1):
                    l = (s-1)/2
                    # blank
                    if s%2 == 0:
                        if s == L-1:
                            betas[s,t] = betas[s,t+1] * params[blank,t]
                        else:
                            betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
                    # same label twice
                    elif s == L-2 or seq[l] == seq[l+1]:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
                    else:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                                     * params[seq[l],t]

                c = np.sum(betas[start:end,t])
                betas[start:end,t] = betas[start:end,t] / c
                llBackward += np.log(c)

            # Compute gradient with respect to unnormalized input parameters
            grad = np.zeros(params.shape)
            ab = alphas*betas
            for s in xrange(L):
                # blank
                if s%2 == 0:
                    grad[blank,:] += ab[s,:]
                    ab[s,:] = ab[s,:]/params[blank,:]
                else:
                    grad[seq[(s-1)/2],:] += ab[s,:]
                    ab[s,:] = ab[s,:]/(params[seq[(s-1)/2],:]) 
            absum = np.sum(ab,axis=0)

            grad = params - grad / (params * absum) 

            '''
            # Check for underflow or zeros in denominator of gradient
            llDiff = np.abs(llForward-llBackward)
            if llDiff > 1e-5 or np.sum(absum==0) > 0:
                print "Diff in forward/backward LL : %f"%llDiff
                print "Zeros found : (%d/%d)"%(np.sum(absum==0),absum.shape[0])
                return -llForward,grad,True
            '''
        except (FloatingPointError,ZeroDivisionError) as e:
                print '\nInside exception clause.....\n'
		print e.message
                llForward=0.0
                return -llForward,grad,True

        return -llForward,grad,False

    

