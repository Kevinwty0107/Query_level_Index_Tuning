"""
Reimplement SPGSequential{Actor,Critic} from 
    - https://arxiv.org/abs/1805.07010
    - https://github.com/pemami4911/sinkhorn-policy-gradient.pytorch
to allow for experimentation.

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment # Hungarian algorithm
"""
Aside: on assignment problem: 
- C[i,j], C \in \mathbb{R}^{N \times N}, is cost of assigning worker i a task j
- want a cheapest, bijective assignment
- this is isomorphic to rounding M to P (or rather, -M to P) 
"""
from pathos.multiprocessing import ProcessingPool


class SPGSequentialActor(nn.Module):
    """π(S)
    """
    
    def __init__(self, N, K, embed_dim, rnn_dim, sinkhorn_rds=5, sinkhorn_tau=1, n_workers=4, bidirectional=True):
        """
        Args:
            N (int): # discrete objects in COP
            K (int): # features per discrete object in COP 
            ...
            n_workers (int): n separate processes to spin up for sklearn's non-vectorized O(n^3) Hungarian alg
        """
        super(SPGSequentialActor, self).__init__()
        # constants
        self.N = N
        self.K = K
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.n_workers = n_workers

        # submodules        
        self.embedding = nn.Linear(in_features=self.K, out_features=self.embed_dim)
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.rnn_dim, bidirectional=bidirectional) 
        scale = 2 if bidirectional else 1
        self.h0 = torch.zeros(scale, self.rnn_dim, requires_grad=False)
        self.fc = nn.Linear(in_features=scale*rnn_dim, out_features=self.N)
        self.sinkhorn = Sinkhorn(self.N, sinkhorn_rds=sinkhorn_rds, sinkhorn_tau=sinkhorn_tau)
        self.round = linear_assignment
        
        if self.n_workers > 0:
            self.pool = ProcessingPool(self.n_workers)

    def forward(self, x, round=True):
        """
        Args:
            x (torch.Tensor): x is [b, N, K]
        """
        batch_size = x.size()[0]

        ## embed
        #   shapes: [b,N,K] x [K, embed_dim] = [b,N,embed_dim]
        x = F.leaky_relu(self.embedding(x)) 

        ## learn representation with rnn in both directions
        # n.b. batch dim is dim 1 for nn.GRU
        x = torch.transpose(x, 0, 1)
        h0 = self.h0.unsqueeze(1).repeat(1, batch_size, 1) # this is like np.tile, not np.repeat
        
        x, _ = self.gru(x, h0)
        #   shapes: [b, N, 2*rnn_dim]
        x = torch.transpose(x, 0, 1)

        ## map to permutation-matrix like tensor
        M_non_stochastic = self.fc(x)

        ## sinkhorn
        M = self.sinkhorn(M_non_stochastic)

        ## round
        if not round:
            return M, _
        
        M_arr = M.data.cpu().numpy() # keep M as torch.Tensor
        
        if self.n_workers > 0:
            # split batch into subbatches to be processed by separate processes
            M_data_splits = np.split(M_arr, self.n_workers, 0)
            P_splits = self.pool.map(parallel_assignment, M_data_splits)
            P = [P for P_split in P_splits for P in P_split]
        else:
            P = []
            for idx in range(batch_size):
                p = torch.zeros(self.N, self.N)
                assignment = self.round(-M_arr[idx]) # want maximal, not minimal, assignment 
                p[assignment[:,0], assignment[:,1]] = 1
                P.append(p)
        P = torch.stack(P)

        return M, P
          
class SPGSequentialCritic(nn.Module):
    """Q(S,A) 
        similar architecture to actor, but stops before Sinkhorn applications to S, fusing S with A to compute Q(S,A)
    """

    def __init__(self, N, K, embed_dim, rnn_dim, bidirectional=True):
        super(SPGSequentialCritic, self).__init__() 
        # constants
        self.N = N
        self.K = K
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim

        # submodules
        self.embedding_S = nn.Linear(K, embed_dim)
        self.embedding_P = nn.Linear(N, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.bn_S = nn.BatchNorm1d(num_features=N) # batchnorms stabilize separate activations from S, P 
        self.bn_P = nn.BatchNorm1d(num_features=N)
        self.bn_combine = nn.BatchNorm1d(num_features=N)

        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.rnn_dim, bidirectional=bidirectional) 
        scale = 2 if bidirectional else 1
        self.h0 = torch.zeros(scale, self.rnn_dim, requires_grad=False)

        self.fc0 = nn.Linear(scale*rnn_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 1) 
        self.fc2 = nn.Linear(N, 1)


    def forward(self, s, p):

        batch_size = s.size()[0]
        s = F.leaky_relu(self.bn_S(self.embedding_S(s)))
        p = F.leaky_relu(self.bn_P(self.embedding_P(p)))
        x = F.leaky_relu(self.bn_combine(s+p))

        x = torch.transpose(x, 0, 1)
        h0 = self.h0.unsqueeze(1).repeat(1, batch_size, 1)
        x, _ = self.gru(x, h0)
        x = torch.transpose(x, 0, 1) # [b, N, 2*rnn_dim]
        
        # squash to a scalar Q-value
        x = F.leaky_relu(self.fc0(x)) # [b, N, embed_dim]
        x = self.fc1(x) # [b, N, 1]
        return self.fc2(torch.transpose(x, 1, 2)) # [b, 1, 1]
        

class Sinkhorn(nn.Module):
    """
    Rather than compute 
        S(X) = S_0 = exp(X), S_i = col_norm(row_norm(S_{i-1}(X)))

    exponentiate (implicitly) and then take logs (implicitly), so that 1.
    exponentials are stable 2. normalizations are stable, and then exponentiate at the end 
    out of log space
    """

    def __init__(self, N, sinkhorn_rds=5, sinkhorn_tau=0.01):
        super(Sinkhorn, self).__init__()
        self.N = N
        self.rds = sinkhorn_rds
        self.tau = sinkhorn_tau
    
    def row_norm(self, x):
        return x - logsumexp(x, dim=2, keepdim=True)

    def col_norm(self, x):
        return x - logsumexp(x, dim=1, keepdim=True)

    def forward(self, x):
        x = x / self.tau
        for _ in range(self.rds):
            x = self.row_norm(x)
            x = self.col_norm(x)
        return torch.exp(x) + 1e-6


#
# util
#

def parallel_assignment(M_batch):
    P_batch = []
    sz, N, N = M_batch.shape
    for batch_idx in range(sz):
        p = torch.zeros(N, N)
        assignment = linear_assignment(-M_batch[batch_idx])
        p[assignment[:,0], assignment[:,1]] = 1
        P_batch.append(p)
    return P_batch

def logsumexp(x, dim, keepdim=False):
    """
    logsumexp for batched input
    recall logsumexp(x) = c + logsumexp(x-c)
    """
    c, _ = torch.max(x, dim=dim, keepdim=True)
    return c + (x - c).exp().sum(dim=dim, keepdim=True).log()