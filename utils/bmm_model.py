import numpy as np
import torch
import scipy.stats as stats

################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min,np_array=False):
        if not np_array:
            x_i = x.clone().cpu().numpy()
            x_i = np.array((self.lookup_resolution * x_i).astype(int))
        else:
            x_i = x
            x_i = (self.lookup_resolution * x_i).astype(np.uint8)
        
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    # def plot(self):
    #     x = np.linspace(0, 1, 100)
    #     plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
    #     plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
    #     plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
    
    
def bmm_probs(neighbour_CE,all_index,device):
    ### BMM ###
    # outliers detection
    n_CE_cp = np.copy(neighbour_CE)
    max_perc = np.percentile(n_CE_cp, 95)
    min_perc = np.percentile(n_CE_cp, 5)
    n_CE_cp = n_CE_cp[(n_CE_cp<=max_perc) & (n_CE_cp>=min_perc)]

    # "loss" -> we are using CE measure here
    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6

    n_CE_cp = (n_CE_cp - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    n_CE_cp[n_CE_cp>=1] = 1-10e-4
    n_CE_cp[n_CE_cp <= 0] = 10e-4

    print('######## Estimating BMM ########')
    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(n_CE_cp)
    bmm_model.create_lookup(1)
    print('######## BMM created ########')
    
    neighbour_CE = (neighbour_CE - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)
    neighbour_CE[neighbour_CE >= 1] = 1 - 10e-4
    neighbour_CE[neighbour_CE <= 0] = 10e-4
    
    for i in range(round(neighbour_CE.shape[0]/100)):
        
        B = bmm_model.look_lookup(neighbour_CE[i*100:(i*100)+100], bmm_model_maxLoss, bmm_model_minLoss,np_array=True)
        if i == 0:
            B_t = B
        else:
            B_t = np.concatenate((B_t, B))
        
    B_sorted = np.zeros(len(B_t))
    B_sorted[all_index.cpu().numpy()] = B_t
    
    return B_sorted