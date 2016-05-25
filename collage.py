import numpy as np
from matplotlib import pyplot as plt
from graph_cut import graph_cut

class ImageCollage(object):
    
    def __init__(self, mode, num_images):
        """
        Class for creating collages of images
        """
        
        self.mode = mode
        self.K = num_images
        
        tmp = plt.imread('data/' + mode + '/image1.jpg')
        self.n, self.m = tmp.shape[:2]
        
        self.inf = 1e+6
        self.inputs, self.masks = self.inputs_and_masks()
        self.phi_unaries, self.c_pairwise = self.create_potentials()
        
    def inputs_and_masks(self):
        """
        Read images and masks from the disk
        """
        
        inputs = np.zeros((self.K, self.n, self.m, 3))
        masks = np.zeros((self.K, self.n, self.m))
        
        for k in xrange(self.K):
            inputs[k] = plt.imread('data/' + self.mode + '/image' + str(k+1) + '.jpg')
            cur_mask = plt.imread('data/' + self.mode + '/mask' + str(k+1) + '.jpg')
            masks[k] = np.all(cur_mask < 250, axis=2)    
            
        return inputs, masks
    
    def create_potentials(self):
        """
        Create unary and pairwise potentials
        """
        
        (K, n, m, inputs, masks) = (self.K, self.n, self.m, self.inputs, self.masks)
        N_vert = n * m
        M_edg = 2 * (n-1) * (m-1) + (n-1) + (m-1)

        c_pairwise = np.zeros((M_edg, 3))
        phi_unaries = np.zeros((N_vert, K))
        i_vert = 0
        j_edge = 0
        for ih in xrange(n):
            for iw in xrange(m):
                if (masks[:, ih, iw].sum()>1e-1):
                    phi_unaries[i_vert] = self.inf * (1 - masks[:, ih, iw])
                
                if ih + 1 < n:
                    c_cur = 0
                    for p1 in xrange(K):
                        for p2 in xrange(K):
                            c_try = np.linalg.norm(inputs[p1, ih, iw] - inputs[p2, ih+1, iw])
                            c_cur = max(c_cur, c_try)
                    #c_pairwise[j_edge] = i_vert, i_vert + m, np.max(
                    #    np.linalg.norm(inputs[:, ih, iw] - inputs[:, ih+1, iw], axis=1))
                    c_pairwise[j_edge] = i_vert, i_vert + m, c_cur
                    j_edge += 1
                if iw + 1 < m:
                    c_cur = 0
                    for p1 in xrange(K):
                        for p2 in xrange(K):
                            c_try = np.linalg.norm(inputs[p1, ih, iw] - inputs[p2, ih, iw+1])
                            c_cur = max(c_cur, c_try)
                    c_pairwise[j_edge] = i_vert, i_vert + 1, c_cur
                    #c_pairwise[j_edge] = i_vert, i_vert + 1, np.max(
                    #    np.linalg.norm(inputs[:, ih, iw] - inputs[:, ih, iw+1], axis=1))
                    j_edge += 1
                i_vert += 1
        
        return phi_unaries, c_pairwise
    
    def d(self, x_i, x_j):
        return int(x_i != x_j)

    def alpha_expansion(self, max_iter=100, tol=1e-3, return_energies=False):
        
        (K, n, m, inputs, masks) = (self.K, self.n, self.m, self.inputs, self.masks)
        (phi_unaries, c_pairwise, d) = (self.phi_unaries, self.c_pairwise, self.d)
        
        # N_vert -- number of vertices
        # K -- number of classes
        N_vert, K = phi_unaries.shape
        # M_edg -- number of edges
        M_edg = c_pairwise.shape[0]

        X = np.zeros(N_vert, dtype='int')

        E_old = np.inf
        E_list = []
        #alpha = 0
        for i_iter in xrange(max_iter):
            # randomly select alpha
            #alpha = np.random.randint(K)
            #alpha = (alpha + 1) % K
            for alpha in xrange(K):
                ################ Construct the graph #######################
                terminal_weights = np.zeros((N_vert, 2))
                edge_weights = np.zeros((M_edg, 4))

                terminal_weights[:, 0] = phi_unaries[np.arange(N_vert), X] # y_i = 0 (X_i^{new} != \alpha)
                terminal_weights[X == alpha, 0] = 2 * self.inf # y_i cannot be zero where X_i^{old} = \alpha
                terminal_weights[:, 1] = phi_unaries[:, alpha] # y_i = 1 (X_i^{new} = alpha)

                for j in xrange(M_edg):
                    from_, to_, weight = c_pairwise[j]
                    from_, to_ = int(from_), int(to_)
                    phi_a = weight * d(X[from_], X[to_])
                    phi_b = weight * d(X[from_], alpha)
                    phi_c = weight * d(alpha, X[to_])
                    phi_d = weight * d(alpha, alpha)

                    phi_b_hat = (phi_b - phi_a + phi_c - phi_d)/2
                    edge_weights[j] = (from_, to_, phi_b_hat, phi_b_hat)
                    terminal_weights[from_][0] += phi_a
                    terminal_weights[from_][1] += phi_d + (phi_c - phi_d - phi_b + phi_a)/2
                    terminal_weights[to_][1] += (phi_b - phi_a - phi_c + phi_d)/2

                ################ Run graph cut #############################
                E, labels = graph_cut(terminal_weights, edge_weights)
                X[labels == 0] = alpha
                E_list.append(E)

            ################ Compute energy ############################
            if E_old - E < tol: break
            else: E_old = E

        if return_energies:
            return X, np.array(E_list)
        else:
            return X
        
    def alpha_beta_swap(self, max_iter=10, tol=1e-3, return_energies=False):
        '''
        Parameters:
            phi_unary -- matrix of unary potentials (N_vert x K)
            c_pairwise -- matrix of weights on the lattice (M_edg x 3),
                where each of 3 numbers are (from, to, c_{ij})
            d -- metric function        
        '''    
        (K, n, m, inputs, masks) = (self.K, self.n, self.m, self.inputs, self.masks)
        (phi_unaries, c_pairwise, d) = (self.phi_unaries, self.c_pairwise, self.d)
        
        # N_vert -- number of vertices
        # K -- number of classes
        N_vert, K = phi_unaries.shape
        # M_edg -- number of edges
        M_edg = c_pairwise.shape[0]

        X = np.random.randint(0, K, N_vert)

        E_list = []
        E_old = np.inf
        for i_iter in xrange(max_iter):
            # cycle  through alpha and beta (alpha < beta)
            for alpha in xrange(K-1):
                for beta in xrange(alpha+1, K):
                    ################ Construct the graph #######################
                    cur_ind_to_global = np.where((X == alpha) | (X == beta))[0]
                    cur_global_to_ind = {el: iel for iel, el in enumerate(cur_ind_to_global)}
                    N_vert_cur = len(cur_ind_to_global)
                    terminal_weights = np.zeros((N_vert_cur, 2))

                    froms_, tos_ = c_pairwise[:, 0].astype('int'), c_pairwise[:, 1].astype('int')
                    M_edg_cur = np.sum(((X[froms_] == alpha) & (X[tos_] == beta)) | 
                                       ((X[froms_] == alpha) & (X[tos_] == alpha)) | 
                                       ((X[froms_] == beta) & (X[tos_] == beta)) |
                                       ((X[froms_] == beta) & (X[tos_] == alpha)))

                    edge_weights = np.zeros((M_edg_cur, 4))

                    terminal_weights[:, 0] = phi_unaries[cur_ind_to_global, beta] # y_i = 0 (X_i^{new} != \alpha)
                    terminal_weights[:, 1] = phi_unaries[cur_ind_to_global, alpha] # y_i = 0 (X_i^{new} != \alpha)

                    j_cur = 0
                    for j in xrange(M_edg):
                        from_, to_, weight = c_pairwise[j]
                        from_, to_ = int(from_), int(to_)
                        if (X[from_] == alpha or X[from_] == beta) and (X[to_] == alpha or X[to_] == beta):
                            cur_from_, cur_to_ = cur_global_to_ind[from_], cur_global_to_ind[to_]
                            edge_weights[j_cur] = (cur_from_, cur_to_, 
                                                   weight * d(beta, alpha), weight * d(alpha, beta))
                            j_cur += 1
                        elif X[from_] == alpha or X[from_] == beta:
                            terminal_weights[cur_global_to_ind[from_], 0] += weight * d(beta, X[to_])
                            terminal_weights[cur_global_to_ind[from_], 1] += weight * d(alpha, X[to_])
                        elif X[to_] == alpha or X[to_] == beta:
                            terminal_weights[cur_global_to_ind[to_], 0] += weight * d(beta, X[from_])
                            terminal_weights[cur_global_to_ind[to_], 1] += weight * d(alpha, X[from_])

                    ################ Run graph cut #############################
                    E, labels = graph_cut(terminal_weights, edge_weights)
                    X[cur_ind_to_global[labels == 0]] = alpha
                    X[cur_ind_to_global[labels == 1]] = beta
                    for i in range(N_vert):
                        if X[i] != alpha & X[i] != beta:
                            E += phi_unaries[i, X[i]]

                    for j in range(M_edg):
                        from_, to_, weight = c_pairwise[j]
                        from_, to_ = int(from_), int(to_)
                        if X[from_] != alpha and X[from_] != beta and X[to_] != alpha and X[to_] != beta:
                            E += weight * d(X[from_], X[to_])
                            
                    E = 0
                    for i in range(N_vert):
                        E += phi_unaries[i, X[i]]

                    for j in range(M_edg):
                        from_, to_, weight = c_pairwise[j]
                        from_, to_ = int(from_), int(to_)
                        E += weight * d(X[from_], X[to_])
                            
                    E_list.append(E)
            ################ Compute energy ############################
            if E_old - E < tol: break
            else: E_old = E

        if return_energies:
            return X, np.array(E_list)
        else:
            return X
    
    def show_collage(self, X, atype='a', save=False):
        """
        Show resulting collage and save it to file
        """
        
        result = np.zeros((self.n, self.m, 3))
        X = X.reshape(self.n, self.m)
        for k in xrange(self.K):
            result += self.inputs[k] * (X == k)[:, :, None]
            
        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        ax[0].imshow(X.astype('uint8'))
        ax[1].imshow(result.astype('uint8'))
        
        if (save):
            fig.savefig('results/' + self.mode + '_collage_' + atype + '.jpg')
            
    def show_inputs_and_masks(self, save=False):
        """
        Show input images for collage and masks based on them
        """
        
        fig, ax = plt.subplots(2, self.K, figsize=(15, 8))

        images = [self.inputs, self.masks]
        
        for i in xrange(2):
            for k in xrange(self.K):
                ax[i,k].imshow(images[i][k].astype('uint8'))
                
        if (save):
            fig.savefig('results/' + self.mode + '_inputsmasks.jpg')
            
    def plot_energy(self, Es, lim=3, atype='a', save=False):
        """
        Plot energy
        """
        
        x = np.arange(Es.size)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x[lim:], Es[lim:], lw=2)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.set_ylabel('Energy', fontsize=14)
        ax.grid()

        if (save):
            fig.savefig('results/' + self.mode + '_energy_' + atype + '.jpg')