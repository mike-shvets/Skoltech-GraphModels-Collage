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
                phi_unaries[i_vert] = self.inf * (1 - masks[:, ih, iw])
                if ih + 1 < n:
                    c_pairwise[j_edge] = i_vert, i_vert + m, np.max(
                        np.linalg.norm(inputs[:, ih, iw] - inputs[:, ih+1, iw], axis=1))
                    j_edge += 1
                if iw + 1 < m:
                    c_pairwise[j_edge] = i_vert, i_vert + 1, np.max(
                        np.linalg.norm(inputs[:, ih, iw] - inputs[:, ih, iw+1], axis=1))
                    j_edge += 1
                i_vert += 1
        
        return phi_unaries, c_pairwise
    
    def d(self, x_i, x_j):
        return int(x_i != x_j)

    def alpha_expansion(self, max_iter=100, tol=1e-3):
        
        (K, n, m, inputs, masks) = (self.K, self.n, self.m, self.inputs, self.masks)
        (phi_unaries, c_pairwise, d) = (self.phi_unaries, self.c_pairwise, self.d)
        
        # N_vert -- number of vertices
        # K -- number of classes
        N_vert, K = phi_unaries.shape
        # M_edg -- number of edges
        M_edg = c_pairwise.shape[0]

        X = np.zeros(N_vert, dtype='int')

        E_old = np.inf
        alpha = 0
        for i_iter in xrange(max_iter):
            # randomly select alpha
            #alpha = np.random.randint(K)
            alpha = (alpha + 1) % K

            ################ Construct the graph #######################
            terminal_weights = np.zeros((N_vert, 2))
            edge_weights = np.zeros((M_edg, 4))

            terminal_weights[:, 0] = phi_unaries[np.arange(N_vert), X] # y_i = 0 (X_i^{new} != \alpha)
            terminal_weights[X == alpha, 0] = self.inf # y_i cannot be zero where X_i^{old} = \alpha
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

            ################ Compute energy ############################
            if E_old - E < tol: break
            else: E_old = E

        return X
    
    def show_collage(self, X, save=False):
        """
        Show resulting collage and save it to file
        """
        
        result = np.zeros((self.n, self.m, 3))
        X = X.reshape(self.n, self.m)
        for k in xrange(self.K):
            result += self.inputs[k] * (X == k)[:, :, None]
            
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(result.astype('uint8'))
        
        if (save):
            fig.savefig('results/' + self.mode + '.jpg')