import numpy as np
from scipy.special import erfinv
from sklearn.neighbors import NearestNeighbors


def adod_outlier_detection(X: np.ndarray, perplexity: int = None, p_cum: float = 0.999, 
                           use_nns: bool = True, n_neighbors_multiplier: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ADOD (Adaptive Density Outlier Detection)
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix
    perplexity : int, optional
        Perplexity parameter，used to calculate adaptive local scale. If None, set to 2*floor(sqrt(n))
    p_cum : float, optional
        Cumulative probability parameter, falls within the calculated boundary. 0.999 as default, which refers to the three-sigma rule
    use_nns : bool, optional
        Use nns or not. True as default
    n_neighbors_multiplier : int, optional
        Nearest neighbors multipliers。3 as default, find 3*sqrt(n) nearest neighbors
    
    Returns:
    --------
    scores : np.ndarray, shape (n_samples,)
        Anomalous scores of each samples. The higher the score, the higher probability to be anomalous
    ranks : np.ndarray, shape (n_samples,)
        Descending sort of sample index by anomalous socres
    r : np.ndarray, shape (n_samples,)
        Adaptive neighborhood boundary of each sample
    local_density : np.ndarray, shape (n_samples,)
        Local density estimate of each sample
    
    Example:
    --------
    scores, ranks = adod_outlier_detection(X, perplexity=None)
    """
    n_samples = X.shape[0]
    
    # Step 1: Initialize parameters, especially perplexity calculation
    if perplexity is None:
        perplexity = 2 * int(np.floor(np.sqrt(n_samples)))
    
    print(f"ADOD parameter: n_samples={n_samples}, perplexity={perplexity}, p_cum={p_cum}")
    
    # Step 2: Use NNS to calculate distance matrix
    if use_nns:
        # Use NNS
        k_neighbors = min(n_neighbors_multiplier * int(np.floor(np.sqrt(n_samples))), n_samples - 1)
        print(f"Use NNS to find {k_neighbors} neighbors")
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 to containi itself
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        # Create sparse distance matrix (only save k neareast)
        D_sparse = distances
        I_sparse = indices
    else:
        # Calculate complete distance matrix
        print("Calculate complete distance matrix")
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                D[i, j] = np.linalg.norm(X[i] - X[j])
        
    
    # Step 3: Calculate adaptive local scale σ_i
    print("Calculate adaptive local scale...")
    sigma = np.zeros(n_samples)
    for i in range(n_samples):
        if use_nns:
            D_i = D_sparse[i, 1:]  # exclude itself (index 0)
        else:
            D_i = np.delete(D[i], i)  # exclude itself
        sigma[i] = binary_search_sigma(D_i, perplexity)
    
    # Step 4: Get adaptive neighborhood boundary r_i
    print("Get adaptive neighborhood boundary...")
    r = np.sqrt(2) * sigma * erf_inv(2 * p_cum - 1)
    
    # Step 5: Construct mutual neighbor graph G
    print("Construct mutual neighbor graph...")
    edges = []
    neighbor_dict = {}  # Store mutal neighbor list of each point
    for i in range(n_samples):
        neighbor_dict[i] = []
    if use_nns:
        # Use sparse distance matrix to construct G   
        for i in range(n_samples):
            for idx in range(1, len(I_sparse[i])):  # Skip itself
                j = I_sparse[i, idx]
                if i < j:  # Avoid repetation
                    D_ij = D_sparse[i, idx]
                    if D_ij <= min(r[i], r[j]):
                        edges.append((i, j))
                        neighbor_dict[i].append(j)
                        neighbor_dict[j].append(i)
    else:
        # Use complete distance matrix to construct G
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D_ij = D[i, j]
                if D_ij <= min(r[i], r[j]):
                    edges.append((i, j))
                    neighbor_dict[i].append(j)
                    neighbor_dict[j].append(i)
    
    # Step 6: Estimate local density d_i
    print("Estimate local density...")
    local_density = np.zeros(n_samples)
    for i in range(n_samples):
        deg_i = len(neighbor_dict[i])
        local_density[i] = (deg_i + 1) / r[i]  # +1 to contain itself
    
    # Step 7: Anomalous scores
    print("Calculate anomalous scores...")
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        # Initial scores as reciprocal of local density
        scores[i] = 1.0 / local_density[i]
        
        # Calculate the density diffence of neighbors
        if len(neighbor_dict[i]) > 0:
            # Calculate wights
            weights = []
            density_diffs = []
            
            if use_nns:
                # Use sparse distance matrix
                for j in neighbor_dict[i]:
                    idx_in_sparse = np.where(I_sparse[i] == j)[0]
                    if len(idx_in_sparse) > 0:
                        D_ij = D_sparse[i, idx_in_sparse[0]]
                    else:
                        # if no j in neighbors of i, find i in neighbors of j
                        idx_in_sparse = np.where(I_sparse[j] == i)[0]
                        if len(idx_in_sparse) > 0:
                            D_ij = D_sparse[j, idx_in_sparse[0]]
                        else:
                            D_ij = np.linalg.norm(X[i] - X[j])
                    
                    w_ij = 1.0 / D_ij if D_ij > 0 else 1.0
                    weights.append(w_ij)
                    density_diffs.append(1.0 / local_density[j] - 1.0 / local_density[i])
            else:
                for j in neighbor_dict[i]:
                    D_ij = D[i, j]
                    w_ij = 1.0 / D_ij if D_ij > 0 else 1.0
                    weights.append(w_ij)
                    density_diffs.append(1.0 / local_density[j] - 1.0 / local_density[i])
            
            # Normalize
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            density_diffs = np.array(density_diffs)
            
            # Add weights to density difference
            scores[i] += np.sum(weights * density_diffs)
    
    # Descending sort by scores (Anomalous first)
    ranks = np.argsort(-scores)
    
    print("ADOD Finished!")
    return scores, ranks, r, local_density


def binary_search_sigma(D_i: np.ndarray, target_perplexity: float, 
                        tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    Use binary search to find perplexity to reach target σ
    
    Parameters:
    -----------
    D_i : np.ndarray
        Distance from i to other points
    target_perplexity : float
        Target perplexity
    tol : float
        Convergence tolerance
    max_iter : int
        Max iterations
    
    Returns:
    --------
    sigma : float
        make perplexity to reach σ
    """
    # Initialize search range
    sigma_min = 1e-10
    sigma_max = 1e10
    sigma = 1.0
    
    for _ in range(max_iter):
        # Calculate probability 
        P_i = np.exp(-D_i**2 / (2 * sigma**2))
        sum_P_i = np.sum(P_i)
        
        if sum_P_i == 0:
            # When sum_P_i is 0, it means that the current sigma is too small, causing the similarity of all points to point i to be 0.
            # In this case, we cannot obtain a meaningful probability distribution.
            # Therefore, as a second best option, we assume that the similarity of point i to all other points is the same, that is, a uniform distribution.
            P_i = np.ones_like(D_i) / len(D_i)
        else:
            P_i = P_i / sum_P_i
        
        # Calculate entropy
        # Avoid log(0)
        P_i = np.maximum(P_i, 1e-12)
        H_i = -np.sum(P_i * np.log2(P_i))
        
        # Calculate perplexity
        perplexity = 2 ** H_i
        
        # Check convergence
        if abs(perplexity - target_perplexity) < tol:
            break
        
        # Adjust search range of σ
        if perplexity > target_perplexity:
            sigma_max = sigma
            sigma = (sigma_min + sigma) / 2
        else:
            sigma_min = sigma
            sigma = (sigma + sigma_max) / 2
    
    
    return sigma


def erf_inv(x: float) -> float:
    """
    inverse of the error function
    Use scipy.special.erfinv
    """
    return erfinv(x)


# ADOD for unknown data（Algorithm 2）
def adod_for_unknown_data(X_unk: np.ndarray, X_kn: np.ndarray, 
                          r_kn: np.ndarray, d_kn: np.ndarray, 
                          perplexity: int = None, p_cum: float = 0.999) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply ADOD for unknown data(based on known data)
    
    Parameters:
    -----------
    X_unk : np.ndarray, shape (m_samples, n_features)
        Unknown data
    X_kn : np.ndarray, shape (n_samples, n_features)
        Known data
    r_kn : np.ndarray, shape (n_samples,)
        Neighborhood boundary of known points
    d_kn : np.ndarray, shape (n_samples,)
        Local density of known points
    perplexity : int, optional
        Perplexity parameter
    p_cum : float, optional
        Cumulative probability
    
    Returns:
    --------
    scores_unk : np.ndarray, shape (m_samples,)
        Scores of unknown data points
    ranks_unk : np.ndarray, shape (m_samples,)
        Index sorted by unknown datascores
    r_unk_i : np.ndarray, shape (m_samples,)
        Adaptive neighborhood boundary of unknown data points
    d_unk_i : np.ndarray, shape (m_samples,)
        Local density estimate of unkown data points
    """
    m_samples = X_unk.shape[0]
    n_samples = X_kn.shape[0]
    
    if perplexity is None:
        perplexity = 2 * int(np.floor(np.sqrt(n_samples)))
    
    scores_unk = np.zeros(m_samples)
    
    for i in range(m_samples):
        # Calculate distance of unknown points to known points
        D_unk_i = np.array([np.linalg.norm(X_unk[i] - X_kn[j]) for j in range(n_samples)])
        
        # Calculate adaptive local scale of unknown points
        sigma_unk_i = binary_search_sigma(D_unk_i, perplexity)
        
        # Get neighborhood boundary of unknwown points
        r_unk_i = np.sqrt(2) * sigma_unk_i * erf_inv(2 * p_cum - 1)
        
        # Find neighbors
        neighbor_list = []
        for j in range(n_samples):
            if D_unk_i[j] <= min(r_unk_i, r_kn[j]):
                neighbor_list.append(j)
        
        # Estimate local density
        d_unk_i = (len(neighbor_list) + 1) / r_unk_i
        
        # Calculate scores
        scores_unk[i] = 1.0 / d_unk_i
        
        if len(neighbor_list) > 0:
            weights = []
            density_diffs = []
            
            for j in neighbor_list:
                D_ij = D_unk_i[j]
                w_ij = 1.0 / D_ij if D_ij > 0 else 1.0
                weights.append(w_ij)
                density_diffs.append(1.0 / d_kn[j] - 1.0 / d_unk_i)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            density_diffs = np.array(density_diffs)
            
            scores_unk[i] += np.sum(weights * density_diffs)
    
    ranks_unk = np.argsort(-scores_unk)
    
    return scores_unk, ranks_unk, r_unk_i, d_unk_i
