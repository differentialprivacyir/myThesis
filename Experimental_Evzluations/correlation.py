
from collections import Counter
import numpy as np
import pandas as pd


class correlation:
    def __init__(self, original_data, perturbed_data):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.MAE = self.correlation_calculation()


    def compute_joint_probabilities(self, x, y):
        """Compute joint probabilities for two categorical variables."""
        joint_counts = Counter(zip(x, y))
        total = len(x)
        joint_probs = {k: v / total for k, v in joint_counts.items()}
        return joint_probs

    def compute_entropy(self, probabilities):
        """Compute entropy from a probability distribution."""
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def joint_entropies_categorical(self, data):
        """Compute pairwise joint entropies for categorical data."""
        n_variables = data.shape[1]
        joint_entropies = np.zeros((n_variables, n_variables))

        for i in range(n_variables):
            for j in range(n_variables):
                if i != j:
                    # Calculate joint probabilities for variables i and j
                    joint_probs = self.compute_joint_probabilities(data[:, i], data[:, j])
                    joint_entropies[i, j] = self.compute_entropy(joint_probs.values())
                else:
                    # Self entropy for a variable
                    marginal_probs = np.unique(data[:, i], return_counts=True)[1] / len(data)
                    joint_entropies[i, j] = self.compute_entropy(marginal_probs)
        return joint_entropies

    def mutual_info_matrix_categorical(self,df, normalized=True):
        """Compute mutual information matrix for categorical data."""
        data = df.to_numpy()
        n_variables = data.shape[1]
        j_entropies = self.joint_entropies_categorical(data)
        entropies = j_entropies.diagonal()
        entropies_tile = np.tile(entropies, (n_variables, 1))
        sum_entropies = entropies_tile + entropies_tile.T
        mi_matrix = sum_entropies - j_entropies
        # if normalized:
        #     mi_matrix = mi_matrix * 2 / sum_entropies    
        return pd.DataFrame(mi_matrix, index=df.columns, columns=df.columns)


    def correlation_calculation(self):
        original_correlation = np.array(self.mutual_info_matrix_categorical(self.original_data))
        perturbed_correlation = np.array(self.mutual_info_matrix_categorical(self.perturbed_data))
        # Get the indices of the upper triangular part, excluding the diagonal
        triu_indices = np.triu_indices_from(original_correlation, k=1)  # k=1 excludes the diagonal
   
        # Extract the upper triangular elements excluding diagonal
        upper_tri1 = original_correlation[triu_indices]
        upper_tri2 = perturbed_correlation[triu_indices]
        
        # Compute the mean absolute difference
        mean_abs_diff = np.mean(np.abs(upper_tri1 - upper_tri2))
        return mean_abs_diff



