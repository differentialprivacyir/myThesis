
import numpy as np
import itertools



class second_perturbation:
    def __init__(self, epsilon, attributes_domain, client_data, clusters, PBC):
        self.epsilon = epsilon
        self.attributes_domain = attributes_domain
        self.client_data = client_data
        self.clusters = clusters
        self.PBC = PBC
        self.randomized_data = self.anonymizing()

    def anonymizing(self):
        dataset_randomize = self.client_data.copy()
        for i, cl in enumerate(self.clusters):
            Q = dict()
            e = self.epsilon * self.PBC[i]

            # Calculate Q for each attribute in the cluster
            for atr in cl:
                eps = e  # Can adjust this per attribute if needed
                p = np.exp(eps) / (np.exp(eps) + len(self.attributes_domain.get(atr)) - 1)
                q = (1 - p) / (len(self.attributes_domain.get(atr)) - 1)
                Q[atr] = self.Q_calculation(self.attributes_domain.get(atr), p, q)

            # Compute compound Q matrix
            Compound_Q = Q[cl[0]]
            for atr in cl[1:]:
                Compound_Q = np.kron(Compound_Q, Q[atr])

            # Apply Generalized Randomized Response
            dataset_randomize = self.generalized_randomize_response(cl, dataset_randomize, Compound_Q)
        return dataset_randomize

    def Q_calculation(self, domain, p, q):
        Q = np.full((len(domain), len(domain)), q)
        np.fill_diagonal(Q, p)
        return Q

    def generalized_randomize_response(self, cluster, dataset, SQ):
        # Prepare all combinations of values for the cluster
        d = [self.attributes_domain[atr] for atr in cluster]
        combinations = list(itertools.product(*d))
        combination_map = {tuple(comb): idx for idx, comb in enumerate(combinations)}

        # Apply GRR to all rows
        for idx, row in dataset.iterrows():
            current_values = tuple(row[atr] for atr in cluster)
            if current_values in combination_map:
                combination_index = combination_map[current_values]
                random_index = np.random.choice(len(combinations), p=SQ[combination_index])
                randomized_values = combinations[random_index]

                # Update the dataset with randomized values
                for j, atr in enumerate(cluster):
                    dataset.at[idx, atr] = randomized_values[j]
        return dataset
