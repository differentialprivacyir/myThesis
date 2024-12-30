import numpy as np



class first_perturbation:
    def __init__(self, epsilon, attributes_domain, client_data):
        self.epsilon = epsilon
        self.attributes_domain = attributes_domain
        self.client_data = client_data
        self.randomized_data = self.anonymizing()

    def anonymizing(self):
        perturbation_attributes = []
        eps = self.epsilon
        for i in self.attributes_domain.keys():
            p = np.exp(eps) / (np.exp(eps) + len(self.attributes_domain[i]) - 1)
            q = (1 - p) / (len(self.attributes_domain[i]) - 1)
            indexx = self.attributes_domain[i].index(self.client_data.get(i))
            perturbation_list = [p if i == indexx else q for i in range(len(self.attributes_domain[i]))]
            perturbation_attributes.append(self.generalized_randomize_response(self.attributes_domain[i], perturbation_list))
        return perturbation_attributes

    def generalized_randomize_response(self, data_domain, perturbation_list):
        return np.random.choice(data_domain, p=perturbation_list)


