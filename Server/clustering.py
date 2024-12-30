
from collections import Counter
import numpy as np
import math
import random


class clustering():
    def __init__(self, bayesian_network, attributes_domain,dataset):
        self.BN = bayesian_network
        self.attributes_domain = attributes_domain
        self.dataset = dataset
        
        len3 = False
        while not len3:
            self.clusters = self.Attribute_Clustering()
            len3 = all(len(sublist) < 4 for sublist in self.clusters)
        self.PBC = self.privacy_budget_coefficient()


    # clustering function
    def Attribute_Clustering(self):

        S = list(self.attributes_domain.keys())
        bayesianNetwork = self.BN[:]
        clusters = []  # All clusters are stored here

        while len(S) > 0:
            # choose random variable
            random_attribute = random.choice(S)
            # markove blanket calculation
            MB = self.markove_blanket(random_attribute,bayesianNetwork)

            if random_attribute not in MB:
                MB.insert(0, random_attribute)

            # add created cluster to clusters
            clusters.append(MB)

            # remove created cluster from S
            [S.remove(x) for x in MB if x in S]

            # remove created cluster from bayesianNetwork
            bayesianNetwork = [(t[0], [item for item in t[1] if item not in MB]) for t in bayesianNetwork if t[0] not in MB]
        return clusters



    # markove blanket function
    # MB(x) = Pa(x) ⋃ Ch(x) ⋃ {Pa(y)∣y ∈ Ch(x)}
    # x = atr in my code
    def markove_blanket(self, atr, bayesianNetwork):
        MB = []

        # Pa(x)
        parents = self.attribute_parents(atr, bayesianNetwork)
        [MB.append(x) for x in parents if x not in MB]

        # Ch(x)
        children = self.attribute_children(atr, bayesianNetwork)
        [MB.append(x) for x in children if x not in MB]

        for child in children:

            # {Pa(y)∣y ∈ Ch(x)}
            child_parents = self.attribute_parents(child, bayesianNetwork)
            [MB.append(x) for x in child_parents if x not in MB]

        return MB




    def attribute_parents(self, atr, bayesianNetwork):
        parents = []
        for AP in bayesianNetwork:
            if AP[0] == atr:
                parents.extend(list(AP[1]))
        return parents




    def attribute_children(self, atr, bayesianNetwork):
        children = []
        for AP in bayesianNetwork:
            if atr in AP[1]:
                children.extend([AP[0]])
        return children




    # PBC(CLi) = ( 1/CIF(CLi) )/( ∑ (j=1 to j=t )1/CIF(CLj) )
    # t = number of cluster
    def privacy_budget_coefficient(self):
        cl_cif = self.importance_factor()
        sum_of_reverse_cif = 0
        for cif in cl_cif:
            sum_of_reverse_cif += (1/cif)

        cl_pbc = []
        for cif in cl_cif:
            pbc = ((1/cif)/sum_of_reverse_cif)
            cl_pbc.append(pbc)

        pbc = self.new_PBC_calculation(cl_pbc)
        return pbc




    # CIF(CLi) = (∑ (Aj∈CLi) H(Aj) )/(∑ (k=1 to k=d) H(Ak))
    # d = number of attributs
    def importance_factor(self):
        #attributes_entropy, sum_of_entropy = self.joint_entropies()
        CIF = []
        entropy = []
        sum_of_entropy = 0
        for cl in self.clusters:
            cl_entropy = 0
            for atr in cl:
                cl_entropy += self.entropies(self.dataset[atr])
            entropy.append(cl_entropy)
            sum_of_entropy += cl_entropy

        for l in range(len(self.clusters)):
          cl_cif = (entropy[l]/sum_of_entropy)
          CIF.append(cl_cif)

        return CIF

    def entropies(self,atr):
      counts = Counter(atr)
      total = len(atr)

      # Calculate probabilities
      probabilities = [count / total for count in counts.values()]

      # Compute entropy
      entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
      return entropy



    # H(Ai) = − ∑ (x∈Ωi) Pr[Ai = x]logPr[Ai = x]
    def entropy(self):
        ent = dict()
        sum_of_entropy = 0
        for atr in self.attributes_domain.keys():
            pr = (1/(len(self.attributes_domain.get(atr))))
            H = (pr)*(math.log(pr))*(-1)
            sum_of_entropy += H
            ent.update({atr:H})
        return ent, sum_of_entropy



    def new_PBC_calculation(self,pbc):
        maxx = max(pbc)
        multi = (1/maxx)
        new_PBC = [i * multi for i in pbc]
        return new_PBC


