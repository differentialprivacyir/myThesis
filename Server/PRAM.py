
import itertools
import pandas as pd
import numpy as np



class PRAM:
    def __init__(self, epsilon, clusters, PBC, attributes_domain, dataset):
        self.epsilon = epsilon
        self.clusters = clusters
        self.PBC = PBC
        self.attributes_domain = attributes_domain
        self.dataset = dataset
        self.randomized_data , self.randomized_data2, self.randomized_data3,self.randomized_data4 = self.purturbation()



    def purturbation(self):
        print("========================= invariant PRAM ==========================")
        i = -1
        dataset_randomize = self.dataset.copy()
        dataset_randomize2 = self.dataset.copy()
        dataset_randomize3 = self.dataset.copy()
        dataset_randomize4 = self.dataset.copy()

        for cl in self.clusters:
            i = i + 1
            Q = dict()
            print(cl)
            # Q calculation for each variable in cluster i-th
            for atr in cl:
                e = self.epsilon * self.PBC[i]
                eps = e
                p = (np.exp(eps)/(np.exp(eps)+len(self.attributes_domain.get(atr))-1))
                q = (1-p)/(len(self.attributes_domain.get(atr))-1)
                Q.update({atr:self.Q_calculation(self.attributes_domain.get(atr), p, q)})

            # Q calculation for Compound Variables
            Qatr1 = Q.get(cl[0])
            if len(cl)>1 :
                Qatr2 = Q.get(cl[1])
                Compound_Q = np.kron(Qatr1,Qatr2)
                for atr in cl[2:]:
                    Compound_Q = np.kron(Compound_Q, Q.get(atr))
            else:
                Compound_Q = Q.get(cl[0])

            ## LAMBDA calculation
            LAMBDA = self.lambda_Compound_calculation(cl, dataset_randomize)
            EPI = self.estimatePI(Compound_Q,LAMBDA)

            EPI2 = self.estimatePI_Frequency_estimation2(cl,dataset_randomize2, Compound_Q)

            EPI3 = self.estimatePI3_combination(cl,dataset_randomize3, Compound_Q,LAMBDA)

            float_list = [float(value) for value in LAMBDA]
            EPI4 = self.em_estimate_original_prob3(np.array(float_list), Compound_Q)
            
           
            # Q' calculation
            SQ = dict()
            SQ.update({'cl'+str(i):self.Q_bar_calculation(cl, EPI, Compound_Q)})

            SQ2 = dict()
            SQ2.update({'cl'+str(i):self.Q_bar_calculation(cl, EPI2, Compound_Q)})
            
            SQ3 = dict()
            SQ3.update({'cl'+str(i):self.Q_bar_calculation(cl, EPI3, Compound_Q)}) 
            
            SQ4 = dict()
            SQ4.update({'cl'+str(i):self.Q_bar_calculation(cl, EPI4, Compound_Q)})


            # SQ5 = dict()
            # SQ5.update({'cl'+str(i):self.Q_bar_calculation(cl, EPI5, Compound_Q)})
            # second GRR
            # dataset_randomize = self.generalized_randomize_response(cl, dataset_randomize, SQ.get('cl'+str(i)))

            # dataset_randomize2 = self.generalized_randomize_response(cl, dataset_randomize2, SQ2.get('cl'+str(i)))
            
            # dataset_randomize3 = self.generalized_randomize_response(cl, dataset_randomize3, SQ3.get('cl'+str(i)))
            # dataset_randomize4 = self.generalized_randomize_response(cl, dataset_randomize4, SQ4.get('cl'+str(i)))

            dataset_randomize,dataset_randomize2,dataset_randomize3,dataset_randomize4 = self.generalized_randomize_response2(cl, dataset_randomize, dataset_randomize2, dataset_randomize3, dataset_randomize4,SQ.get('cl'+str(i)),SQ2.get('cl'+str(i)),SQ3.get('cl'+str(i)) , SQ4.get('cl'+str(i)))
            # dataset_randomize5 = self.generalized_randomize_response(cl, dataset_randomize5, SQ5.get('cl'+str(i)))

        return dataset_randomize, dataset_randomize2,dataset_randomize3,dataset_randomize4
    # ,dataset_randomize5


    def em_estimate_original_prob3(self, perturbed_probs, pram_matrix, max_iter=1000, tol=1e-6):
        num_categories = len(perturbed_probs)
        original_probs = np.full(num_categories, 1.0 / num_categories)  # Initial guess

        for iteration in range(max_iter):
            # E-step: Calculate expected probabilities for original data
            denominators = pram_matrix.T @ original_probs  # Shape: (num_categories,)
            denominators = np.where(denominators == 0, 1e-10, denominators)  # Avoid division by zero
            expected_counts = (pram_matrix * original_probs[:, np.newaxis]) / denominators
            expected_counts = expected_counts @ perturbed_probs

            # M-step: Update probabilities for original data
            new_original_probs = expected_counts / expected_counts.sum()

            # Check for convergence
            if np.linalg.norm(new_original_probs - original_probs, ord=1) < tol:
                print(f"Converged after {iteration + 1} iterations.")
                break

            original_probs = new_original_probs

        return original_probs

    # apply GRR based on compound variable(Q or Q')
    def generalized_randomize_response2(self, cluster, dataset1, dataset2, dataset3, dataset4, SQ1, SQ2, SQ3 , SQ4):

        combinations_number = 1
        for atr in cluster:
            combinations_number = combinations_number * len(self.attributes_domain.get(atr))

        d = []
        for atr in cluster:
            d.append(self.attributes_domain.get(atr))

        combinations = list(itertools.product(*d))

        combinations_list = [d for d in range(len(combinations))]
        for u in range(len(dataset1)):
            valuess1 = []
            valuess2 = []
            valuess3 = []
            valuess4 = []
            for atr in cluster:
                valuess1.append(dataset1[atr].values[u])
                valuess2.append(dataset2[atr].values[u])
                valuess3.append(dataset3[atr].values[u])
                valuess4.append(dataset4[atr].values[u])

            combination_index1 = combinations.index(tuple(valuess1))
            combination_index2 = combinations.index(tuple(valuess2))
            combination_index3 = combinations.index(tuple(valuess3))
            combination_index4 = combinations.index(tuple(valuess4))

            random_index1 = np.random.choice(combinations_list , p=SQ1[combination_index1])
            random_index2 = np.random.choice(combinations_list , p=SQ2[combination_index2])
            random_index3 = np.random.choice(combinations_list , p=SQ3[combination_index3])
            random_index4 = np.random.choice(combinations_list , p=SQ4[combination_index4])

            k = -1
            for atr in cluster:
                k = k + 1
                dataset1.loc[u, [atr]] =  combinations[random_index1][k]
                dataset2.loc[u, [atr]] =  combinations[random_index2][k]
                dataset3.loc[u, [atr]] =  combinations[random_index3][k]
                dataset4.loc[u, [atr]] =  combinations[random_index4][k]

        return dataset1,dataset2,dataset3,dataset4

    def estimatePI3_combination(self, cl,dataset_randomize3, Compound_Q,LAMBDA):
        Q_inverse = np.linalg.inv(Compound_Q)
        #Q_inverse=self.matrix_inverse_normalization(Q)
        L = (np.array(LAMBDA)).transpose()
        estimate_pi = np.dot(Q_inverse, L)
        negativee = False
        for val in estimate_pi:
            if val <= (-0.1):
                negativee = True

        if negativee:
            return self.estimatePI_Frequency_estimation2(cl,dataset_randomize3,Compound_Q)
        else:
            for val in estimate_pi:
                if val < 0:
                    regularized = estimate_pi - np.min(estimate_pi) + 1e-6  # Avoids division by zero or extreme scaling
                    # Step 2: Normalize to sum to 1
                    normalized = regularized / np.sum(regularized)
                    return normalized
            return estimate_pi     

        
        

    def estimatePI_Frequency_estimation(self,cl,FPResult, CQ):
        Pi = []
        d = []

        # calculate all possible combinations
        for at in cl:
            d.append(self.attributes_domain.get(at))
        combinations = list(itertools.product(*d))

        # calculate Lambda for each combination
        index = -1
        for combination in combinations:
            index = index + 1
            query_condition = pd.Series([True] * len(FPResult))
            for col, val in zip(cl, combination):
                query_condition &= (FPResult[col] == val)
            counts = query_condition.sum()
            combination_lambda = counts / len(FPResult)
            q = (1-CQ[index][index])/(len(CQ[index])-1)
            #q = sum(CQ[index][:index] + CQ[index][index + 1:])
            FE = (combination_lambda-(len(FPResult)*q))/(len(FPResult)*(CQ[index][index]-q))
            Pi.append(FE)
        min_val = min(Pi)
        max_val = max(Pi)
        normalized_data = [(x - min_val) / (max_val - min_val) for x in Pi]

        # Step 2: Scale to sum to 1
        total_sum = sum(normalized_data)
        final_normalized = [x / total_sum for x in normalized_data]

        return final_normalized




    def estimatePI_Frequency_estimation2(self,cl,FPResult, CQ):
        Pi = []
        d = []

        # calculate all possible combinations
        for at in cl:
            d.append(self.attributes_domain.get(at))
        combinations = list(itertools.product(*d))

        # calculate Lambda for each combination
        index = -1
        for combination in combinations:
            index = index + 1
            query_condition = pd.Series([True] * len(FPResult))
            for col, val in zip(cl, combination):
                query_condition &= (FPResult[col] == val)
            counts = query_condition.sum()
            combination_lambda = counts
            q = (1-CQ[index][index])/(len(CQ[index])-1)
            #q = sum(CQ[index][:index] + CQ[index][index + 1:])
            FE = (combination_lambda-(len(FPResult)*q))/(len(FPResult)*(CQ[index][index]-q))
            Pi.append(FE)
        # print("pppppppppppppppppppkkkkkkkkkkkkk")
        # print(Pi)
        # print("pppppppppppppppppppkkkkkkkkkkkkkk")
        regularized = Pi - np.min(Pi) + 1e-6  # Avoids division by zero or extreme scaling

        # Step 2: Normalize to sum to 1
        normalized = regularized / np.sum(regularized)

        return normalized



    # Q calculation
    def Q_calculation(self, domain, p, q):
        Q = np.full((len(domain), len(domain)), q)
        np.fill_diagonal(Q, p)

        return Q



    # Lambda calculation(λ)
    def lambda_Compound_calculation(self, cl ,FPResult):
        LAMBDA = []
        d = []

        # calculate all possible combinations
        for at in cl:
            d.append(self.attributes_domain.get(at))
        combinations = list(itertools.product(*d))

        # calculate Lambda for each combination
        for combination in combinations:
            query_condition = pd.Series([True] * len(FPResult))
            for col, val in zip(cl, combination):
                query_condition &= (FPResult[col] == val)
            counts = query_condition.sum()
            combination_lambda = counts / len(FPResult)
            LAMBDA.append(combination_lambda)

        return LAMBDA



    # Π estimation (estimate of the original attribute variable distribution (Π = Q^-1 . λ))
    def estimatePI(self, Q, Landa):
        Q_inverse = np.linalg.inv(Q)
        #Q_inverse=self.matrix_inverse_normalization(Q)
        L = (np.array(Landa)).transpose()
        estimate_pi = np.dot(Q_inverse, L)
        for val in estimate_pi:
            if val < 0:
                regularized = estimate_pi - np.min(estimate_pi) + 1e-6  # Avoids division by zero or extreme scaling
                # Step 2: Normalize to sum to 1
                normalized = regularized / np.sum(regularized)
                return normalized
        return estimate_pi


    
    def em_estimate_original_prob2(self, perturbed_probs, pram_matrix, max_iter=10000, tol=1e-8, reg_constant=1e-9):
        """
        Enhanced EM algorithm for estimating original distribution probabilities given perturbed probabilities.

        Parameters:
            perturbed_probs: np.array
                Array of observed perturbed probabilities.
            pram_matrix: np.array
                Transition (PRAM) matrix used for perturbation.
            max_iter: int
                Maximum number of iterations for the EM algorithm.
            tol: float
                Convergence tolerance for stopping criterion.
            reg_constant: float
                Regularization constant to prevent instability in updates.

        Returns:
            original_probs: np.array
                Refined estimated probabilities of the original distribution.
        """
        num_categories = len(perturbed_probs)

        # More informed prior: Start with slightly perturbed uniform distribution
        original_probs = np.full(num_categories, 1.0 / num_categories) + reg_constant

        for iteration in range(max_iter):
            # E-step: Calculate expected probabilities for original data
            expected_counts = np.zeros(num_categories)

            for j in range(num_categories):
                denominator = pram_matrix[:, j] @ original_probs + reg_constant  # Regularized denominator
                for i in range(num_categories):
                    expected_counts[i] += (
                        pram_matrix[i, j] * original_probs[i] /
                        denominator * perturbed_probs[j]
                    )

            # M-step: Update probabilities for original data, normalize with regularization
            new_original_probs = (expected_counts + reg_constant) / (expected_counts.sum() + num_categories * reg_constant)

            # Check for convergence with strict tolerance
            if np.linalg.norm(new_original_probs - original_probs, ord=1) < tol:
                break

            original_probs = new_original_probs

        return original_probs


    def em_estimate_original_prob(self,perturbed_probs, pram_matrix, max_iter=1000, tol=1e-6):
        """
        EM algorithm to estimate original distribution probabilities given perturbed probabilities.

        Parameters:
            perturbed_probs: np.array
                Array of observed perturbed probabilities.
            pram_matrix: np.array
                Transition (PRAM) matrix used for perturbation.
            max_iter: int
                Maximum number of iterations for the EM algorithm.
            tol: float
                Convergence tolerance for stopping criterion.

        Returns:
            original_probs: np.array
                Estimated probabilities of the original distribution.
        """
        # Initial guess for original probabilities (uniform distribution or other prior knowledge)
        num_categories = len(perturbed_probs)
        original_probs = np.full(num_categories, 1.0 / num_categories)  # Start with equal probabilities

        for iteration in range(max_iter):
            # E-step: Calculate expected probabilities for original data
            expected_counts = np.zeros(num_categories)

            for j in range(num_categories):
                for i in range(num_categories):
                    expected_counts[i] += (
                        pram_matrix[i, j] * original_probs[i] /
                        (pram_matrix[:, j] @ original_probs) * perturbed_probs[j]
                    )

            # M-step: Update probabilities for original data
            new_original_probs = expected_counts / expected_counts.sum()

            # Check for convergence
            if np.linalg.norm(new_original_probs - original_probs, ord=1) < tol:
                break

            original_probs = new_original_probs

        return original_probs

    # Q invrse normalization
    def matrix_inverse_normalization(self, Q):
        Q_inverse = np.linalg.inv(Q)
        #return Q_inverse
        min_value = Q_inverse.min()
        A_shifted = Q_inverse - min_value + 1e-6
        row_sums = A_shifted.sum(axis=1, keepdims=True)
        A_row_normalized = A_shifted / row_sums
        A_row_normalized_rounded = np.round(A_row_normalized, decimals=15)
        return A_row_normalized_rounded



    # Q' calculation
    def Q_bar_calculation(self, cluster, EPI, CQ):

        combinations = 1
        for atr in cluster:
            combinations = combinations * len(self.attributes_domain.get(atr))
        SQ = []
        EPI_CQ = [[EPI[j] * CQ[j, i] for j in range(combinations)] for i in range(combinations)]
        denominators = [sum(EPI[k] * CQ[k, i] for k in range(combinations)) for i in range(combinations)]
        # Now iterate and calculate the qbar values
        for i in range(combinations):
            attribute_Q = []
            for j in range(combinations):
                a = EPI[j]*CQ[j,i]
                s = denominators[i]
                qbar = a / s if s != 0 else 0
                attribute_Q.append(qbar)
            SQ.append(attribute_Q)
        min_value = min([item for sublist in SQ for item in sublist])

        # Convert SQ to a numpy array for array operations
        SQ_array = np.array(SQ)

        # Shift the array by subtracting the minimum value
        A_shifted = SQ_array - min_value

        # Calculate row sums for normalization, keeping dimensions for broadcasting
        row_sums = A_shifted.sum(axis=1, keepdims=True)

        # Normalize each row by its row sum
        A_row_normalized = A_shifted / row_sums

        # Round the result to 15 decimal places
        A_row_normalized_rounded = np.round(A_row_normalized, decimals=15)

        return A_row_normalized_rounded
        #return SQ



    # apply GRR based on compound variable(Q or Q')
    def generalized_randomize_response(self, cluster, dataset, SQ):

        combinations_number = 1
        for atr in cluster:
            combinations_number = combinations_number * len(self.attributes_domain.get(atr))

        d = []
        for atr in cluster:
            d.append(self.attributes_domain.get(atr))

        combinations = list(itertools.product(*d))

        for u in range(len(dataset)):
            valuess = []
            for atr in cluster:
                valuess.append(dataset[atr].values[u])

            combination_index = combinations.index(tuple(valuess))

            combinations_list = [d for d in range(len(combinations))]
            random_index = np.random.choice(combinations_list , p=SQ[combination_index])

            k = -1
            for atr in cluster:
                k = k + 1
                dataset.loc[u, [atr]] =  combinations[random_index][k]

        return dataset

