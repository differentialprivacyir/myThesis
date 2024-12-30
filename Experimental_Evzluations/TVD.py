import itertools
import pandas as pd


import numpy as np
import itertools
import pandas as pd


from itertools import product
import numpy as np
import itertools
   



class TVD:
    def __init__(self, original_data, perturbed_data, domains):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.domains = domains
        self.tvd = self.total_variation_distance()

    def total_variation_distance(self):
        # Precompute frequencies for original and perturbed data
        freq_original = self.compute_frequencies(self.original_data)
        freq_perturbed = self.compute_frequencies(self.perturbed_data)

        # Calculate TVD using precomputed frequencies
        tvd = 0
        for key in freq_original.keys():
            o_prob = freq_original.get(key, 0)
            p_prob = freq_perturbed.get(key, 0)
            tvd += abs(o_prob - p_prob)

        return tvd * 0.5

    def compute_frequencies(self, data):
        # Create a multi-index from the Cartesian product of domains
        all_combinations = list(product(*[self.domains[col] for col in data.columns]))
        freq_dict = {comb: 0 for comb in all_combinations}

        # Count occurrences of each combination
        grouped = data.groupby(list(data.columns)).size()
        total_count = len(data)

        for key, count in grouped.items():
            freq_dict[key] = count / total_count

        return freq_dict


