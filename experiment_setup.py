# -*- coding: utf-8 -*-
"""
Simulates experimental trials by generating visual and auditory stimulus
samples using a probabilistic generative model. 

Based on the generative model described in Kording et al. (2007),
section "Material and Methods", subsection "Generative model".

Kording, K.P., Beierholm, U., Ma, W.J., Quartz, S., Tenenbaum, J.B., Shams, 
L.: Causal Inference in Multisensory Perception. PLoS ONE 2(9), e943 (2007),
doi:10.1371/journal.pone.0000943

First released on Thu Nov 22 2023  

@author: João Filipe Ferreira
"""

# Global imports
import numpy as np              # Maths
from scipy.stats import binom   # Binomial distribution generator (from statistics toolbox)

def experiment_simulation(p_common, mu_p, sigma_p, sigma_v, sigma_a, trials):
    """
    Generates simulated experimental trial data for visual and auditory stimuli.
    
    Uses a generative model to sample stimulus parameters and sensor readings.
    
    Args:
        p_common: Probability of stimuli sharing a common cause 
        mu_p: Mean of prior distribution over stimulus source locations
        sigma_p: Std dev of prior over source locations 
        sigma_v: Std dev of visual sensory noise distribution 
        sigma_a: Std dev of auditory sensory noise distribution
        trials: Number of trials to simulate
        
    Returns: 
        real_C_exp: Generated values for number of sources 
        real_s_v_exp: Generated actual visual source locations
        real_s_a_exp: Generated actual auditory source locations  
        x_v_exp: Generated sensor readings on visual stimuli
        x_a_exp: Generated sensor readings on auditory stimuli
    """

    # ** STUDENT: MODIFY THE FOLLOWING CODE **
    # Generate values for C="actual number of sources"
    # Sample from Bernoulli distribution with probability p_common
    # binom.rvs(n=1, p, size) gives 1 with prob p and 0 otherwise
    c_samples = binom.rvs(1, p_common, size=trials)

    # Map 1 → common cause (C=1), 0 → independent causes (C=2)
    real_C_exp = np.where(c_samples == 1, 1, 2)

    # Generate values for actual positions s_v
    real_s_v_exp = np.random.normal(mu_p, sigma_p, trials)

    # Generate values for actual positions s_a depending on C
    real_s_a_exp = np.zeros(trials)

    # If C = 1 (common cause), s_a = s_v
    mask_common = (real_C_exp == 1)
    real_s_a_exp[mask_common] = real_s_v_exp[mask_common]

    # If C = 2 (independent causes), draw independently
    mask_independent = (real_C_exp == 2)
    real_s_a_exp[mask_independent] = np.random.normal(
        mu_p,
        sigma_p,
        np.sum(mask_independent)
    )

    # Generate values for sensor readings x_v and x_a
    x_v_exp = np.random.normal(real_s_v_exp, sigma_v, trials)
    x_a_exp = np.random.normal(real_s_a_exp, sigma_a, trials)
    ## ** STUDENT: END OF MODIFIED CODE **

    return real_C_exp, real_s_v_exp, real_s_a_exp, x_v_exp, x_a_exp