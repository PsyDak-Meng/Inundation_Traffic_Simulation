import random
import numpy as np

def originDestination_generation(p_res, p_biz, p_res_roads, p_biz_roads):
    # Args
    #   p_res: the probability a car is generated by residential effect
    #   p_biz: the probability a car is generated by business effect
    #   p_res_roads: given residential effect, the p of generating a car at each road
    #   p_biz_roads: given business effect, the p of generating a car at each road
    
    p_res_roads = {k: v * p_res for k, v in p_res_roads.items()}
    p_biz_roads = {k: v * p_biz for k, v in p_biz_roads.items()}
    p_generation = {key: p_res_roads[key] + p_biz_roads[key] for key in p_res_roads}
    
    roads = list(p_generation.keys())
    p = list(p_generation.values())

    # replace non-finite values with the maximum finite value
    # max_finite_value = np.nanmax([w for w in p if math.isfinite(w)])
    # finite_p = [w if math.isfinite(w) else max_finite_value for w in p]
    # p = finite_p
    
    return random.choices(roads, p, k = 1)[0]
    
def numbers_exponential(total_timeStep):
    numbers_exp = np.exp(-0.3 * np.arange(total_timeStep))
    numbers_exp /= np.sum(numbers_exp)
    return numbers_exp

def p_residential_business(numbers_exp):
    numbers_exp_flip = np.flip(numbers_exp)
    numbers_exp_sum = numbers_exp_flip + numbers_exp
    p_residential = numbers_exp / numbers_exp_sum
    p_business = numbers_exp_flip / numbers_exp_sum
    return p_residential.tolist(), p_business.tolist()