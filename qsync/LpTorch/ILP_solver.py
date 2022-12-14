'''
    Solve the ILP problem given the information
'''
import numpy as np
from pulp import *
import pulp as plp
import torch
from torchvision import models
import pandas as pd

# Hutchinson_trace = np.array([0.19270050525665214, 0.026702478528024147, 0.09448622912168206, 0.07639402896166123, 0.009688886813818761, 0.006711237132549379, 0.024378638714551495, 0.011769247241317709, 0.00250671175308565, 0.008039989508690044, 0.008477769792080075, 0.0017263438785415808, 0.00313899479806392, 0.004365175962448905, 0.0015507987700405636, 0.0015619333134963605, 0.002701229648662333, 0.002383587881922602, 0.0010320576839146385, 0.0025724433362483905, 0.0046385480090998495, 0.0008540530106982067, 0.003809824585913301, 0.0019535077735770134, 0.0005609235377052988, 0.0018696154002090816, 0.0016933275619534234, 0.000607571622821307, 0.00044498199713291893, 0.0008382285595869651, 0.0006031448720029394, 0.0003951200051233584, 0.000718781026080718, 0.0008380718063555912, 0.00037973490543638077, 0.000828751828521126, 0.0013068615226087446, 0.0004476536996660732, 0.0012669296702365405, 0.0013098433846609064, 0.00032695921254355777, 0.0012137993471687096, 0.0010635341750463423, 0.0002480300900066659, 0.0005943655269220451, 0.0007215500227179646, 0.000612712872680123, 0.00025412911781990733, 0.0005049526807847556, 0.0007025265367691074, 0.0002485941222403296, 0.000525604293216296])
solver = plp.PULP_CBC_CMD(msg=False)

def select_bit_with_ILP(M_B=None, T_B=None, M_mem=None, M_L=None, M_sens=None, available_bits = [8, 16, 20, 32]):
    assert M_B is not None and T_B is not None and M_mem is not None and M_L is not None and M_sens is not None, "missing required computation term"
    
    available_bits_range = [i for i in range(len(available_bits))]
    layers = len(M_mem)
    
    # M_mem, M_L = get_constraints_matrix(config, available_bits)
    # # import pdb; pdb.set_trace()
    # M_mem = np.array(M_mem)
    # M_L = np.array(M_L)
    # # test sense, follows simple rule
    # M_sens = np.random.rand(layers, 1)
    # M_sens = np.insert(M_sens, 1, M_sens[:,0] * 0.9, axis=1)
    # M_sens = np.insert(M_sens, 2, M_sens[:,0] * 0.81, axis=1)
    # import pdb; pdb.set_trace()

    

    problem = LpProblem('Bit-Width Solver', LpMinimize)
    # variable, x represents layer idx, y represents bit width.
    # while given x, sum of the y_i should be 1 #select one bits
    x_vars  = {(i,j):plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i,j)) for i in range(layers) for j in available_bits_range}

    profiled_mem = {(i,j): M_mem[i,j] for i in range(layers) for j in available_bits_range}
    profiled_latency = {(i,j): M_L[i,j] for i in range(layers) for j in available_bits_range}
    profiled_sensitivity = {(i,j): M_sens[i,j] for i in range(layers) for j in available_bits_range}


    # constraint for choosing one bit from set
    for i in range(layers):
        problem.addConstraint(
        plp.LpConstraint(e=plp.lpSum(x_vars[i,j] for j in available_bits_range),
                        sense=plp.LpConstraintEQ,
                        rhs=1,
                        name="constraint_sole_bit{0}".format(i)))
    # import pdb; pdb.set_trace()
    # constraint for mem
    for i in range(layers):
        problem.addConstraint(
        plp.LpConstraint(e=plp.lpSum(x_vars[i,j] * profiled_mem[i,j] for i in range(layers) for j in available_bits_range),
                        sense=plp.LpConstraintLE,
                        rhs=M_B,
                        name="constraint_mem_constraints{0}".format(i)))
    # latency
    for i in range(layers):
        problem.addConstraint(
        plp.LpConstraint(e=plp.lpSum(x_vars[i,j] * profiled_latency[i,j] for i in range(layers) for j in available_bits_range),
                        sense=plp.LpConstraintLE,
                        rhs=T_B,
                        name="constraint_latency_constraints{0}".format(i)))
    
    objective = plp.lpSum(x_vars[i,j] * profiled_sensitivity[i,j] for i in range(layers) for j in available_bits_range)
    problem.sense = plp.LpMinimize
    problem.setObjective(objective)

    result = problem.solve(solver)
    if result == -1:
        print("Haven't found the ILP")
        return None, None
    out_value = objective.value()
    # import pdb; pdb.set_trace()

    opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])
    opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
    opt_df.reset_index(inplace=True)
    opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
    opt_df.drop(columns=["variable_object"], inplace=True)
    # opt_df.to_csv("./optimization_solution.csv")

    return opt_df, out_value


def select_bit_with_ILP_mem_only(M_B=None, M_mem=None, M_sens=None, available_bits = [8, 16, 20, 32], layer_name=None):
    assert M_B is not None and M_mem is not None and M_sens is not None, "missing required computation term"
    
    available_bits_range = [i for i in range(len(available_bits))]
    layers = len(M_mem)
    

    var_prefix = 'var'
    problem = LpProblem('Bit-Width Solver', LpMinimize)
    # variable, x represents layer idx, y represents bit width.
    # while given x, sum of the y_i should be 1 #select one bits
    idx_of_layers = [f"{i},{j}" for i in range(layers) for j in available_bits_range]
    x_vars = LpVariable.dicts(name=var_prefix, indexs=idx_of_layers, lowBound=0, upBound = 1, cat='Integer')

    profiled_mem = {f"{i},{j}": M_mem[i,j] for i in range(layers) for j in available_bits_range}
    profiled_sensitivity = {f"{i},{j}": M_sens[i,j] for i in range(layers) for j in available_bits_range}

    # constraint for choosing one bit from set
    for i in range(layers):
        problem += plp.lpSum(x_vars[f"{i},{j}"] for j in available_bits_range) == 1
    problem += plp.lpSum(x_vars[f"{i},{j}"] * profiled_mem[f"{i},{j}"] for i in range(layers) for j in available_bits_range) <= M_B

    problem += plp.lpSum(x_vars[f"{i},{j}"] * profiled_sensitivity[f"{i},{j}"] for i in range(layers) for j in available_bits_range)
    result = problem.solve(solver)

    # import pdb; pdb.set_trace()

    if result == -1:
        print("Haven't found the ILP")
        return None
    out_value = problem.objective
    # print(out_value)
    selection_result = {}
    # for i in range(layers):
    #     for j in available_bits_range:
    #         if x_vars[f"{i},{j}"] == 1:
    #             print(i,j)
    #             if layer_name is not None:
    #                 selection_result[layer_name[i]] = available_bits[j]
    #             break 
    for v in problem.variables():
        if v.varValue == 1:
            v_name =v.name
            splited_name = v_name.split('_')[1]
            layer_idx, bit_idx = splited_name.split(',')
            cur_ln = layer_name[int(layer_idx)]
            cur_lb = available_bits[int(bit_idx)]
            selection_result[cur_ln] = cur_lb
    # import pdb; pdb.set_trace()

    # opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])
    # opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
    # opt_df.reset_index(inplace=True)
    # opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
    # opt_df.drop(columns=["variable_object"], inplace=True)
    # # opt_df.to_csv("./optimization_solution.csv")

    return selection_result


if __name__ == '__main__':
    # test ILP solver 
    test_solver()
    # read the result and analysis
    # from utils import interpret_ILP_result
    # interpret_ILP_result("./optimization_solution.csv")
