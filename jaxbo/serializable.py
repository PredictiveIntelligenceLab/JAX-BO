import numpy as onp
import jax.numpy as np
import json 






def serializable_MF(opt_params_list, X_f_L, y_f_L, X_f_H, y_f_H, X_c_L_list, y_c_L_list, X_c_H_list, y_c_H_list, bounds, gmm_vars):
    return_params = []
    for k in range(len(opt_params_list)):
        return_params.append(opt_params_list[k].tolist())

    return_data = {}
    return_data["X_f_L"] = X_f_L.tolist()
    return_data["y_f_L"] = y_f_L.tolist()
    return_data["X_f_H"] = X_f_H.tolist()
    return_data["y_f_H"] = y_f_H.tolist()

    return_constraints = []
    for k in range(len(X_c_L_list)):
        temp_constraints = {}
        temp_constraints["X_c_L"] = X_c_L_list[k].tolist()
        temp_constraints["y_c_L"] = y_c_L_list[k].tolist()
        temp_constraints["X_c_H"] = X_c_H_list[k].tolist()
        temp_constraints["y_c_H"] = y_c_H_list[k].tolist()
        return_constraints.append(temp_constraints)

    return_bounds = {}
    return_bounds["lb"] = bounds["lb"].tolist()
    return_bounds["ub"] = bounds["ub"].tolist()

    return_gmm_vars = []
    for k in range(len(gmm_vars)):
        return_gmm_vars.append(gmm_vars[k].tolist())

    return_dictionary = [return_params, return_data, return_constraints, return_bounds, return_gmm_vars]

    return return_dictionary









def deserializable_MF(return_dictionary):

    return_params, return_data, return_constraints, return_bounds, return_gmm_vars = return_dictionary

    opt_params_list = []
    for k in range(len(return_params)):
        opt_params_list.append(np.array(return_params[k]))

    X_f_L = np.array(return_data["X_f_L"])
    y_f_L = np.array(return_data["y_f_L"])
    X_f_H = np.array(return_data["X_f_H"])
    y_f_H = np.array(return_data["y_f_H"])

    X_c_L_list = [] 
    y_c_L_list = []
    X_c_H_list = []
    y_c_H_list = []
    for k in range(len(return_constraints)):
        X_c_L_list.append(np.array(return_constraints[k]["X_c_L"]))
        y_c_L_list.append(np.array(return_constraints[k]["y_c_L"]))
        X_c_H_list.append(np.array(return_constraints[k]["X_c_H"]))
        y_c_H_list.append(np.array(return_constraints[k]["y_c_H"]))

    bounds = {}
    bounds["lb"] = np.array(return_bounds["lb"])
    bounds["ub"] = np.array(return_bounds["ub"])

    gmm_vars = []
    for k in range(len(return_gmm_vars)):
        gmm_vars.append(np.array(return_gmm_vars[k]))

    return opt_params_list, X_f_L, y_f_L, X_f_H, y_f_H, X_c_L_list, y_c_L_list, X_c_H_list, y_c_H_list, bounds, gmm_vars


