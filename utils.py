import numpy as np
import math

# The code is for population gradient of two-layered ReLU network with zero-mean Gaussian
# input.

def single_term(w_r, w):
    norm_w_r = np.linalg.norm(w_r)
    norm_w = np.linalg.norm(w)
    corr = np.dot(w_r, w) / norm_w_r / norm_w
    theta = math.acos(min(max(corr, -1), 1))
    deltaw = ((math.pi - theta) * w + norm_w / norm_w_r * math.sin(theta) * w_r) / 2 / math.pi
    return deltaw

def gradient_with_single_term(ws, ws_star):
    d = ws.shape[0]
    n_node = ws.shape[1]

    deltaws1 = np.zeros((d, n_node))
    deltaws2 = np.zeros((d, n_node))
    for i in range(n_node):
        for j in range(n_node):
            deltaws1[:,i] += single_term(ws[:,i], ws_star[:,j])
            deltaws2[:,i] += single_term(ws[:,i], ws[:,j])
    return deltaws1 - deltaws2

def compute_term(ws_ref, ws, ws_ref_norm, ws_norm):
    # [#ref, #w]
    corrs = np.dot(ws_ref.T, ws) / ws_ref_norm[:,None] / ws_norm[None,:]
    corrs[corrs<-1] = -1
    corrs[corrs>1] = 1

    theta = np.arccos(corrs)
    alpha = ws_norm[None,:] / ws_ref_norm[:,None]
    # [d, #ref, #w]
    deltaw1 = (math.pi - theta[None,:,:]) * ws[:,None,:]
    deltaw2 = (alpha * np.sin(theta))[None,:,:] * ws_ref[:,:,None]
    deltaw = np.sum(deltaw1 + deltaw2, axis=2) / math.pi / 2
    # [d, #ref]
    return deltaw

def compute_gradient(ws, ws_star):
    ws_norm = np.linalg.norm(ws, axis=0)
    ws_star_norm = np.linalg.norm(ws_star, axis=0)

    deltaw_star = compute_term(ws, ws_star, ws_norm, ws_star_norm)
    deltaw_self = compute_term(ws, ws, ws_norm, ws_norm)
    #
    return deltaw_star - deltaw_self

# Compute gradient in a different way.
def construct_M(es, es_ref):
    n_ref = es_ref.shape[1]
    n = es.shape[1]
    # Input: es_ref: [d, #wref], es: [d, #w]
    # [#wref, #wref]
    cos_theta_ref = np.dot(es_ref.T, es_ref)

    # [#w, #wref], [k, i]
    cos_theta = np.dot(es.T, es_ref)
    cos_theta = np.clip(cos_theta, -1, 1)

    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    # [k, i, j], [#w, #ref, #ref]
    M = (math.pi - theta)[:, :, None] * cos_theta[:, None, :] + sin_theta[:, :, None] * cos_theta_ref[None, :, :]
    # Make it [#ref #ref #w]
    return np.transpose(M, (1, 2, 0))

def split(M):
    n = M.shape[2]
    n_ref = M.shape[0]

    # Get the core of M
    M_r = np.zeros((n_ref, n))
    M_off = np.zeros((n_ref*(n_ref-1), n))

    counter = 0
    for i in range(n_ref):
        for j in range(n_ref):
            if i == j:
                M_r[i, :] = M[i, i, :]
            else:
                M_off[counter, :] = M[i, j, :]
                counter += 1

    return M_r, M_off

def compute_gradient_new(ws, ws_star):
    ws_norm = np.linalg.norm(ws, axis=0)
    ws_star_norm = np.linalg.norm(ws_star, axis=0)
    es = ws / ws_norm[None, :]
    es_star = ws_star / ws_star_norm[None,:]

    n = ws.shape[1]

    M_star = construct_M(es_star, es).reshape(n*n, n)
    M = construct_M(es, es).reshape(n*n, n)
    diff = np.dot(M_star, ws_star_norm) - np.dot(M, ws_norm)
    return diff.reshape(n, n) / 2 / math.pi

