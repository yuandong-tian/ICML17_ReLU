d = 2
n = 2

angles = [3*math.pi/8, 7*math.pi/8]
for i, angle in enumerate(angles):
    es = np.zeros((d, n))
    #ws_star = np.eye(d)
    es[:,0] = [1, 0]
    es[:,1] = [np.cos(angle), np.sin(angle)]
    M = construct_M(es, es)
    M_r, M_off = split(M)
    coeffs = np.linalg.solve(M_r, M_off.T)

    w_star_angles = np.linspace(0, 2*math.pi, 100)
    es_star = np.zeros((d, w_star_angles.shape[0]))
    es_star[0,:] = np.cos(w_star_angles)
    es_star[1,:] = np.sin(w_star_angles)
    M_star = construct_M(es_star, es)
    M_star_r, M_star_off = split(M_star)

    # L_12 and L_21
    diff = M_star_off.T - np.dot(M_star_r.T, coeffs)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    show_res(es[:2,:], es_star[:2,:], diff[:,0]>=0)

    plt.subplot(122)
    #show_res(es[:2,:], es_star[:2,:], diff[:,1]>=0)
    plt.plot(w_star_angles, diff[:,0], linewidth=2, label='L_12')
    plt.plot(w_star_angles, diff[:,1], linewidth=2, label='L_21')
    #plt.plot(w_star_angles, diff[:,1]+diff[:,0])
    plt.plot(w_star_angles, np.zeros_like(w_star_angles))
    plt.legend()
    #plt.show()
    plt.savefig('score-%d.pdf' % i)
