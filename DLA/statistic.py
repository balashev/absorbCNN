def get_stats(ind, d=None, model=None, sdss=None):
    if d != None:
        specs, loglams, *other = d.get_spec(ind, sdss=sdss)

        if model != None:
            preds = model.predict(specs)
    dla = []

    m = (preds[0] > 0.2).flatten()
    if sum(m) > 3:
        zd = 10 ** (loglams[m] - preds[1].flatten()[m] * 1e-4) / lya - 1
        z = distr1d(zd, bandwidth=0.7)
        z.stats()
        z.plot()
        zint = [min(zd)] + list(z.x[argrelextrema(z.inter(z.x), np.less)[0]]) + [max(zd)]
        zmax = z.x[argrelextrema(z.inter(z.x), np.greater)]
        for i in range(len(zint)-1):
            mz = (z.x > zint[i]) * (z.x < zint[i+1])
            if max([z.inter(x) for x in z.x[mz]]) > z.inter(z.point) / 3:
                mz = (zd > zint[i]) * (zd < zint[i+1])
                if sum(mz) > 3:
                    dla.append([ind, np.median(preds[0][m][mz])])
                    z1 = distr1d(zd[mz])
                    z1.kde(bandwidth=0.1)
                    z1.stats()
                    N = preds[2].flatten()[m][mz]
                    N = distr1d(N)
                    N.stats()
                    dla[-1].extend([z1.point] + list(z1.interval - z1.point))
                    dla[-1].extend([N.point] + list(N.interval - N.point))
                    dla[-1]
    return dla