from math import ceil
import numpy as np
from tqdm import tqdm


class MCCF(object):
    def __init__(self, evt1, evt2, bin_width, deltaT, tmin, tmax, searchtime):
        """
        evt1,evt2:array
        bin_width:float
        deltaT:float
        tmin,tmax:float
        searchtime:array like,two values.
        """
        self.evt1 = evt1
        self.evt2 = evt2
        self.bin_width = bin_width
        self.deltaT = deltaT
        self.tmin = tmin
        self.tmax = tmax
        self.searchtime = searchtime
        self.bins = ceil((tmax - self.tmin) / bin_width)
        self.all_lag = np.arange(searchtime[0], searchtime[1], deltaT)
        self.lc1, _ = np.histogram(a=evt1, bins=self.bins, range=(tmin, tmax))
        self.lc1err = np.sqrt(self.lc1)
        self.mccfvalue = self._mccfvalue()
        self.maxlag = np.around(self.all_lag[np.argmax(self.mccfvalue)], 10)
        self.lagerr = 0

    def _mccfvalue(self):
        """
        返回mccf值.if max lag is positive,then evt2 is latter than evt1.
        """
        v1 = self.lc1 - np.mean(self.lc1)
        sig1 = np.sqrt(sum(v1 ** 2))
        v1 = v1 / sig1
        LC2 = np.array([(np.histogram(a=self.evt2, bins=self.bins,
                                      range=(self.tmin + lag, self.tmax + lag)))[0] for lag in self.all_lag])
        V2 = LC2 - LC2.mean(axis=1).reshape(LC2.shape[0], 1)
        SIG2 = np.sqrt((V2 ** 2).sum(axis=1).reshape(LC2.shape[0], 1))
        V2 = V2 / SIG2
        all_mccf = (v1 * V2).sum(axis=1)
        return all_mccf

    def evaluateLagErr(self, NMC=1000, method='1'):
        """
        返回lag误差，method='1','2','3'分别对应原始光变加高斯随机数与误差之积，原始光变服从正态分布，原始光变服从泊松分布。
        """
        self.LAG = []
        LC2 = np.array([(np.histogram(a=self.evt2, bins=self.bins,
                                      range=(self.tmin + lag, self.tmax + lag)))[0] for lag in self.all_lag])
        LC2err = np.sqrt(LC2)
        lengthlc = len(self.lc1)
        if method == '1':
            for i in tqdm(range(NMC)):
                randomlc1 = self.lc1 + self.lc1err * \
                    np.random.normal(0, 1, lengthlc)
                v1 = randomlc1 - np.mean(randomlc1)
                sig1 = np.sqrt(sum(v1 ** 2))
                v1 = v1 / sig1
                randomLC2 = LC2 + LC2err * \
                    np.random.normal(0, 1, [len(self.all_lag), lengthlc])
                V2 = randomLC2 - \
                    randomLC2.mean(axis=1).reshape(randomLC2.shape[0], 1)
                SIG2 = np.sqrt(
                    (V2 ** 2).sum(axis=1).reshape(randomLC2.shape[0], 1))
                V2 = V2 / SIG2
                all_mccf = (v1 * V2).sum(axis=1)
                self.LAG.append(self.all_lag[np.argmax(all_mccf)])
            self.lagerr = round(np.std(self.LAG), 5)
        elif method == '2':
            for i in tqdm(range(NMC)):
                randomlc1 = np.random.normal(self.lc1, self.lc1err, lengthlc)
                v1 = randomlc1 - np.mean(randomlc1)
                sig1 = np.sqrt(sum(v1 ** 2))
                v1 = v1 / sig1
                randomLC2 = np.random.normal(
                    LC2, LC2err, [len(self.all_lag), lengthlc])
                V2 = randomLC2 - \
                    randomLC2.mean(axis=1).reshape(randomLC2.shape[0], 1)
                SIG2 = np.sqrt(
                    (V2 ** 2).sum(axis=1).reshape(randomLC2.shape[0], 1))
                V2 = V2 / SIG2
                all_mccf = (v1 * V2).sum(axis=1)
                self.LAG.append(self.all_lag[np.argmax(all_mccf)])
            self.lagerr = round(np.std(self.LAG), 5)
        elif method == '3':
            for i in tqdm(range(NMC)):
                randomlc1 = np.random.poisson(self.lc1)
                v1 = randomlc1 - np.mean(randomlc1)
                sig1 = np.sqrt(sum(v1 ** 2))
                v1 = v1 / sig1
                randomLC2 = np.random.poisson(
                    lam=LC2, size=[len(self.all_lag), lengthlc])
                V2 = randomLC2 - \
                    randomLC2.mean(axis=1).reshape(randomLC2.shape[0], 1)
                SIG2 = np.sqrt(
                    (V2 ** 2).sum(axis=1).reshape(randomLC2.shape[0], 1))
                V2 = V2 / SIG2
                all_mccf = (v1 * V2).sum(axis=1)
                self.LAG.append(self.all_lag[np.argmax(all_mccf)])
            self.lagerr = round(np.std(self.LAG), 5)
        else:
            raise ValueError('Method out of range!')
