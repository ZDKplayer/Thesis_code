import numpy as np
from stingray import AveragedPowerspectrum, AveragedCrossspectrum
import matplotlib.pyplot as plt
from astropy.modeling.models import Lorentz1D
from math import ceil
from stingray.simulator import simulator, base
from stingray.events import EventList
from stingray.lightcurve import Lightcurve
import warnings
warnings.filterwarnings('ignore')

# 模拟满足给定功率谱的光变曲线
def simu_lor(v0, fwhm, mean_rate, rms, duration):
    """
    v0:中心频率
    fwhm:半高全宽
    mean_rate:平均计数率
    rms: rms
    duration:光变的持续时间
    """
    model = Lorentz1D(amplitude=1, x_0=v0, fwhm=fwhm)  # 洛伦兹模型
    dt = 0.001  # 光变曲线的时间bin宽
    N = ceil(duration / dt)  # 光变bin的点数
    mean = mean_rate * dt  # 源光变的平均值
    simu = simulator.Simulator(
        N=N,
        mean=mean,
        dt=dt,
        rms=rms,
        tstart=0)  # 初始化光变
    lc = simu.simulate(model)  # lor光变成分
    lc.counts[lc.counts < 0] = 0  # 防止计数小于0
    return EventList(base.simulate_times(lc=lc))


r1evt = simu_lor(v0=0, fwhm=3, mean_rate=2000, rms=0.3, duration=5000)
r2evt = simu_lor(v0=0, fwhm=4, mean_rate=2000, rms=0.2, duration=5000)
q1evt = simu_lor(v0=1, fwhm=0.1, mean_rate=2000, rms=0.15, duration=5000)
q2evt = simu_lor(v0=1, fwhm=0.2, mean_rate=2000, rms=0.10, duration=5000)
dt = 0.01
r1lc = r1evt.to_lc(dt=dt, tstart=0, tseg=5000)
r2lc = r2evt.to_lc(dt=dt, tstart=0, tseg=5000)
q1lc = q1evt.to_lc(dt=dt, tstart=0, tseg=5000)
q2lc = q2evt.to_lc(dt=dt, tstart=0, tseg=5000)

# 给定相位延迟的理论谱
freq = np.fft.rfftfreq(n=q1lc.n, d=q1lc.dt)


def Q_dis_phi(f):
    return -0.5 * np.exp(-(f - 1)**2 / (0.2**2))


def R_dis_phi(f):
    return np.ones(len(f)) * 0.5


q_dis_phi = Q_dis_phi(freq)
r_dis_phi = R_dis_phi(freq)

# 不改变两信号的功率谱，但是使两信号的时延谱满足给定的分布


def make_lag_lc(lc1, lc2, dis_phi):
    S1 = np.fft.rfft(lc1.counts)
    S2 = np.fft.rfft(lc2.counts)
    mo = abs(S1) * abs(S2)
    cp_R = mo * np.cos(dis_phi)
    cp_I = mo * np.sin(dis_phi)
    cp = np.array([complex(r, i) for r, i in zip(cp_R, cp_I)])
    cp[0] = mo[0]
    newS2 = cp / np.conjugate(S1)
    newlc2 = Lightcurve(
        time=lc2.time,
        counts=np.fft.irfft(newS2),
        dt=lc2.dt,
        skip_checks=True)
    return lc1, newlc2


nq1lc, nq2lc = make_lag_lc(q1lc, q2lc, q_dis_phi)
nr1lc, nr2lc = make_lag_lc(r1lc, r2lc, r_dis_phi)
nq2lc.counts[nq2lc.counts < 0] = 0
nr2lc.counts[nr2lc.counts < 0] = 0

# 计算功率谱
segment_size=20
r1ps = AveragedPowerspectrum(lc=nr1lc, segment_size=segment_size, norm='leahy')
r2ps = AveragedPowerspectrum(lc=nr2lc, segment_size=segment_size, norm='leahy')
q1ps = AveragedPowerspectrum(lc=nq1lc, segment_size=segment_size, norm='leahy')
q2ps = AveragedPowerspectrum(lc=nq2lc, segment_size=segment_size, norm='leahy')
df = 0.03
q1ps_rebin = q1ps.rebin_log(df)
q2ps_rebin = q2ps.rebin_log(df)
r1ps_rebin = r1ps.rebin_log(df)
r2ps_rebin = r2ps.rebin_log(df)

# 根据交叉谱计算相位延迟谱


def get_lag(q1lc, q2lc):
    acp = AveragedCrossspectrum(lc1=q1lc, lc2=q2lc, segment_size=segment_size)
    acp = acp.rebin_log(f=0.03)
    tlag, tlagerr = acp.time_lag()
    plag, plagerr = tlag * \
        (2 * np.pi * acp.freq), tlagerr * (2 * np.pi * acp.freq)
    lag_f = acp.freq
    return lag_f, plag, plagerr


lag_f, lag12, lag12err = get_lag(nq2lc, nq1lc)
_, lag34, lag34err = get_lag(nr2lc, nr1lc)

# 画图
fig = plt.figure(dpi=100,figsize=(12,6))
ax = fig.add_subplot(121)
ax.errorbar(q1ps_rebin.freq, q1ps_rebin.power,
            yerr=q1ps_rebin.power_err, color='black', fmt='--', label=r'$P_{q_1}$')
ax.errorbar(q2ps_rebin.freq, q2ps_rebin.power,
            yerr=q2ps_rebin.power_err, color='b', fmt='--', label=r'$P_{q_2}$')
ax.errorbar(r1ps_rebin.freq, r1ps_rebin.power,
            yerr=r1ps_rebin.power_err, color='r', fmt='--', label=r'$P_{r_1}$')
ax.errorbar(r2ps_rebin.freq, r2ps_rebin.power,
            yerr=r2ps_rebin.power_err, color='g', fmt='--', label=r'$P_{r_2}$')
ax.set_xlabel('frequency (Hz)', fontsize=14)
ax.set_ylabel('power (leahy)', fontsize=14)
ax.legend(fontsize=13)
ax.set_xlabel('frequency (Hz)', fontsize=14)
ax.set_ylabel('power (leahy)', fontsize=14)
ax.text(x=0.02, y=0.95, s='a', transform=ax.transAxes, fontsize=13)
ax.set_xlim(lag_f[0], lag_f[-1])
ax.loglog()
ax.tick_params(labelsize=13)

ax = fig.add_subplot(122)
ax.errorbar(
    lag_f,
    lag34,
    lag34err,
    fmt='s',
    color='b',
    label=r'simulation $\Delta\phi(r_2,r_1)$',
    markersize=3)
ax.errorbar(
    lag_f,
    lag12,
    lag12err,
    fmt='s',
    color='black',
    label=r'simulation $\Delta\phi(q_2,q_1)$',
    markersize=3)
ax.plot(freq, r_dis_phi, 'r--')
ax.set_yticks(ticks=np.arange(-0.6, 1, 0.1))
ax.set_ylim(-0.56, 1)
ax.set_xlim(lag_f[0], lag_f[-1])
ax.plot(freq, q_dis_phi, 'r--', label='theoretical curves')
ax.semilogx()
ax.text(x=0.02, y=0.95, s='b', transform=ax.transAxes, fontsize=13)
ax.legend(fontsize=13)
ax.set_xlabel('frequency (Hz)', fontsize=14)
ax.set_ylabel('phase lag (rad)', fontsize=14)
ax.tick_params(labelsize=13)
fig.tight_layout()
plt.show()
