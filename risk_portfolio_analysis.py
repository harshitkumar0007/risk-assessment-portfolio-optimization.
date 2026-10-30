"""Week 3: Risk assessment (VaR/CVaR) and portfolio optimisation on a 10-stock universe."""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, linprog
from scipy import sparse
from sklearn.covariance import LedoitWolf
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":.3,"font.size":9,
                     "axes.spines.top":False,"axes.spines.right":False})
np.random.seed(42); os.makedirs("figs", exist_ok=True)
R = {}
TICK = ["AAPL","AMZN","JPM","XOM","WMT","PFE","T","MA","GE","SBUX"]
RF, ANN, CAP, NOTIONAL = 0.02, 252, 0.35, 1_000_000
COL = {"Equal weight":"#7f7f7f","Min variance":"#1f77b4","Max Sharpe":"#d62728","Risk parity":"#2ca02c","Min CVaR":"#9467bd"}

# ============ 1. DATA ============
px = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")[TICK].loc["2006-06-01":].dropna()
ret = px.pct_change().dropna()                       # simple daily returns
R["data"] = {"start":str(ret.index[0].date()),"end":str(ret.index[-1].date()),"n":len(ret),"assets":TICK}
train, test = ret.loc[:"2014-12-31"], ret.loc["2015-01-01":]
R["split"] = {"train_n":len(train),"train_end":"2014-12-31","test_n":len(test),"test_start":str(test.index[0].date())}
ew_full = ret.mean(axis=1)                            # equal-weight (daily rebalanced) market proxy

# ============ 2. RISK METRICS ============
def var_cvar_hist(r, a=.95):
    q = np.quantile(r, 1-a); return -q, -r[r<=q].mean()
def var_normal(r, a=.95):
    z = stats.norm.ppf(1-a); m, s = r.mean(), r.std(ddof=1)
    return -(m+z*s), -(m - s*stats.norm.pdf(z)/(1-a))
def var_t(r, a=.95):
    df, m, s = stats.t.fit(r); q = stats.t.ppf(1-a, df)
    es = -(m - s*(stats.t.pdf(q, df)/(1-a))*((df+q**2)/(df-1)))
    return -(m+s*q), es
def var_cf(r, a=.95):
    z = stats.norm.ppf(1-a); S, K = stats.skew(r), stats.kurtosis(r)
    zc = z + (z**2-1)*S/6 + (z**3-3*z)*K/24 - (2*z**3-5*z)*S**2/36
    return -(r.mean()+r.std(ddof=1)*zc)
def var_mc(r, a=.95, n=200_000):
    sim = np.random.normal(r.mean(), r.std(ddof=1), n); return var_cvar_hist(sim, a)
def maxdd(r):
    c = (1+r).cumprod(); return float((c/c.cummax()-1).min())
def perf(r, rf=RF):
    n = len(r); cagr = (1+r).prod()**(ANN/n)-1; vol = r.std(ddof=1)*np.sqrt(ANN)
    dn = r[r<0].std(ddof=1)*np.sqrt(ANN); v95,c95 = var_cvar_hist(r.values,.95); v99,c99 = var_cvar_hist(r.values,.99)
    dd = maxdd(r)
    return {"CAGR":cagr,"Vol":vol,"Sharpe":(r.mean()*ANN-rf)/vol,"Sortino":(r.mean()*ANN-rf)/dn,"MaxDD":dd,
            "Calmar":cagr/abs(dd),"VaR95":v95,"CVaR95":c95,"VaR99":v99,"CVaR99":c99}

asset_rows = {}
for t in TICK:
    r = ret[t]; p = perf(r); p.update({"Skew":stats.skew(r),"ExKurt":stats.kurtosis(r)}); asset_rows[t] = p
R["asset_risk"] = asset_rows
R["corr_mean"] = float(ret.corr().values[np.triu_indices(10,1)].mean())

methods = {}
for a in (.95,.99):
    h = var_cvar_hist(ew_full.values,a); n_ = var_normal(ew_full.values,a); t_ = var_t(ew_full.values,a); m_ = var_mc(ew_full.values,a)
    methods[f"{int(a*100)}"] = {"Historical":h,"Normal (parametric)":n_,"Student-t":t_,"Monte Carlo (normal)":m_,
                                "Cornish-Fisher":(var_cf(ew_full.values,a),None)}
R["var_methods"] = methods
R["ew_full_stats"] = {"skew":float(stats.skew(ew_full)),"exkurt":float(stats.kurtosis(ew_full)),"jb_p":float(stats.jarque_bera(ew_full)[1]),
                      "t_df":float(stats.t.fit(ew_full.values)[0])}
R["notional"] = NOTIONAL

# --- VaR backtest (rolling 250-day window) ---
def kupiec(x, N, p):
    ph = x/N
    ll0 = (N-x)*np.log(1-p) + (x*np.log(p) if x else 0)
    ll1 = (N-x)*np.log(1-ph) + (x*np.log(ph) if x else 0)
    lr = -2*(ll0-ll1); return float(lr), float(1-stats.chi2.cdf(lr,1))
def christoffersen_ind(h):
    h = np.asarray(h).astype(int); a,b = h[:-1],h[1:]
    n00=((a==0)&(b==0)).sum(); n01=((a==0)&(b==1)).sum(); n10=((a==1)&(b==0)).sum(); n11=((a==1)&(b==1)).sum()
    if n01+n11==0 or n01==0: return None, None
    pi0=n01/(n00+n01); pi1=n11/(n10+n11) if (n10+n11)>0 else 0; pi=(n01+n11)/(n00+n01+n10+n11)
    def L(p,n0,n1): return (n0*np.log(1-p) if p<1 else 0)+(n1*np.log(p) if p>0 else 0)
    lr = -2*(L(pi,n00+n10,n01+n11)-(L(pi0,n00,n01)+L(pi1,n10,n11))); return float(lr), float(1-stats.chi2.cdf(lr,1))
W=250; r_ = ew_full.values; bt = {}
vh = {99:[],95:[]}; vn={99:[],95:[]}
for t in range(W,len(r_)):
    win = r_[t-W:t]
    for a in (99,95):
        vh[a].append(-np.quantile(win,1-a/100)); vn[a].append(-(win.mean()+stats.norm.ppf(1-a/100)*win.std(ddof=1)))
act = r_[W:]
for a in (99,95):
    for nm,v in (("Historical",vh[a]),("Normal",vn[a])):
        hit = act < -np.array(v); x = int(hit.sum()); N = len(act); lr,p = kupiec(x,N,1-a/100); lri,pi_ = christoffersen_ind(hit)
        bt[f"{nm} {a}%"] = {"N":N,"exceptions":x,"expected":N*(1-a/100),"rate":x/N,"kupiec_p":p,"christ_ind_p":pi_}
R["var_backtest"] = bt
bt_dates = ew_full.index[W:]
hit99 = act < -np.array(vh[99])

# ============ 3. PLOTS: data & risk ============
fig,ax = plt.subplots(1,2,figsize=(9.5,3.8),gridspec_kw={"width_ratios":[1.25,1]})
(px/px.iloc[0]).plot(ax=ax[0],logy=True,lw=.9,legend=False); ax[0].set_title("Growth of $1 (log scale), 2006–2018"); ax[0].legend(ncol=2,fontsize=6.5)
im = ax[1].imshow(ret.corr(),cmap="RdBu_r",vmin=-1,vmax=1); ax[1].set_xticks(range(10)); ax[1].set_yticks(range(10))
ax[1].set_xticklabels(TICK,rotation=90,fontsize=7); ax[1].set_yticklabels(TICK,fontsize=7); ax[1].grid(False); ax[1].set_title("Return correlation")
for i in range(10):
    for j in range(10): ax[1].text(j,i,f"{ret.corr().values[i,j]:.2f}",ha="center",va="center",fontsize=5.5)
plt.tight_layout(); plt.savefig("figs/01_universe.png"); plt.close()

h95 = var_cvar_hist(ew_full.values,.95); h99 = var_cvar_hist(ew_full.values,.99)
fig,ax = plt.subplots(1,2,figsize=(9.5,3.6))
ax[0].hist(ew_full*100,bins=90,density=True,color="#9fb8d6",edgecolor="none"); xs=np.linspace(ew_full.min()*100,ew_full.max()*100,400)
ax[0].plot(xs,stats.norm.pdf(xs,ew_full.mean()*100,ew_full.std()*100),"r",lw=1,label="Normal fit")
for v,c,l in ((h95[0],"#ff7f0e","VaR 95%"),(h99[0],"#d62728","VaR 99%"),(h99[1],"k","CVaR 99%")): ax[0].axvline(-v*100,color=c,ls="--",lw=1.1,label=l)
ax[0].set_xlim(-12,12); ax[0].set_title("Equal-weight portfolio daily returns (%)"); ax[0].legend(fontsize=7)
stats.probplot(ew_full,plot=ax[1]); ax[1].set_title("Q–Q plot vs. normal")
plt.tight_layout(); plt.savefig("figs/02_distribution.png"); plt.close()

fig,ax = plt.subplots(figsize=(9.5,3.5))
ax.plot(bt_dates,act*100,color="#999",lw=.5,label="Daily return"); ax.plot(bt_dates,-np.array(vh[99])*100,color="#d62728",lw=1,label="−Historical VaR 99% (250d)")
ax.scatter(bt_dates[hit99],act[hit99]*100,color="k",s=14,zorder=5,label=f"Exceptions ({hit99.sum()})")
ax.set_ylabel("%"); ax.set_title("Rolling 99% VaR backtest, equal-weight portfolio"); ax.legend(fontsize=7,loc="lower right")
plt.tight_layout(); plt.savefig("figs/03_var_backtest.png"); plt.close()

# ============ 4. OPTIMISATION (train window) ============
def est(rw):
    mu = rw.mean().values*ANN; cov = LedoitWolf().fit(rw.values).covariance_*ANN; return mu,cov
mu, cov = est(train); n = len(TICK)
R["train_cov_shrinkage"] = float(LedoitWolf().fit(train.values).shrinkage_)
bnds = [(0,CAP)]*n; cons = [{"type":"eq","fun":lambda w:w.sum()-1}]; w0 = np.ones(n)/n
def vol_(w,c): return np.sqrt(w@c@w)
def opt_minvar(mu,cov): return minimize(lambda w:w@cov@w,w0,bounds=bnds,constraints=cons,method="SLSQP",options={"ftol":1e-12,"maxiter":500}).x
def opt_maxsharpe(mu,cov): return minimize(lambda w:-(w@mu-RF)/vol_(w,cov),w0,bounds=bnds,constraints=cons,method="SLSQP",options={"ftol":1e-12,"maxiter":500}).x
def rc(w,cov): s=vol_(w,cov); return w*(cov@w)/s
def opt_rp(mu,cov):
    f=lambda w: np.sum((rc(w,cov)/vol_(w,cov)-1/n)**2)
    return minimize(f,w0,bounds=bnds,constraints=cons,method="SLSQP",options={"ftol":1e-14,"maxiter":1000}).x
def opt_mincvar(rw,beta=.95,cap=CAP):
    Rm = rw.values; T = len(Rm); c = np.r_[np.zeros(n),1,np.full(T,1/((1-beta)*T))]
    A = sparse.hstack([sparse.csr_matrix(-Rm),-np.ones((T,1)),-sparse.eye(T)]).tocsr(); b=np.zeros(T)
    Aeq = np.r_[np.ones(n),0,np.zeros(T)].reshape(1,-1)
    res = linprog(c,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=[1],bounds=[(0,cap)]*n+[(None,None)]+[(0,None)]*T,method="highs"); return res.x[:n]
def opt_all(rw):
    m,c = est(rw)
    return {"Equal weight":w0.copy(),"Min variance":opt_minvar(m,c),"Max Sharpe":opt_maxsharpe(m,c),"Risk parity":opt_rp(m,c),"Min CVaR":opt_mincvar(rw)}
W_tr = opt_all(train)
R["weights"] = {k:dict(zip(TICK,map(float,v))) for k,v in W_tr.items()}
R["train_stats"] = {k:{"ret":float(w@mu),"vol":float(vol_(w,cov)),"sharpe":float((w@mu-RF)/vol_(w,cov))} for k,w in W_tr.items()}
R["risk_contrib"] = {k:dict(zip(TICK,map(float,rc(w,cov)/vol_(w,cov)))) for k,w in W_tr.items()}
R["asset_train"] = {t:{"mu":float(mu[i]),"vol":float(np.sqrt(cov[i,i]))} for i,t in enumerate(TICK)}

# efficient frontier (long-only, cap)
mvw = W_tr["Min variance"]; targets = np.linspace(mvw@mu, np.sort(mu)[-3:].mean()*0.98, 40)
front=[]
for tr in targets:
    c2 = cons+[{"type":"eq","fun":lambda w,tr=tr:w@mu-tr}]
    res = minimize(lambda w:w@cov@w,w0,bounds=bnds,constraints=c2,method="SLSQP",options={"ftol":1e-12,"maxiter":500})
    if res.success: front.append((np.sqrt(res.fun),tr))
front=np.array(front)
rand = np.random.dirichlet(np.ones(n),6000); rand = rand[(rand<=CAP).all(1)]
rr = rand@mu; rv = np.sqrt(np.einsum("ij,jk,ik->i",rand,cov,rand))
fig,ax = plt.subplots(figsize=(8.2,4.8))
sc=ax.scatter(rv*100,rr*100,c=(rr-RF)/rv,cmap="viridis",s=5,alpha=.5); plt.colorbar(sc,label="Sharpe ratio")
ax.plot(front[:,0]*100,front[:,1]*100,"k-",lw=2,label="Efficient frontier (long-only, 35% cap)")
for i,t in enumerate(TICK): ax.scatter(np.sqrt(cov[i,i])*100,mu[i]*100,marker="x",color="k",s=22); ax.annotate(t,(np.sqrt(cov[i,i])*100,mu[i]*100),fontsize=7,xytext=(3,3),textcoords="offset points")
for k,w in W_tr.items(): ax.scatter(vol_(w,cov)*100,(w@mu)*100,s=90,color=COL[k],edgecolor="k",zorder=6,label=k)
ax.set_xlabel("Annualised volatility (%)"); ax.set_ylabel("Annualised expected return (%)"); ax.set_title("Efficient frontier, training window 2006–2014"); ax.legend(fontsize=7,loc="lower right")
plt.tight_layout(); plt.savefig("figs/04_frontier.png"); plt.close()

fig,ax = plt.subplots(1,2,figsize=(9.5,3.8))
wdf = pd.DataFrame(W_tr,index=TICK).T; wdf.plot(kind="bar",stacked=True,ax=ax[0],width=.75,colormap="tab20",legend=False); ax[0].set_ylabel("Weight"); ax[0].set_title("Optimised weights (training window)"); ax[0].tick_params(axis="x",rotation=20)
rcdf = pd.DataFrame(R["risk_contrib"]).T; rcdf.plot(kind="bar",stacked=True,ax=ax[1],width=.75,colormap="tab20",legend=False); ax[1].set_title("Share of portfolio risk by asset"); ax[1].tick_params(axis="x",rotation=20)
h,l = ax[0].get_legend_handles_labels(); fig.legend(h,l,ncol=10,fontsize=7,loc="lower center",bbox_to_anchor=(.5,-.02))
plt.tight_layout(rect=(0,.06,1,1)); plt.savefig("figs/05_weights.png"); plt.close()

# ============ 5. SIMULATION ENGINE ============
def simulate(rets, sched, cost_bps=0.0):
    """sched: list of (start_pos, weights). Buy-and-hold between rebalances; returns daily series and turnovers."""
    out = pd.Series(index=rets.index,dtype=float); prev_end=None; tos=[]
    for i,(s,w) in enumerate(sched):
        e = sched[i+1][0] if i+1<len(sched) else len(rets); blk = rets.iloc[s:e]
        cum = (1+blk).cumprod().values; v = cum@w; pv = np.r_[1,v[:-1]]; pr = v/pv-1
        if prev_end is not None:
            to = np.abs(w-prev_end).sum(); tos.append(to); pr[0] -= cost_bps/1e4*to
        out.iloc[s:e] = pr; prev_end = w*cum[-1]/v[-1]
    return out.dropna(), tos

# ---- Static weights, out-of-sample 2015–2018, annual rebalance to target ----
sched_oos = lambda w: [(p,w) for p in range(0,len(test),252)]
oos = {k:simulate(test,sched_oos(w))[0] for k,w in W_tr.items()}
R["oos_perf"] = {k:perf(r) for k,r in oos.items()}
ins = {k:simulate(train,[(p,w) for p in range(0,len(train),252)])[0] for k,w in W_tr.items()}
R["ins_perf"] = {k:perf(r) for k,r in ins.items()}
fig,ax = plt.subplots(2,1,figsize=(8.8,6),sharex=True,gridspec_kw={"height_ratios":[2,1]})
for k,r in oos.items():
    c=(1+r).cumprod(); ax[0].plot(c,color=COL[k],lw=1.3,label=k); ax[1].plot(c/c.cummax()-1,color=COL[k],lw=.9)
ax[0].set_title("Out-of-sample growth of $1 (Jan 2015 – Apr 2018), weights fixed from 2006–2014 data"); ax[0].legend(fontsize=7); ax[1].set_title("Drawdown",fontsize=9)
plt.tight_layout(); plt.savefig("figs/06_oos.png"); plt.close()

# ---- Walk-forward re-optimisation (3y lookback, annual rebalance), 2009-06 → 2018-04, with 10 bps cost ----
LB = 756; starts = list(range(LB,len(ret),252)); wf_w = {k:[] for k in COL}
for s in starts:
    ow = opt_all(ret.iloc[s-LB:s])
    for k in COL: wf_w[k].append(ow[k])
wf, wf_to = {}, {}
for k in COL:
    r,to = simulate(ret,[(s,w) for s,w in zip(starts,wf_w[k])],cost_bps=10); wf[k]=r.loc[ret.index[starts[0]]:]; wf_to[k]=float(np.mean(to))
R["wf"] = {"n_rebal":len(starts),"first":str(ret.index[starts[0]].date()),"perf":{k:perf(r) for k,r in wf.items()},"avg_turnover":wf_to}
R["wf_avg_weights"] = {k:dict(zip(TICK,map(float,np.mean(wf_w[k],axis=0)))) for k in COL}
fig,ax = plt.subplots(figsize=(8.8,4))
for k,r in wf.items(): ax.plot((1+r).cumprod(),color=COL[k],lw=1.2,label=k)
ax.set_yscale("log"); ax.set_title("Walk-forward backtest: annual re-optimisation, 3-year lookback, 10 bps costs"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/07_walkforward.png"); plt.close()

# ============ 6. SCENARIO ANALYSIS ============
# (a) historical stress episodes (buy & hold from episode start)
eps = {"GFC (Oct 2007 – Mar 2009)":("2007-10-09","2009-03-09"),"US downgrade (Jul – Oct 2011)":("2011-07-22","2011-10-03"),
       "China/oil sell-off (Aug 2015 – Feb 2016)":("2015-08-18","2016-02-11"),"Feb 2018 correction":("2018-01-26","2018-02-08")}
stress={}
for nm,(a,b) in eps.items():
    seg = ret.loc[a:b]; stress[nm] = {k: float((1+seg).cumprod().iloc[-1].values @ w - 1) for k,w in W_tr.items()}
R["stress"] = stress
fig,ax = plt.subplots(figsize=(9,3.8)); xi=np.arange(len(eps)); bw=.16
for j,k in enumerate(COL): ax.bar(xi+(j-2)*bw,[stress[e][k]*100 for e in eps],bw,color=COL[k],label=k)
ax.set_xticks(xi); ax.set_xticklabels([e.replace(" (","\n(") for e in eps],fontsize=7); ax.set_ylabel("Return (%)"); ax.axhline(0,color="k",lw=.6)
ax.set_title("Historical stress replay (fixed training-window weights)"); ax.legend(fontsize=7,ncol=3); plt.tight_layout(); plt.savefig("figs/08_stress.png"); plt.close()

# (b) Monte-Carlo regimes, 1-year horizon, multivariate Student-t (df=5)
corr = cov/np.outer(np.sqrt(np.diag(cov)),np.sqrt(np.diag(cov))); sd = np.sqrt(np.diag(cov))
def regime(dmu,vm,lam):
    C = (1-lam)*corr+lam*np.ones((n,n)); np.fill_diagonal(C,1); s = sd*vm; return mu+dmu, np.outer(s,s)*C
REG = {"Bull":(+0.05,0.8,0.0),"Base":(0,1.0,0.0),"Bear":(-0.15,1.3,0.2),"Crisis":(-0.35,2.0,0.5)}
NS, DF, H = 10000, 5, 252; mc={}; paths={}
for rn,(dmu,vm,lam) in REG.items():
    m_,c_ = regime(dmu,vm,lam); L = np.linalg.cholesky(c_/ANN)
    z = np.random.standard_normal((NS,H,n)); g = np.random.chisquare(DF,(NS,H,1))/DF
    daily = m_/ANN + (z/np.sqrt(g))*np.sqrt((DF-2)/DF) @ L.T     # t-distributed, unit-variance scaling
    mc[rn]={}
    for k,w in W_tr.items():
        pr = daily@w; cum = np.cumprod(1+pr,axis=1); term = cum[:,-1]-1; dd = (cum/np.maximum.accumulate(cum,axis=1)-1).min(axis=1)
        q5 = np.quantile(term,.05)
        mc[rn][k] = {"mean":float(term.mean()),"p5":float(q5),"cvar5":float(term[term<=q5].mean()),"p_loss":float((term<0).mean()),
                     "p_loss20":float((term<-.2).mean()),"med_dd":float(np.median(dd))}
        if k in ("Equal weight","Min variance","Max Sharpe") : paths[(rn,k)] = term
R["mc_regimes"] = {"params":{k:{"d_mu":v[0],"vol_mult":v[1],"corr_shift":v[2]} for k,v in REG.items()},"results":mc,"sims":NS}
fig,ax = plt.subplots(1,4,figsize=(10,3),sharey=True)
for a_,rn in zip(ax,REG):
    for k in ("Equal weight","Min variance","Max Sharpe"): a_.hist(paths[(rn,k)]*100,bins=60,alpha=.45,color=COL[k],label=k,density=True)
    a_.set_title(rn,fontsize=9); a_.set_xlabel("1-yr return (%)"); a_.set_xlim(-100,200)
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig("figs/09_mc.png"); plt.close()

# (c) market-beta shocks
bm = np.array([np.cov(ret[t],ew_full)[0,1]/ew_full.var() for t in TICK]); R["betas"]=dict(zip(TICK,map(float,bm)))
R["port_beta"]={k:float(w@bm) for k,w in W_tr.items()}
R["beta_shock"]={f"{int(s*100)}%":{k:float(w@bm*s) for k,w in W_tr.items()} for s in (-.10,-.20,-.30)}
json.dump(R,open("results.json","w"),indent=1,default=float)
print("done")
