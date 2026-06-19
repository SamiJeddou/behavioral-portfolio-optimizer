#!/usr/bin/env python3
"""Rolling-study figures: breach-by-universe, return distribution, and breach-vs-N."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=[".","/sessions/vigilant-nice-ride/mnt/PAPER"]
def wilson(k,n,z=1.96):
    if n==0:return 0,0,0
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return p,max(0,c-h),min(1,c+h)
def load(f):
    R=pd.read_csv(f);n=len(R);k=int(R.br1.sum());p,lo,hi=wilson(k,n);return R,n,k,p,lo,hi

# --- Fig 1 & 2: by universe (4 grid + pre-2008 large scalable) ---
SPEC=[("rb_AAPL_MSFT_JPM.csv","Stocks\n(grid)"),("rb_XLK_XLF_XLE.csv","Sectors\n(grid)"),
      ("rb_QQQ_TLT_GLD.csv","Multi-asset\n(grid)"),("rb_KO_JNJ_PG.csv","Defensive\n(grid)"),
      ("rb_N24.csv","Large 24\n(scalable)")]
labels=[];rate=[];lo=[];hi=[];rets=[];ns=[]
for f,lab in SPEC:
    if not os.path.exists(f):continue
    R,n,k,p,a,b=load(f);labels.append(lab);rate.append(p*100);lo.append((p-a)*100);hi.append((b-p)*100);rets.append(R.cum1.values*100);ns.append(n)
fig,ax=plt.subplots(figsize=(8,4.2));x=np.arange(len(labels))
ax.bar(x,rate,color="#4a76b8",width=0.6,yerr=[lo,hi],capsize=5,ecolor="#222")
ax.axhline(5,ls="--",color="#c0392b",lw=1.5,label="5% in-sample target")
for i,(r,n) in enumerate(zip(rate,ns)):ax.text(i,r+hi[i]+1.2,f"n={n}",ha="center",fontsize=8,color="#444")
ax.set_xticks(x);ax.set_xticklabels(labels,fontsize=9);ax.set_ylabel("Out-of-sample breach rate (%)")
ax.set_title("Breach rate of the $-10\\%$ floor by universe (Wilson 95% CI)")
ax.legend(frameon=False,fontsize=9);ax.set_ylim(0,max([r+h for r,h in zip(rate,hi)])+6);fig.tight_layout()
for d in OUT:fig.savefig(os.path.join(d,"rolling_breach.png"),dpi=150)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.2))
bp=ax.boxplot(rets,whis=(5,95),showfliers=False,patch_artist=True,widths=0.55)
for b in bp["boxes"]:b.set(facecolor="#cfe0f3",edgecolor="#33567f")
for m in bp["medians"]:m.set(color="#c0392b",lw=1.6)
ax.axhline(-10,ls="--",color="#c0392b",lw=1.4,label="$-10\\%$ floor")
ax.set_xticklabels(labels,fontsize=9);ax.set_ylabel("Realised 1-year window return (%)")
ax.set_title("Distribution of realised out-of-sample window returns");ax.legend(frameon=False,fontsize=9);fig.tight_layout()
for d in OUT:fig.savefig(os.path.join(d,"rolling_returns.png"),dpi=150)
plt.close(fig)

# --- Fig 3: breach vs N (fixed engine/constraint/cap, crash-spanning) ---
NS=[(5,"rb_N05.csv"),(10,"rb_N10.csv"),(15,"rb_N15.csv"),(24,"rb_N24.csv")]
xs=[];ys=[];el=[];eh=[];nn=[]
for N,f in NS:
    if not os.path.exists(f):continue
    R,n,k,p,a,b=load(f);xs.append(N);ys.append(p*100);el.append((p-a)*100);eh.append((b-p)*100);nn.append(n)
fig,ax=plt.subplots(figsize=(7.2,4.2))
ax.errorbar(xs,ys,yerr=[el,eh],fmt="o-",color="#1f4e8c",lw=2,capsize=5,ms=7)
ax.axhline(5,ls="--",color="#c0392b",lw=1.5,label="5% in-sample target")
for xx,yy,m in zip(xs,ys,nn):ax.text(xx,yy+eh[xs.index(xx)]+1.0,f"n={m}",ha="center",fontsize=8,color="#444")
ax.set_xlabel("Universe size N (equities)");ax.set_ylabel("Out-of-sample breach rate (%)")
ax.set_title("Breach rate vs universe size (scalable engine, $-20\\%$ CVaR floor, 20% cap)")
ax.set_xticks(xs);ax.legend(frameon=False,fontsize=9);ax.set_ylim(0,max([y+h for y,h in zip(ys,eh)])+5);fig.tight_layout()
for d in OUT:fig.savefig(os.path.join(d,"breach_vs_N.png"),dpi=150)
plt.close(fig)
print("by-universe:",list(zip(labels,[f'{r:.1f}%' for r in rate])))
print("breach-vs-N:",list(zip(xs,[f'{y:.1f}%' for y in ys])))
