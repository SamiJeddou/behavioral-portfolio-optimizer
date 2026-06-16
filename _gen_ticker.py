import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
closes=[100.0]; p=100.0
def step(drift, vol, k):
    global p
    for _ in range(k):
        p = p + drift + rng.normal(0, vol)
        closes.append(p)
# Off-centre early high, then a long jagged DESCENT to a much lower end (net down-trend):
step(2.6, 1.1, 6)    # steep early rally -> high near the LEFT (~bar 6)
step(-1.7,1.7, 6)    # first leg down
step(1.1, 1.4, 4)    # weak bounce -> a LOWER high (~bar 16)
step(-2.3,2.0, 12)   # long, choppy decline to a low end (well below the start)
c = np.array(closes[1:]); n=len(c)
o = np.empty(n); o[0]=100.0; o[1:]=c[:-1]
o = o + rng.normal(0,0.3,n)
hi = np.maximum(o,c) + np.abs(rng.normal(0,1.1,n))
lo = np.minimum(o,c) - np.abs(rng.normal(0,1.1,n))
mv = np.abs(c-np.concatenate([[c[0]],c[:-1]]))
vol = (np.abs(rng.normal(1,0.4,n)) + 0.9*mv); vol/=vol.max()
print('candles', n, 'start %.0f end %.0f peak@%d'%(c[0], c[-1], int(np.argmax(hi))))

UP='#2ea043'; DN='#e34a4a'; BGF='#0d1117'; PANEL='#0e1521'; BORD='#2a3340'; W=0.66
fig=plt.figure(figsize=(9.88,4.42), dpi=100); fig.patch.set_facecolor(BGF)
gs=fig.add_gridspec(2,1,height_ratios=[3.05,1],hspace=0.16,left=0.038,right=0.985,top=0.875,bottom=0.05)
axp=fig.add_subplot(gs[0]); axv=fig.add_subplot(gs[1])
for ax in (axp,axv):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color(BORD); s.set_linewidth(1)
    ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(length=0)
for i in range(n):
    col=UP if c[i]>=o[i] else DN
    axp.plot([i,i],[lo[i],hi[i]],color=col,lw=1.5,solid_capstyle='round',zorder=2)
    b=max(abs(c[i]-o[i]),0.3)
    axp.add_patch(plt.Rectangle((i-W/2, min(o[i],c[i])), W, b, facecolor=col, edgecolor=col, lw=0.6, zorder=3))
axp.set_xlim(-1,n); axp.set_ylim(lo.min()-2.5, hi.max()+2.5)
for i in range(n):
    col=UP if c[i]>=o[i] else DN
    axv.add_patch(plt.Rectangle((i-W/2,0),W,vol[i],facecolor=col,edgecolor='none',alpha=0.92))
axv.set_xlim(-1,n); axv.set_ylim(0, vol.max()*1.15)
fig.text(0.042,0.93,'EQUITY',color='#E3C77E',fontsize=15,fontweight='bold')
fig.text(0.135,0.932,'· daily OHLC',color='#8b97a8',fontsize=11)
fig.text(0.985,0.932,'Candles & volume',color='#8b97a8',fontsize=11,style='italic',ha='right')
axv.text(0.012,0.80,'Vol',transform=axv.transAxes,color='#8b97a8',fontsize=9,va='top')
fig.savefig('home_ticker3.png',facecolor=BGF)
print('saved')
