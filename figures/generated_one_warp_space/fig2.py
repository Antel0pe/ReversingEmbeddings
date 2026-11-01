import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from warp2 import render_at, make_basis, ROOT_P
SC = str(_HERE)
N=112; SCL=N/28.0; B=make_basis(N)
def warp(src,th):
    x0=(14+14*np.einsum('k,kij->ij',th[:6],B))*SCL-0.5
    y0=(14+14*np.einsum('k,kij->ij',th[6:],B))*SCL-0.5
    return map_coordinates(src,[y0,x0],order=1,mode="constant",cval=0.0)
d=np.load(f"{SC}/w4.npz"); TH,TX,P=d["TH"],d["TX"],d["P"]
root=render_at(ROOT_P[None],N)[0]; I=render_at(P,N)
j=int(np.argmax(P[:,4]))

# ---- the degenerate direction among the three vertical coefficients ----
idx=[6,8,11]; th0=TX[j].copy(); base=warp(root,th0); eps=1e-4
J=[]
for i in idx:
    tp=th0.copy(); tp[i]+=eps; tm=th0.copy(); tm[i]-=eps
    J.append(((warp(root,tp)-warp(root,tm))/(2*eps)).ravel())
J=np.stack(J,1)
Vt=np.linalg.svd(J,full_matrices=False)[2]; sv=np.linalg.svd(J,compute_uv=False)
step=0.5/14.0                                  # coefficient change worth a 0.5 px slide
dg=np.zeros(12); dg[idx]=Vt[-1]*step           # least-observable combination
dp=np.zeros(12); dp[idx]=Vt[0]*step            # most-observable combination, SAME size
img_dg=warp(root,th0+dg); img_dp=warp(root,th0+dp)
e_dg=np.abs(img_dg-base).max(); e_dp=np.abs(img_dp-base).max()
r_dg=np.linalg.norm(img_dg-base)/np.linalg.norm(base)
r_dp=np.linalg.norm(img_dp-base)/np.linalg.norm(base)

# ---- moments method on the same images, same grid ----
dd=28.0/N; c=(np.arange(N)+0.5)*dd
X=np.broadcast_to(c,(N,N)); Y=np.broadcast_to(c[:,None],(N,N))
m00=I.sum((1,2)); mx=(I*X).sum((1,2))/m00; my=(I*Y).sum((1,2))/m00
vxx=(I*X*X).sum((1,2))/m00-mx*mx-dd*dd/12; vyy=(I*Y*Y).sum((1,2))/m00-my*my-dd*dd/12
vxy=(I*X*Y).sum((1,2))/m00-mx*my
hh=np.sqrt(12*vyy); ss=-vxy/vyy; ww=np.sqrt(np.clip(12*(vxx-ss*ss*vyy),0,None))
Qm=np.stack([mx,my,hh,ww,np.degrees(np.arctan(ss))],1)
def inv(T):
    rx,rxs,x1,ry,y1=T[:,1],T[:,2],T[:,0],T[:,8],T[:,6]
    s=rxs/rx; cyn=-y1/ry; cxn=-x1/rx-s*cyn
    return np.stack([14+14*cxn,14+14*cyn,19.75/ry,3.2/rx,np.degrees(np.arctan(s))],1)
Qw=inv(TH)
ew=np.median(np.abs(Qw-P),0); em=np.median(np.abs(Qm-P),0)

fig=plt.figure(figsize=(17.0,9.4))
gs=GridSpec(2,5,figure=fig,height_ratios=[1.0,1.30],hspace=0.30,wspace=0.40,
            left=0.055,right=0.975,top=0.845,bottom=0.075)
fig.suptitle("Why fitting warps from pixels still fails on ONE knob:\n"
             "the three coefficients controlling its vertical map are not equally visible\n"
             "(generated-1 family, 112x112 render, same 120 settings as the previous figure)",
             fontsize=15,fontweight="bold",y=0.975)
vd=max(e_dg,e_dp)
items=[(base,f"F1. BASELINE\nroot warped to target\n({P[j,0]:.1f}, {P[j,1]:.1f}, {P[j,2]:.1f}, {P[j,3]:.1f}, {P[j,4]:.0f}°)","g",None),
       (img_dg,"F2. change the 3 vertical warp\ncoefficients along their\nLEAST-visible combination","g",None),
       (img_dg-base,f"F3. F2 minus F1\nlargest pixel change {e_dg:.4f}\nwhole-image change {r_dg*100:.2f}%","d",None),
       (img_dp,"F4. SAME-SIZE change of the same\n3 coefficients, along their\nMOST-visible combination","g",None),
       (img_dp-base,f"F5. F4 minus F1\nlargest pixel change {e_dp:.4f}\nwhole-image change {r_dp*100:.2f}%","d",None)]
for ci,(img,ttl,kind,_) in enumerate(items):
    ax=fig.add_subplot(gs[0,ci])
    if kind=="g": im=ax.imshow(img,cmap="gray_r",vmin=0,vmax=1)
    else: im=ax.imshow(img,cmap="RdBu_r",vmin=-vd,vmax=vd)
    ax.set_title(ttl,fontsize=9.6); ax.set_xticks([]); ax.set_yticks([])
    if kind=="d":
        plt.colorbar(im,ax=ax,fraction=0.046)
        if ci==2:
            iax=ax.inset_axes([0.55,0.24,0.42,0.42])
            iax.imshow(img,cmap="RdBu_r",vmin=-e_dg,vmax=e_dg)
            iax.set_xticks([]); iax.set_yticks([])
            for sp in iax.spines.values(): sp.set_edgecolor("#b03030"); sp.set_lw(1.4)
            iax.set_xlabel("same data,\nown scale (x87)",fontsize=7.2,color="#b03030",labelpad=2)
            ax.text(0.03,0.96,"blank at the shared scale\nbecause it really is blank",
                    transform=ax.transAxes,fontsize=7.8,color="#b03030",va="top")
        ax.set_xlabel("red = more ink, blue = less\nF3 and F5 share this scale",fontsize=8.3)
    elif ci==0: ax.set_xlabel("dark = coverage 1, white = 0",fontsize=8.3)
    else: ax.set_xlabel("step length identical in F2 and F4",fontsize=8.3)
ax=fig.add_subplot(gs[1,0:2])
ax.scatter(TH[:,11],TH[:,6]-TX[:,6],s=26,color="#1f77b4",alpha=0.8,edgecolor="none")
r=np.corrcoef(TH[:,11],TH[:,6]-TX[:,6])[0,1]
ax.set_xlabel("spurious y←y² coefficient the fit invented\n(should be exactly 0)",fontsize=10)
ax.set_ylabel("error the fit made in y←1\n(the vertical-shift coefficient)",fontsize=10)
ax.set_title(f"G. Each dot is one of 120 fitted images.\n"
             f"The two mistakes cancel exactly: correlation {r:+.3f}",fontsize=11.5)
ax.grid(alpha=0.3); ax.axhline(0,color="0.6",lw=0.8); ax.axvline(0,color="0.6",lw=0.8)
ax.text(0.03,0.04,"A straight line here means the fit never\n"
        "chose between 'slide the bar' and 'curve the\n"
        "vertical map'. Both give the same picture,\n"
        "so the data cannot separate them.",
        transform=ax.transAxes,fontsize=9,va="bottom",
        bbox=dict(fc="white",ec="0.7",alpha=0.92))
ax=fig.add_subplot(gs[1,2:5])
kn=["cx","cy","height","width","lean"]; xp=np.arange(5); bw=0.36
ax.bar(xp-bw/2,ew,bw,label="warp fitted from pixels, then inverted",color="#1f77b4")
ax.bar(xp+bw/2,em,bw,label="6 linear moments of the image, closed form",color="#2a9d4a")
ax.set_yscale("log"); ax.set_xticks(xp); ax.set_xticklabels(kn,fontsize=10.5)
ax.set_ylabel("median error recovering the true knob\n(pixels for cx/cy/height/width, degrees for lean)",fontsize=9.6)
ax.set_title("H. Both methods see only pixels. Same 120 images, same 112x112 grid.\n"
             "Lower is better; note the log scale",fontsize=11.5)
for x,v in zip(xp-bw/2,ew): ax.text(x,v*1.35,f"{v:.3g}",ha="center",fontsize=8.6,color="#1f77b4")
for x,v in zip(xp+bw/2,em): ax.text(x,v*1.35,f"{v:.3g}",ha="center",fontsize=8.6,color="#2a9d4a")
ax.legend(fontsize=9.5,loc="upper right"); ax.grid(alpha=0.3,axis="y",which="both")
ax.set_ylim(min(em.min(),ew.min())*0.25, max(ew.max(),em.max())*300)
ax.annotate("cy is read off the two vertical\ncoefficients alone - the ones F2/F3\nshow the image barely constrains",
            xy=(1+0.02,ew[1]*0.9),xytext=(-0.45,ew[1]*17),fontsize=9.2,color="#b03030",
            arrowprops=dict(arrowstyle="->",color="#b03030",lw=1.4))
fig.savefig(f"{SC}/warp_identifiability.png",dpi=135,facecolor="white")
print("fig2 saved")
print("J singular values (3 vertical coeffs):", np.array2string(sv,precision=4))
print(f"degenerate change: max px {e_dg:.5f} rel {r_dg:.5f} | pure cy: max px {e_dp:.5f} rel {r_dp:.5f}")
print("warp median err:", np.array2string(ew,precision=4))
print("moment median err:", np.array2string(em,precision=5))
