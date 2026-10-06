"""Three shareable figures using the actual generated-one renderer and saved fits.

Run from the repository root: .venv/bin/python figures/twitter/progress_2026_10_05/make_figures.py
"""
from pathlib import Path
import sys, json, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from grey_ones import render, RANGES
OUT = Path(__file__).resolve().parent
BG='#101923'; PANEL='#192735'; WHITE='#f4f7fa'; MUTED='#b7c4d1'
BLUE='#63c7f4'; ORANGE='#ffaf70'; PINK='#ef73c3'; GREEN='#77e2b2'
INK=LinearSegmentedColormap.from_list('ink',[PANEL,WHITE])
SIGNED=LinearSegmentedColormap.from_list('signed',[ORANGE,BG,BLUE])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'text.color':WHITE,
 'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':MUTED,
 'axes.facecolor':BG,'savefig.facecolor':BG})


def base(number,title,subtitle):
 f=plt.figure(figsize=(15,10),facecolor=BG)
 f.text(.045,.952,f'{number:02d}  /  GENERATED 1s',fontsize=14,color=BLUE,weight='bold')
 f.text(.045,.891,title,fontsize=31,weight='bold')
 f.text(.045,.842,subtitle,fontsize=17,color=MUTED)
 return f


def txt(f,x,y,s,size=18,color=WHITE,**kw):
 return f.text(x,y,s,fontsize=size,color=color,**kw)


def image(f,rect,values,label='',caption='',outside=False):
 ax=f.add_axes(rect); arr=np.asarray(values).reshape(28,28)
 ax.imshow(arr,cmap=INK,vmin=0,vmax=1,interpolation='nearest')
 if outside:
  mask=(arr<0)|(arr>1); overlay=np.zeros((28,28,4));overlay[mask]=matplotlib.colors.to_rgba(PINK)
  ax.imshow(overlay,interpolation='nearest')
 ax.set_axis_off()
 if label: ax.text(.5,1.06,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=18,weight='bold')
 if caption: ax.text(.5,-.09,caption,transform=ax.transAxes,ha='center',va='top',fontsize=16,color=MUTED)
 return ax


def save(f,name):
 f.savefig(OUT/name,dpi=120)
 # Detect overlapping text boxes; axes ticks and the image pixels are not text boxes.
 f.canvas.draw(); renderer=f.canvas.get_renderer(); labels=[]
 for t in list(f.texts)+[t for ax in f.axes for t in ax.texts]:
  if t.get_text().strip(): labels.append((t.get_text(),t.get_window_extent(renderer)))
 overlaps=[]
 for i,(a,ba) in enumerate(labels):
  for b,bb in labels[i+1:]:
   if ba.overlaps(bb): overlaps.append([a,b])
 if overlaps: print('TEXT OVERLAPS',name,overlaps)
 plt.close(f)
 return overlaps

checks={}
# 1. Same endpoints, shorter route. Compute path lengths again with actual float32 render.
case=json.loads((ROOT/'claude/lean_shortcut_results.json').read_text())[1]
start=np.array(case['start']); end=np.array(case['end']); depth=case['sine_height_only']['height_depth']
t=np.linspace(0,1,8193); direct=start[None,:]+t[:,None]*(end-start)
detour=direct.copy();detour[:,2]-=depth*np.sin(np.pi*t)
assert (detour>=RANGES[:,0]).all() and (detour<=RANGES[:,1]).all()
a=render(direct).reshape(len(t),-1).astype(float);b=render(detour).reshape(len(t),-1).astype(float)
lengths=[float(np.linalg.norm(np.diff(v,axis=0),axis=1).sum()) for v in [a,b]]
assert np.array_equal(a[[0,-1]],b[[0,-1]])
gain=100*(1-lengths[1]/lengths[0])
assert abs(gain-5.052)<.01
f=base(1,'Shorten, tilt, then grow back','Same start and finish. A temporary height dip reduces total pixel change.')
idx=np.linspace(0,len(t)-1,5,dtype=int)
for row,(ims,label,col,y) in enumerate([(a,'Tilt only',ORANGE,.59),(b,'Height dip',GREEN,.35)]):
 txt(f,.045,y+.065,label,19,col,weight='bold')
 for j,k in enumerate(idx):
  image(f,[.20+j*.151,y,.13,.195],ims[k],caption=f'{direct[k,4]:g}°' if row==0 else '')
ax=f.add_axes([.20,.13,.44,.14])
ax.plot(direct[:,4],direct[:,2],color=ORANGE,lw=3)
ax.plot(detour[:,4],detour[:,2],color=GREEN,lw=3)
ax.set(xlim=(-10,35),ylim=(18.9,20.8),xticks=[-10,12.5,35],yticks=[19,20.5],xlabel='Tilt (degrees)',ylabel='Height (pixels)')
ax.tick_params(labelsize=13); ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
ax.spines[['top','right']].set_visible(False)
txt(f,.71,.242,f'{gain:.2f}% less',31,GREEN,weight='bold')
txt(f,.71,.198,'accumulated pixel change',16,MUTED)
txt(f,.71,.148,f'{lengths[0]:.2f} → {lengths[1]:.2f} pixel-L2 units',16)
txt(f,.045,.045,'100 large-tilt cases: median saving 2.92%. Sampled routes; no shortest-path claim.',14,MUTED)
txt(f,.045,.016,'28×28 ink coverage: dark = 0, white = 1. Center fixed at (14.5, 14.5); width 4.6 px.',12,MUTED)
checks['01_shortcut.png']=save(f,'01_shortcut.png')
# 2. Exact mixed finite difference; show raw violations rather than hiding clipping.
p0=RANGES.mean(1);p1=p0.copy();p1[3]+=1;p2=p0.copy();p2[4]+=15;p3=p1.copy();p3[4]+=15
ims=render([p0,p1,p2,p3]).astype(float); naive=ims[1]+ims[2]-ims[0]
err=float(np.linalg.norm(naive-ims[3]));bad=int(((naive<0)|(naive>1)).sum())
assert abs(err-3.4600997)<1e-6 and bad==26
f=base(2,"Two changes don't simply add",'Widen by 1 pixel. Tilt by 15°. Their combined effect changes with the starting image.')
for j,(v,l,c) in enumerate(zip(ims[:3],['Start','Wider only','More tilted only'],['Width 3.2 px · tilt 12.5°','Width +1 px','Tilt +15°'])):
 image(f,[.105+j*.305,.515,.21,.265],v,l,c)
for j,(v,l,c) in enumerate([(ims[3],'Actual joint change','Both knobs moved'),(naive,'Add separate changes',f'{bad} pixels exceed valid ink'),(ims[3]-naive,'Missing correction',f'Pixel-L2 mismatch: {err:.2f}')]):
 if j<2:
  image(f,[.105+j*.305,.14,.21,.235],v,l,c,outside=j==1)
 else:
  ax=f.add_axes([.105+j*.305,.14,.21,.235]);ax.imshow(v,cmap=SIGNED,vmin=-1,vmax=1,interpolation='nearest');ax.set_axis_off()
  ax.text(.5,1.06,l,transform=ax.transAxes,ha='center',va='bottom',fontsize=18,weight='bold')
  ax.text(.5,-.09,c,transform=ax.transAxes,ha='center',va='top',fontsize=16,color=MUTED)
txt(f,.41,.435,'Wider + tilted − start',17,MUTED,ha='left')
txt(f,.045,.07,'Correction: orange = remove ink; blue = add ink; scale −1 to +1. Pink = coverage outside [0, 1].',13,MUTED)
txt(f,.045,.039,'Full-box five-knob control: 2,048 starts; adding separate changes misses 34.1% of the joint change (median).',12,MUTED)
txt(f,.045,.014,'Synthetic 28×28 coverage: dark = 0, white = 1. Center (14.5, 14.5) and height 19.75 px fixed above.',12,MUTED)
checks['02_interaction.png']=save(f,'02_interaction.png')
# 3. Re-evaluate saved field models on their held-out slice; no refitting.
field=ROOT/'experiments/lean_field_fit_20261004'
spec=importlib.util.spec_from_file_location('lean_field',field/'run.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
metrics=json.loads((field/'results/metrics.json').read_text())
poly=np.load(field/'results/polynomial_degree_5.npz'); edge=np.load(field/'results/piecewise_edge_rule.npz')
pts=np.vstack([module.grid(19,37,.371),np.array([(w,l) for w in RANGES[3] for l in RANGES[4]])])
actual=module.measured_field(pts)
pred=[module.basis(pts,5)[0]@poly['coefficients'],float(edge['fitted_scale'])*module.edge_basis(module.settings_from_slice(pts))]
means=[]
for v,name in zip(pred,['polynomial_degree_5','piecewise_edge_rule']):
 mean=module.field_scores(v,actual)[0]['relative_arrow_error']['mean']*100
 assert abs(mean-metrics['models'][name]['evaluation']['relative_arrow_error']['mean']*100)<1e-9
 means.append(mean)
state=RANGES.mean(1)[None,:]; actual_one=module.measured_field_settings(state)[0]
poly_one=(module.basis(state[:,[3,4]],5)[0]@poly['coefficients'])[0]
edge_one=(float(edge['fitted_scale'])*module.edge_basis(state))[0]
responses=[actual_one,poly_one,edge_one]; lim=np.ceil(max(np.max(abs(v)) for v in responses)*100)/100
f=base(3,'Pixel edges explain the tilt response','Predict which pixels change when the stroke tilts a little.')
image(f,[.07,.505,.20,.27],render(state)[0],'Starting stroke','Tilt 12.5° · width 3.2 px')
for j,(v,l) in enumerate(zip(responses,['Actual response','Polynomial fit','Edge rule'])):
 ax=f.add_axes([.325+j*.221,.505,.19,.27]);ax.imshow(v.reshape(28,28),cmap=SIGNED,vmin=-lim,vmax=lim,interpolation='nearest');ax.set_axis_off()
 ax.text(.5,1.06,l,transform=ax.transAxes,ha='center',va='bottom',fontsize=18,weight='bold')
 error=100*np.linalg.norm(v-actual_one)/np.linalg.norm(actual_one)
 ax.text(.5,-.09,f'{error:.2f}% error here',transform=ax.transAxes,ha='center',va='top',fontsize=16,color=MUTED)
txt(f,.045,.427,f'Response: orange loses ink; blue gains ink; dark = zero. Common scale ±{lim:g} coverage per degree.',14,MUTED)
ax=f.add_axes([.245,.172,.66,.182]);ax.barh([1,0],means,color=[ORANGE,GREEN],height=.42)
ax.set(xlim=(0,66),yticks=[1,0],yticklabels=['Degree-5 polynomial','Geometry + edge rule'],xticks=[0,20,40,60],xlabel='Mean response error (%) · lower is better')
ax.tick_params(labelsize=14);ax.xaxis.label.set_size(15);ax.spines[['top','right','left']].set_visible(False)
for y,val,col in zip([1,0],means,[ORANGE,GREEN]):ax.text(val+1.2,y,f'{val:.2f}%',ha='left',va='center',fontsize=23,color=col,weight='bold')
txt(f,.045,.08,'703 unseen width/tilt settings + 4 corners. Error = mismatch / actual response length, over all 784 pixels.',13,MUTED)
txt(f,.045,.049,'The edge rule uses known geometry + 1 fitted scale. Full five-knob control: 0.76% error on 256 new states.',13,MUTED)
txt(f,.045,.019,'Finite-difference step 0.025°. Center and height fixed above. Stroke coverage: dark = 0, white = 1.',12,MUTED)
checks['03_edge_rule.png']=save(f,'03_edge_rule.png')
# Mobile-size inspection contact sheet. This is a local preview, not an extra post image.
thumbs=[]
for name in checks:
 im=Image.open(OUT/name).convert('RGB');im.thumbnail((720,480));thumbs.append(im)
sheet=Image.new('RGB',(720,480*3),BG)
for i,im in enumerate(thumbs):sheet.paste(im,(0,i*480))
sheet.save(OUT/'preview.png')
summary={'interval':'2026-09-28 through 2026-10-05','renderer':'grey_ones.render',
 'path_steps':8192,'path_lengths_float32':lengths,'height_dip_px':depth,'saving_percent':gain,
 'joint_L2_mismatch':err,'joint_out_of_range_pixels':bad,
 'heldout_response_mean_error_percent':dict(zip(['degree_5','edge_rule'],means)),
 'text_overlaps':checks,'images':list(checks),'figure_pixels':[1800,1200]}
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
