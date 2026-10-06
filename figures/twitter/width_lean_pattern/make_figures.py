"""Pixel explanation and clean line graphs of width-dependent lean responses.

Run from the repository root:
.venv/bin/python figures/twitter/width_lean_pattern/make_figures.py
"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES
OUT=Path(__file__).resolve().parent
BG='#101923';PANEL='#192735';FG='#f4f7fa';MUTED='#b7c4d1';BLUE='#63c7f4';RED='#ff5252';GREEN='#b6e456'
INK=LinearSegmentedColormap.from_list('ink',[PANEL,FG])
SIGNED=LinearSegmentedColormap.from_list('change',[RED,BG,BLUE])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'text.color':FG,
 'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
 'axes.edgecolor':MUTED,'axes.facecolor':BG,'savefig.facecolor':BG})

def base(n,title,subtitle):
 f=plt.figure(figsize=(15,10),facecolor=BG)
 f.text(.045,.953,f'{n:02d}  /  WIDTH × LEAN',fontsize=14,color=BLUE,weight='bold')
 f.text(.045,.892,title,fontsize=31,weight='bold')
 f.text(.045,.842,subtitle,fontsize=17,color=MUTED)
 return f

def save(f,name):
 f.canvas.draw();rr=f.canvas.get_renderer()
 texts=list(f.texts)+[t for a in f.axes for t in a.texts]
 pairs=[]
 for i,a in enumerate(texts):
  for b in texts[i+1:]:
   if a.get_text() and b.get_text() and a.get_window_extent(rr).overlaps(b.get_window_extent(rr)):
    pairs.append([a.get_text(),b.get_text()])
 f.savefig(OUT/name,dpi=120);plt.close(f)
 return pairs

# The same finite lean move at two starting widths.
p=np.tile(RANGES.mean(1),(4,1));p[:,3]=np.repeat([1.8,10.0],2);p[:,4]=np.tile([0.0,20.0],2)
a=render(p).astype(float).reshape(2,2,28,28);changes=a[:,1]-a[:,0]
v=changes.reshape(2,-1)
angle=float(np.degrees(np.arccos(np.clip(v[0]@v[1]/np.linalg.norm(v[0])/np.linalg.norm(v[1]),-1,1))))
f=base(1,'Same tilt. Different pixels.','Thin vs comically wide. Both strokes tilt by exactly 20°.')
for x,title in zip([.28,.55,.82],['Before · upright','After · 20°','Changed pixels']):
 f.text(x,.785,title,ha='center',fontsize=20,weight='bold')
for row,y in enumerate([.485,.155]):
 color=FG if row==0 else GREEN
 f.text(.045,y+.17,f'Width\n{p[row*2,3]:.1f} px',fontsize=23,color=color,weight='bold',linespacing=1.5)
 if row:f.text(.045,y+.105,'5.6× wider',fontsize=17,color=color)
 for col,x in enumerate([.18,.45,.72]):
  ax=f.add_axes([x,y,.20,.265]);img=a[row,col] if col<2 else changes[row]
  ax.imshow(img,cmap=INK if col<2 else SIGNED,vmin=0 if col<2 else -1,vmax=1,interpolation='nearest')
  ax.set_axis_off()
  if col==2:
   cap='Reference tilt response' if row==0 else f'Response directions: {angle:.1f}° apart'
   ax.text(.5,-.07,cap,transform=ax.transAxes,ha='center',va='top',fontsize=16,color=MUTED)
 f.text(.408,y+.13,'→',fontsize=27,ha='center',color=MUTED)
# Large color swatches make the signed pixel-change legend readable on phones.
from matplotlib.patches import Rectangle
for x,color,label in [( .045,RED,'Losing ink'),(.28,BLUE,'Gaining ink'),(.53,BG,'No change')]:
 f.add_artist(Rectangle((x,.078),.014,.021,transform=f.transFigure,facecolor=color,
  edgecolor=MUTED if color==BG else color,linewidth=1))
 f.text(x+.024,.079,label,fontsize=18,color=color if color!=BG else MUTED,weight='bold')
f.text(.745,.081,'Scale: −1 to +1 coverage',fontsize=13,color=MUTED)
f.text(.045,.047,'Strokes: dark = no ink, white = full ink. Direction angle uses all 784 pixel changes.',fontsize=14,color=MUTED)
f.text(.045,.017,'Exaggerated demo: width 10 px exceeds the default range. Line graph uses widths 3.2–4.2 px.',fontsize=13,color=MUTED)
checks={'01_pixels.png':save(f,'01_pixels.png')}

# Recompute the complete original distance sweep using the actual renderer.
lean=np.linspace(-10,35,901);offsets=np.arange(1,11)/10
widths=3.2+offsets

def images(width,angles):
 q=np.tile(RANGES.mean(1),(len(np.atleast_1d(angles)),1));q[:,3]=width;q[:,4]=angles
 return render(q).astype(float).reshape(len(q),784)
normal=images(3.2,lean);normal0=images(3.2,[0])[0]
base_move=normal-normal0
all_images=np.array([images(w,lean) for w in widths]);uprights=np.array([images(w,[0])[0] for w in widths])
from_original=np.linalg.norm(all_images-normal0,axis=2)
aligned=np.linalg.norm((all_images-uprights[:,None,:])-base_move[None,:,:],axis=2)
same=np.linalg.norm(all_images-normal[None,:,:],axis=2)
template=np.sum(offsets[:,None]*aligned,axis=0)/np.sum(offsets**2)
relative=float(np.linalg.norm(aligned-offsets[:,None]*template)/np.linalg.norm(aligned))
# Fresh intermediate widths and half-grid lean values test the shared shape.
fresh_offsets=np.array([.15,.35,.55,.75,.95]);fresh_lean=(lean[:-1]+lean[1:])/2
fresh_normal=images(3.2,fresh_lean);fresh=[]
for off in fresh_offsets:
 x=images(3.2+off,fresh_lean);x0=images(3.2+off,[0])[0]
 fresh.append(np.linalg.norm((x-x0)-(fresh_normal-normal0),axis=1))
fresh=np.array(fresh);pred=fresh_offsets[:,None]*np.interp(fresh_lean,lean,template)
fresh_error=float(np.linalg.norm(fresh-pred)/np.linalg.norm(fresh))
old=json.loads((ROOT/'figures/generated_lean_width_comparison/width_sweep/pattern_analysis.json').read_text())
assert abs(relative-old['metrics']['aligned_mismatch']['relative_frobenius_error'])<1e-7
assert abs(fresh_error-old['metrics']['aligned_mismatch']['held_out_width_and_lean_relative_error'])<1e-7
assert np.max(aligned[:,np.argmin(abs(lean))])==0

# Local lean directions distinguish turning from accumulated displacement.
theta=np.linspace(-9.5,34.5,881)
base_direction=images(3.2,theta+.05)-images(3.2,theta-.05)
angles=[]
for w in widths:
 direction=images(w,theta+.05)-images(w,theta-.05)
 cosine=np.sum(direction*base_direction,axis=1)/(np.linalg.norm(direction,axis=1)*np.linalg.norm(base_direction,axis=1))
 angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
angles=np.array(angles)
peak=int(np.argmax(angles[-1]));peak_angle=float(angles[-1,peak]);peak_lean=float(theta[peak])
assert abs(peak_angle-80.56677078225286)<1e-4
colors=[plt.cm.viridis(v) for v in np.linspace(.25,.92,10)]
# An optional flag keeps the final choice reversible without changing numerical data.
with_angle='--distance-only' not in sys.argv
f=base(2,'Wider strokes follow a pattern',
 'Width increases +0.1 to +1.0 px from normal width 3.2. Center and height stay fixed.')
if with_angle:
 specs=[([.09,.305,.365,.415],False),([.60,.305,.355,.415],True)]
else:
 specs=[([.105,.305,.84,.415],False)]
for rect,is_angle in specs:
 ax=f.add_axes(rect)
 for row,c in zip(angles if is_angle else aligned,colors):
  ax.plot(theta if is_angle else lean,row,color=c,lw=2)
 ax.axvline(0,color=MUTED,alpha=.45,ls=':',lw=1)
 ax.set(xlim=(-10,35),ylim=(0,90 if is_angle else 4.2),xticks=[-10,0,10,20,30,35],
  yticks=[0,30,60,90] if is_angle else [0,1,2,3,4],xlabel='Lean (degrees) · 0 = upright',
  ylabel='Direction difference (degrees)' if is_angle else 'Response mismatch (pixel L2)')
 ax.set_title('Direction of a small lean change' if is_angle else 'Difference in lean-change vectors',fontsize=19,pad=19)
 ax.xaxis.label.set_size(15);ax.yaxis.label.set_size(15);ax.tick_params(labelsize=14)
 ax.grid(axis='y',color=MUTED,alpha=.12);ax.spines[['top','right']].set_visible(False)
 ax.plot([-10,35],[0,0],color=FG,ls='--',lw=2)
 if is_angle:
  ax.scatter([peak_lean],[peak_angle],s=35,color=GREEN,zorder=5)
  ax.annotate(f'{peak_angle:.1f}° apart',xy=(peak_lean,peak_angle),xytext=(8,83),fontsize=15,color=GREEN,
   arrowprops={'arrowstyle':'->','color':GREEN})
 else:
  ax.annotate('Zero at upright\nby subtraction',xy=(0,0),xytext=(-7,1.75),fontsize=13,color=MUTED,
   arrowprops={'arrowstyle':'->','color':MUTED})
ca=f.add_axes([.275,.182,.45,.017]);cmap=LinearSegmentedColormap.from_list('widths',colors)
cb=f.colorbar(plt.cm.ScalarMappable(norm=Normalize(.1,1),cmap=cmap),cax=ca,orientation='horizontal',ticks=[.1,.5,1])
cb.ax.set_xticklabels(['+0.1','+0.5','+1.0']);cb.ax.tick_params(labelsize=14)
cb.set_label('Width increase (pixels)',fontsize=14,labelpad=8)
f.text(.045,.080,'Distance: compare [tilted image − its own upright image] with the same vector at width 3.2.',fontsize=13,color=MUTED)
if with_angle:
 f.text(.045,.048,'Angle: compare unit directions of 0.1° centered lean moves at the same lean. 0° = aligned; 90° = perpendicular.',fontsize=13,color=MUTED)
else:
 f.text(.045,.048,'Each colored line fixes width and varies lean. Similar distance shapes do not mean identical pixel directions.',fontsize=13,color=MUTED)
f.text(.045,.017,'Measurements use all 784 pixels. Fixed center (14.5, 14.5), height 19.75 px; sampled width/lean slice.',fontsize=13,color=MUTED)
checks['02_pattern.png']=save(f,'02_pattern.png')
# Store an alternative of the original-upright graph if that is the user's chosen reference.
np.savez_compressed(OUT/'plot_data.npz',lean=lean,width_increases=offsets,from_original_upright=from_original,
 upright_aligned_mismatch=aligned,same_lean_gap=same,shared_shape=template,pixel_example_settings=p,pixel_changes=changes,angle_lean=theta,local_direction_angles=angles)
report={'first_image':{'lean_start':0.0,'lean_end':20.0,'widths':[1.8,10.0],'exaggeration':'10-pixel width is outside the default 1.8 to 4.6 range; graph remains 3.2 to 4.2', 'frame_clipping': bool(np.any(a[:,:,[0,-1],:]) or np.any(a[:,:,:,[0,-1]])),'pixel_change_angle_degrees':angle},
 'second_image':{'layout':'Distance plus angle' if with_angle else 'Distance only','local_angle_peak_degrees':peak_angle,'local_angle_peak_lean':peak_lean,'local_angle_step_degrees':.1,'reference':'Difference of tilt-change vectors, subtracting each width-specific upright image',
 'width_increases':offsets.tolist(),'lean_samples':901,'relative_shape_error':relative,'fresh_error':fresh_error,
 'fresh_offsets':fresh_offsets.tolist(),'fresh_lean_samples':len(fresh_lean)},'text_overlaps':checks,
 'understanding_check':{'object':'Actual coverage images and their signed pixel changes, then norms of differences of full 784D displacement vectors',
 'axes_and_colors':'Lean degrees, pixel-L2 mismatch, and local pixel-direction angles if selected; width colors explicitly keyed',
 'starting_case':'First row width 1.8, upright; same 20-degree tilt at widths 1.8 and 10; line curves separately anchored at each width upright',
 'evidence':'Distinct change maps and similarly shaped scalar distance curves; optional local angle graph shows aligned directions near upright and strong turning elsewhere',
 'scope':'Exaggerated demonstration widths 1.8 and 10, with fixed center and height; graph width 3.2 to 4.2; signed pixels unprojected'}}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
ims=[]
for n in checks:
 im=Image.open(OUT/n).convert('RGB');im.thumbnail((750,500));ims.append(im)
sheet=Image.new('RGB',(750,1000),BG)
for i,im in enumerate(ims):sheet.paste(im,(0,500*i))
sheet.save(OUT/'preview.png')
print(json.dumps(report,indent=2))
