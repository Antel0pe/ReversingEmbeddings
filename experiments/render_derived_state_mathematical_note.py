"""Render the mathematical result as a readable PDF, without source notation."""
from pathlib import Path
import json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

out=Path('figures/derived_state_coordinates')
r=json.loads((out/'results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavuserif','pdf.fonttype':42})

def page(title,sub,num):
 f=plt.figure(figsize=(9,11.5),facecolor='white')
 f.text(.075,.952,title,fontsize=20,weight='bold',va='top')
 f.text(.075,.91,sub,fontsize=10.5,color='#4b5563',va='top')
 f.text(.075,.035,'Generated-one family · exact coverage mathematics · 17 October 2026',fontsize=8,color='#64748b')
 f.text(.93,.035,str(num),ha='right',fontsize=8,color='#64748b')
 return f

def para(f,y,text,size=11,width=94):
 f.text(.075,y,'\n'.join(textwrap.wrap(text,width)),fontsize=size,va='top',linespacing=1.5)

def eq(f,y,text,size=18):
 f.text(.10,y,'$'+text+'$',fontsize=size,va='top',color='#133f5e')

def save(pdf,f,n):
 pdf.savefig(f)
 f.savefig(f'/tmp/derived_state_math_page_{n}.png',dpi=110)
 plt.close(f)

with PdfPages(out/'mathematical_functions.pdf') as pdf:
 f=page('1. Vertical center is a state function','Only the 784 current pixel coverages enter these equations.',1)
 para(f,.857,'Let I denote a valid unrounded generated image. Each entry is the ink coverage of row r, column c. Rows and columns run from 0 to 27; rows increase downward. All lengths use canvas pixels.')
 para(f,.765,'Three fixed linear measurements are sufficient: ink in row 10, the upper six rows, and the lower five rows. None is a supplied geometric parameter.')
 eq(f,.689,r'W(I)=\sum_{c=0}^{27} I_{10,c}',19)
 eq(f,.613,r'U(I)=\sum_{r=0}^{5}\,\sum_{c=0}^{27} I_{rc},\qquad L(I)=\sum_{r=23}^{27}\,\sum_{c=0}^{27} I_{rc}',17)
 eq(f,.516,r'c_y(I)=14.5+\frac{L(I)-U(I)}{2W(I)}',24)
 para(f,.425,'Why it works globally: throughout the allowed family, the top t lies between 3.75 and 5.5, and the bottom b between 23.5 and 25.25. Every horizontal slice has width w and stays inside the canvas. Hence:')
 eq(f,.344,r'W=w,\qquad U=w(6-t),\qquad L=w(b-23)',17)
 para(f,.283,'Their midpoint gives the center equation. The constant 14.5 is (6 + 23)/2, from the fixed mask boundaries; it is not an assumed current center. Other exact state functions follow:')
 eq(f,.202,r't=6-\frac{U}{W},\quad b=23+\frac{L}{W},\quad h=17+\frac{U+L}{W},\quad w=W',18)
 para(f,.118,'This holds for all five default ranges, including tilted strokes and cap/subrow boundary settings. A fixed three-row matrix computes W, U, L; a ratio then returns the center. No fitted coefficients or small-neighborhood approximation appear.',size=10.5)
 save(pdf,f,1)

 f=page('2. Recovering all five physical settings','Two exact row-center functions supply lean and horizontal position.',2)
 para(f,.855,'Use full interior rows r = 10 and r = 18. Define cumulative ink to the left of an integer column boundary k. Choose κᵣ as the smallest boundary minimizing its distance from half the row ink.')
 eq(f,.773,r'C_r(k;I)=\sum_{c=0}^{k-1} I_{rc},\qquad C_r(0;I)=0',18)
 eq(f,.695,r'\kappa_r=\min\,\underset{k\in\{0,\ldots,28\}}{\mathrm{argmin}}\left|C_r(k;I)-\frac{W(I)}{2}\right|',17)
 eq(f,.605,r'\mu_r(I)=\kappa_r-C_r(\kappa_r;I)+\frac{W(I)}{2}',20)
 para(f,.519,'μᵣ is the exact centerline position at height r + ½. It is not a weighted average of pixel-center positions. Lean slope s and centerline intercept a are:')
 eq(f,.446,r's=\frac{\mu_{10}-\mu_{18}}{8},\qquad a=\mu_{10}+10.5s',20)
 eq(f,.358,r'c_x=a-sc_y,\qquad \theta=\frac{180}{\pi}\arctan s',20)
 para(f,.274,'The angle θ is in degrees. Positive lean places the top farther right. Thus W, U, L, μ₁₀, μ₁₈ are five state-derived scalars sufficient to recover the original five physical settings.')
 para(f,.18,'Why the boundary choice is exact: the sixteen horizontal subrow intervals share an interior interval whose half-width is at least 0.5717 pixels. A nearest integer is within 0.5 pixels and is inside every interval. Monotonicity of cumulative ink excludes all outside boundaries from minimizing. Inside the intersection, cumulative ink is boundary position minus the mean left edge. Ties give the same μᵣ.',size=10.3)
 save(pdf,f,2)

 f=page('3. One global mathematical image function','Five derived coordinates simplify the coverage equation.',3)
 para(f,.855,'Choose q = (t, b, a, w, s): top, bottom, centerline intercept, width, and slope. Every coordinate is obtained from the image by the previous equations. Denote this state map by E(I).')
 eq(f,.774,r'E(I)=(t(I),b(I),a(I),w(I),s(I))',18)
 para(f,.715,'Let ℓ be the length of intersection of two intervals. For subrow j in pixel row r, its lower edge, upper edge, and midpoint are:')
 eq(f,.645,r'\ell([u,v],[z,d])=\max(0,\min(v,d)-\max(u,z))',16)
 eq(f,.578,r'y^-_{rj}=r+\frac{j}{16},\quad y^+_{rj}=r+\frac{j+1}{16},\quad y_{rj}=r+\frac{j+1/2}{16}',16)
 para(f,.497,'For every pixel (r, c), the explicit global coverage function is:')
 eq(f,.451,r'F_{rc}(t,b,a,w,s)=\sum_{j=0}^{15}\ell([y^-_{rj},y^+_{rj}],[t,b])',17)
 eq(f,.385,r'\quad\cdot\ell\left([c,c+1],\left[a-sy_{rj}-\frac{w}{2},a-sy_{rj}+\frac{w}{2}\right]\right)',17)
 para(f,.303,'Each term multiplies vertical covered length by horizontal covered length. There is no extra factor of sixteen: subrow height is already in the vertical overlap. The same expression handles all pixel-boundary crossings.')
 para(f,.217,'The allowed coordinate domain Q is determined by the original ranges:')
 eq(f,.172,r'14\leq a-s(t+b)/2\leq15,\quad 14\leq(t+b)/2\leq15',15)
 eq(f,.125,r'19\leq b-t\leq20.5,\quad 1.8\leq w\leq4.6,\quad \tan(-10^\circ)\leq s\leq\tan35^\circ',14)
 eq(f,.078,r'\mathcal{M}=\{F(q):q\in Q\},\qquad E(F(q))=q,\quad F(E(I))=I',15)
 save(pdf,f,3)

 f=page('4. Exact finite navigation from the image','Five requested physical increments are the only other variable inputs.',4)
 para(f,.855,'Let δ = (δₓ, δᵧ, δₕ, δw, δθ) denote desired changes in horizontal center, vertical center, height, width, and lean. The first four are in pixels; the angle increment is in degrees.')
 eq(f,.766,r'q=E(I)=(t,b,a,w,s),\qquad c=\frac{t+b}{2}',18)
 eq(f,.678,r's^{\prime}=\tan\left(\arctan s+\frac{\pi\delta_\theta}{180}\right)',21)
 para(f,.588,'Define the five entries of the transformed coordinate tuple Aδ(q) by:')
 eq(f,.538,r't^{\prime}=t+\delta_y-\frac{\delta_h}{2},\qquad b^{\prime}=b+\delta_y+\frac{\delta_h}{2}',19)
 eq(f,.45,r'a^{\prime}=a+\delta_x+s^{\prime}(c+\delta_y)-sc,\qquad w^{\prime}=w+\delta_w',18)
 eq(f,.36,r'T_\delta(I)=F\left(A_\delta(E(I))\right)',24)
 para(f,.267,'This is one exact image-state function. A separate function for each original knob is obtained by setting the other four increments to zero. Every center, edge, slope and width used inside it is derived from the current image. It applies to any finite increment whose endpoint remains allowed.')
 eq(f,.169,r'T_0(I)=I,\qquad T_{-\delta}(T_\delta(I))=I',18)
 eq(f,.098,r'T_\eta(T_\delta(I))=T_{\delta+\eta}(I)',20)
 save(pdf,f,4)

 f=page('5. Knob identity, validation, and exactness','A valid chart need not have the same coordinate directions as the original knobs.',5)
 para(f,.858,'For example, changing top t alone by ε changes physical center by ε/2, height by −ε, and horizontal center by −sε/2. It is a mixed coordinate direction. A pure original vertical translation instead moves (t, b, a) by (ε, ε, sε).')
 para(f,.749,'Let R(I) denote the five original settings recovered on pages 1–2. The finite test for alignment with the original physical knobs is:')
 eq(f,.683,r'R(T_\delta(I))=R(I)+\delta',23)
 para(f,.604,'The equations satisfy this identity algebraically. Five independent coordinates alone would not establish it. At smooth points, the recovered physical response matrix is the identity.')
 para(f,.507,'Numerical checks against the existing generator:',size=12)
 checks=[f"• {r['states']:,} states: random, lattice, near faces, cap/side boundaries, and ties.",
         '• Worst recovered vertical-center error: 1.78 × 10⁻¹⁵ pixels.',
         '• Worst recovered lean error: 2.84 × 10⁻¹⁴ degrees.',
         '• 10,240 single-knob updates; largest pixel error: 1.18 × 10⁻¹⁴.',
         '• 2,048 joint updates; largest pixel error: 9.99 × 10⁻¹⁵.',
         '• Four five-step orders at 128 pairs; largest pixel error: 1.80 × 10⁻¹⁴.']
 f.text(.075,.471,'\n'.join(checks),fontsize=10.4,va='top',linespacing=1.65)
 para(f,.298,'The full-domain identities follow from the geometry, not from sample accuracy. These are explicit reverse-engineered equations for the known family, not a claim of discovering uniquely simplest coordinates from arbitrary data.',size=10.5)
 para(f,.207,'Storage limit: float32 images for centers 14 and 14.000000001 can be bitwise identical. After the same vertical increment of ⅛ pixel, their stored images differ. Consequently the rounded bitmap alone cannot identify every original real state or reproduce every physical action exactly.',size=10.5)
 para(f,.105,'These equations are exact on unrounded coverage. Exact original physical actions from the rounded bitmap alone are impossible; retaining the precise physical state separately avoids the loss.',size=10.2)
 save(pdf,f,5)
print('Rendered five-page mathematical-functions PDF.')
