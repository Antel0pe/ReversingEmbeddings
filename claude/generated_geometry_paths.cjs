module.exports=(function(){
// Generated-one geometry diagnostics. No dependencies.
// Renderer translated from grey_ones.py; calculations use float64.
const ranges=[[14,15],[14,15],[19,20.5],[1.8,4.6],[-10,35]];
function render(q,sub=16,f32=false) {
 const [cx,cy,h,w,angle]=q, slope=Math.tan(angle*Math.PI/180);
 const im=new Float64Array(784),top=cy-h/2,bot=cy+h/2;
 for(let k=0;k<28*sub;k++){
  const yl=k/sub,ym=(k+.5)/sub,wy=Math.max(0,Math.min(yl+1/sub,bot)-Math.max(yl,top));
  if(wy===0)continue;
  const mid=cx-slope*(ym-cy),L=mid-w/2,R=mid+w/2,row=Math.floor(k/sub);
  for(let c=Math.max(0,Math.floor(L));c<Math.min(28,Math.ceil(R));c++){
   const ov=Math.max(0,Math.min(R,c+1)-Math.max(L,c));
   im[row*28+c]+=ov*wy;
  }
 }
 if(f32)for(let k=0;k<784;k++)im[k]=Math.fround(im[k]);
 return im;
}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function norm(a){return Math.sqrt(dot(a,a));}
function sub(a,b){return Float64Array.from(a,(x,i)=>x-b[i]);}
function add(a,b,t=1){return Float64Array.from(a,(x,i)=>x+t*b[i]);}
function scale(a,s){return Float64Array.from(a,x=>s*x);}
function dist(a,b){return norm(sub(a,b));}
function maxabs(a){let m=0;for(const x of a)m=Math.max(m,Math.abs(x));return m;}
function basis(vectors,tol=1e-10) {
 const Q=[];for(const v of vectors){let r=Float64Array.from(v);for(let pass=0;pass<2;pass++)for(const u of Q)r=add(r,u,-dot(r,u));const n=norm(r);if(n>tol)Q.push(scale(r,1/n));}return Q;
}
function perp(v,Q){let r=Float64Array.from(v);for(const u of Q)r=add(r,u,-dot(r,u));return r;}
function eigSym(M) {
 const A=M.map(x=>x.slice()),n=A.length,V=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
 for(let iter=0;iter<100*n*n;iter++){
  let p=0,q=1,m=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(A[i][j])>m){m=Math.abs(A[i][j]);p=i;q=j;}
  if(m<1e-13)break;
  const theta=.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(theta),s=Math.sin(theta);
  const app=A[p][p],aqq=A[q][q],apq=A[p][q];
  for(let k=0;k<n;k++)if(k!==p&&k!==q){const a=A[k][p],b=A[k][q];A[k][p]=A[p][k]=c*a-s*b;A[k][q]=A[q][k]=s*a+c*b;}
  A[p][p]=c*c*app-2*s*c*apq+s*s*aqq;A[q][q]=s*s*app+2*s*c*apq+c*c*aqq;A[p][q]=A[q][p]=0;
  for(let k=0;k<n;k++){const a=V[k][p],b=V[k][q];V[k][p]=c*a-s*b;V[k][q]=s*a+c*b;}
 }
 return Array.from({length:n},(_,i)=>({value:A[i][i],vector:V.map(row=>row[i])})).sort((a,b)=>b.value-a.value);
}
function singular(vs){return eigSym(vs.map(a=>vs.map(b=>dot(a,b)))).map(x=>Math.sqrt(Math.max(0,x.value)));}
function jac(q,eps=1e-5,subrows=16,f32=false){
 return ranges.map((r,j)=>{const d=eps*(r[1]-r[0]),a=q.slice(),b=q.slice();a[j]-=d;b[j]+=d;return scale(sub(render(b,subrows,f32),render(a,subrows,f32)),1/(2*eps));});
}
function rng(seed=20260930){let x=seed>>>0;return ()=>{x=(Math.imul(1664525,x)+1013904223)>>>0;return x/4294967296;};}
function quantiles(xs){const a=xs.slice().sort((x,y)=>x-y);return {min:a[0],p10:a[Math.floor((a.length-1)*.1)],median:a[Math.floor((a.length-1)*.5)],p90:a[Math.floor((a.length-1)*.9)],max:a.at(-1)};}
function breaks(q,subrows=16) {
 const [cx,cy,h,w,angle]=q,slope=Math.tan(angle*Math.PI/180),wb=[1.8,4.6],hb=[19,20.5];
 for(let k=0;k<28*subrows;k++){
  const yl=k/subrows,ym=(k+.5)/subrows;
  for(const v of [2*(cy-yl),2*(yl-cy)])if(v>19&&v<20.5)hb.push(v);
  if(yl+1/subrows<=cy-20.5/2||yl>=cy+20.5/2)continue;
  const mid=cx-slope*(ym-cy);
  for(let c=0;c<=28;c++){const v=2*Math.abs(mid-c);if(v>1.8&&v<4.6)wb.push(v);}
 }
 const uniq=a=>a.sort((a,b)=>a-b).filter((v,i,a)=>i===0||v-a[i-1]>1e-10);
 return {width:uniq(wb),height:uniq(hb)};
}
function cell(q,b=breaks(q)){
 const bounds=(arr,x)=>{for(let i=0;i<arr.length-1;i++)if(x>=arr[i]-1e-12&&x<arr[i+1]-1e-12)return[arr[i],arr[i+1]];return[arr.at(-2),arr.at(-1)];};
 return {w:bounds(b.width,q[3]),h:bounds(b.height,q[2])};
}
function patch(q,c=cell(q),subrows=16) {
 const at=(w,h)=>{const p=q.slice();p[3]=w;p[2]=h;return render(p,subrows);};
 const A=at(c.w[0],c.h[0]),P=at(c.w[1],c.h[0]),R=at(c.w[0],c.h[1]),S=at(c.w[1],c.h[1]);
 const B=sub(P,A),C=sub(R,A),D=sub(sub(S,P),sub(R,A)),Q=basis([B,C,D],1e-10);
 return {A,B,C,D,Q,cell:c,at,predict:(u,v)=>add(add(add(A,B,u),C,v),D,u*v)};
}

function analytic(q,subrows=16) {
 const [cx,cy,h,w,angle]=q,s=Math.tan(angle*Math.PI/180),ds=(1+s*s)*Math.PI/180;
 const im=new Float64Array(784),J=Array.from({length:5},()=>new Float64Array(784)),top=cy-h/2,bot=cy+h/2;
 for(let k=0;k<28*subrows;k++){
  const yl=k/subrows,yh=(k+1)/subrows,ym=(k+.5)/subrows,wy=Math.max(0,Math.min(yh,bot)-Math.max(yl,top));
  if(!wy)continue;
  const dcy=+(bot<yh)-+(top>yl),dh=.5*(+(bot<yh)+ +(top>yl));
  const mid=cx-s*(ym-cy),L=mid-w/2,R=mid+w/2,row=Math.floor(k/subrows);
  for(let c=Math.max(0,Math.floor(L));c<Math.min(28,Math.ceil(R));c++){
   const ov=Math.max(0,Math.min(R,c+1)-Math.max(L,c));if(!ov)continue;
   const ar=+(R<c+1),al=+(L>c),dc=ar-al,dw=.5*(ar+al),ix=row*28+c;
   im[ix]+=ov*wy;
   J[0][ix]+=dc*wy;
   J[1][ix]+=dc*s*wy+ov*dcy;
   J[2][ix]+=ov*dh;
   J[3][ix]+=dw*wy;
   J[4][ix]+=-dc*(ym-cy)*ds*wy;
  }
 }
 for(let j=0;j<5;j++)for(let i=0;i<784;i++)J[j][i]*=ranges[j][1]-ranges[j][0];
 return {im,J};
}

function optimizePath(start,end,free,N=24,maxiter=300,initial=null) {
 const toUnit=q=>q.map((v,j)=>(v-ranges[j][0])/(ranges[j][1]-ranges[j][0]));
 const toQ=u=>u.map((v,j)=>ranges[j][0]+v*(ranges[j][1]-ranges[j][0]));
 const a=toUnit(start),b=toUnit(end),dim=(N-1)*free.length;
 let x=Array.from({length:dim},(_,i)=>{const k=Math.floor(i/free.length)+1,j=free[i%free.length];return initial?initial[k][j]:a[j]+(b[j]-a[j])*k/N;});
 const unpack=x=>Array.from({length:N+1},(_,k)=>k===0?a.slice():k===N?b.slice():a.map((v,j)=>{const p=free.indexOf(j);return p<0?v+(b[j]-v)*k/N:x[(k-1)*free.length+p];}));
 const evaluate=x=>{
  const us=unpack(x),states=us.map(u=>analytic(toQ(u))),grad=new Float64Array(dim);let value=0;
  const ds=[];for(let k=0;k<N;k++){const d=sub(states[k+1].im,states[k].im);ds.push(d);value+=N*dot(d,d);}
  for(let k=1;k<N;k++){const pixelGrad=scale(sub(ds[k-1],ds[k]),2*N);for(let p=0;p<free.length;p++)grad[(k-1)*free.length+p]=dot(pixelGrad,states[k].J[free[p]]);}
  return {value,grad,us};
 };
 let current=evaluate(x),hist=[],accepted=0,failed=false;
 for(let iter=0;iter<maxiter;iter++){
  let d=Array.from(current.grad),alphas=[];
  for(let h=hist.length-1;h>=0;h--){const z=hist[h],alpha=z.rho*dot(z.s,d);alphas[h]=alpha;d=Array.from(add(d,z.y,-alpha));}
  if(hist.length){const last=hist.at(-1);d=Array.from(scale(d,dot(last.s,last.y)/dot(last.y,last.y)));}
  else d=d.map(v=>v*.0001);
  for(let h=0;h<hist.length;h++){const z=hist[h],beta=z.rho*dot(z.y,d);d=Array.from(add(d,z.s,alphas[h]-beta));}
  d=d.map(v=>-v);
  if(dot(d,current.grad)>=0){hist=[];d=Array.from(scale(current.grad,-.0001));}
  let step=1,next=null,xn=null;
  for(let ls=0;ls<22;ls++){
   xn=x.map((v,i)=>Math.max(0,Math.min(1,v+step*d[i])));
   const displacement=sub(xn,x),slope=dot(current.grad,displacement);
   next=evaluate(xn);
   if(next.value<=current.value+1e-4*slope)break;
   step*=.5;next=null;
  }
  if(!next){failed=true;break;}
  const svec=sub(xn,x),yvec=sub(next.grad,current.grad),sy=dot(svec,yvec);
  if(sy>1e-12){hist.push({s:svec,y:yvec,rho:1/sy});if(hist.length>12)hist.shift();}
  const reduction=current.value-next.value;
  x=xn;current=next;accepted++;
  if(reduction<1e-12*Math.max(1,current.value)&&norm(current.grad)<1e-4)break;
 }
 const us=unpack(x),qs=us.map(toQ),length=(samples=32)=>{
  let prev=render(qs[0]),L=0;
  for(let k=0;k<N;k++)for(let m=1;m<=samples;m++){
   const q=qs[k].map((v,j)=>v+(qs[k+1][j]-v)*m/samples),im=render(q);L+=dist(im,prev);prev=im;
  }return L;
 };
 return {energy:current.value,iterations:accepted,failed,gradientNorm:norm(current.grad),qs,us,length};
}
return {optimizePath};

})();
