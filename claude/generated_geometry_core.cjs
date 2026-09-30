module.exports = (function(){
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
return {ranges,render,dot,norm,sub,add,scale,dist,maxabs,basis,perp,eigSym,singular,jac,rng,quantiles,breaks,cell,patch};

})();

