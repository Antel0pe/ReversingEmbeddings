// Numerical surrogate experiment, not the repository's five-knob renderer.
// Run with: node experiments/knob-coordinate-numerical-check.mjs
// No packages required. All evaluation uses 784 pixels with intensities in [0,1].
function render(w,l) {
  const a = new Float64Array(784);
  for (let y=4;y<24;y++) {
    const c=13.5+l*(y-13.5), left=c-w/2, right=c+w/2;
    for(let x=0;x<28;x++)
      a[y*28+x]=Math.max(0,Math.min(x+.5,right)-Math.max(x-.5,left));
  }
  return a;
}
const add=(a,b,s=1)=>Float64Array.from(a,(v,i)=>v+s*b[i]);
const sub=(a,b)=>add(a,b,-1);
function dot(a,b) { let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s; }
const norm=a=>Math.sqrt(dot(a,a));
const rmse=(a,b)=>norm(sub(a,b))/28;
let state=20261002;
function rnd() { state=(Math.imul(1664525,state)+1013904223)>>>0;return state/4294967296; }
const train=[];
for(let i=0;i<21;i++)for(let j=0;j<21;j++)train.push(render(1.5+4*i/20,-.4+.8*j/20));
const mean=new Float64Array(784);
for(const a of train)for(let i=0;i<784;i++)mean[i]+=a[i]/train.length;
const centered=train.map(a=>sub(a,mean));
function cov(v) {
  const out=new Float64Array(784);
  for(const a of centered) {
    const k=dot(a,v)/centered.length;
    for(let i=0;i<784;i++)out[i]+=k*a[i];
  }
  return out;
}
const basis=[];
for(let k=0;k<2;k++) {
  let v=Float64Array.from({length:784},()=>rnd()-.5);
  for(let t=0;t<70;t++) {
    v=cov(v);
    for(const b of basis)v=add(v,b,-dot(v,b));
    const len=norm(v);
    v=Float64Array.from(v,x=>x/len);
  }
  basis.push(v);
}
function pca(a) {
  const d=sub(a,mean);let out=mean;
  for(const b of basis)out=add(out,b,dot(d,b));
  return out;
}
function interp(w,l,N) {
  const u=(w-1.5)/4*(N-1),v=(l+.4)/.8*(N-1);
  const i=Math.min(N-2,Math.floor(u)),j=Math.min(N-2,Math.floor(v));
  const a=u-i,b=v-j,out=new Float64Array(784);
  for(let di=0;di<2;di++)for(let dj=0;dj<2;dj++) {
    const img=render(1.5+4*(i+di)/(N-1),-.4+.8*(j+dj)/(N-1));
    const wt=(di?a:1-a)*(dj?b:1-b);
    for(let k=0;k<784;k++)out[k]+=wt*img[k];
  }
  return out;
}
const tests=Array.from({length:1000},()=>[1.5+4*rnd(),-.4+.8*rnd()]);
const w0=3.4,l0=.07,base=render(w0,l0);
const stats={pca2:[],additive:[]};
for(const N of [5,9,17,33,65])stats["grid"+N]=[];
for(const [w,l] of tests) {
  const truth=render(w,l);
  stats.pca2.push(rmse(truth,pca(truth)));
  stats.additive.push(rmse(truth,sub(add(render(w,l0),render(w0,l)),base)));
  for(const N of [5,9,17,33,65])stats["grid"+N].push(rmse(truth,interp(w,l,N)));
}
const summary={};
for(const [key,a] of Object.entries(stats)) {
  a.sort((x,y)=>x-y);
  summary[key]={meanRMSE:a.reduce((s,x)=>s+x,0)/a.length,p95RMSE:a[949],maxRMSE:a[999]};
}
const finite=[];
for(const h of [1,.5,.25,.125,.0625]) {
  const w=w0+.4*h,l=l0+.08*h,A=render(w,l0),B=render(w0,l),C=render(w,l);
  const R=add(sub(C,A),sub(base,B));
  finite.push({stepScale:h,interactionL2:norm(R),additiveRMSE:norm(R)/28,
    jointL2:norm(sub(C,base)),relativeInteraction:norm(R)/norm(sub(C,base))});
}
const eigenResiduals=basis.map(b=>{
  const cb=cov(b);return norm(sub(cb,Float64Array.from(b,x=>dot(cb,b)*x)))/norm(cb);
});
if(Math.abs(dot(basis[0],basis[1]))>1e-10||eigenResiduals.some(x=>x>1e-8))
  throw new Error("PCA numerical verification failed");
if(summary.grid65.meanRMSE>=summary.grid33.meanRMSE)
  throw new Error("Grid refinement did not reduce held-out mean error");
console.log(JSON.stringify({status:"Synthetic surrogate, not repository renderer",summary,finite,
  checks:{rankBasisDot:dot(basis[0],basis[1]),eigenResiduals}},null,2));

