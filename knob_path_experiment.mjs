
// Matches the 16-subrow coverage rule read from grey_ones.py earlier in this chat.
// This standalone mirror supports float64 diagnostics before float32 storage.
const PIXELS = 784;
const RANGES = [[14,15],[14,15],[19,20.5],[1.8,4.6],[-10,35]];

function render(p) {
  const [cx, cy, height, width, angle] = p;
  const slope = Math.tan(angle * Math.PI / 180);
  const out = new Float64Array(PIXELS);
  const top = cy - height / 2, bottom = cy + height / 2;
  for (let strip = 0; strip < 448; strip++) {
    const low = strip / 16, midpoint = low + 1 / 32;
    const vertical = Math.max(0, Math.min(low + 1 / 16, bottom) - Math.max(low, top)) * 16;
    if (!vertical) continue;
    const center = cx - slope * (midpoint - cy);
    const left = center - width / 2, right = center + width / 2;
    const row = Math.floor(strip / 16);
    for (let col = Math.max(0, Math.floor(left)); col < Math.min(28, Math.ceil(right)); col++) {
      const overlap = Math.max(0, Math.min(right, col + 1) - Math.max(left, col));
      out[row * 28 + col] += overlap * vertical / 16;
    }
  }
  return out;
}

function moved(p, k, step) { const q = p.slice(); q[k] += step; return q; }
function sub(a, b) { return Float64Array.from(a, (v, i) => v - b[i]); }
function plus(a, b) { return Float64Array.from(a, (v, i) => v + b[i]); }
function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
function norm(a) { return Math.sqrt(dot(a, a)); }
function maxabs(a) { return a.reduce((s, v) => Math.max(s, Math.abs(v)), 0); }
function angle(a, b) {
  return Math.acos(Math.max(-1, Math.min(1, dot(a,b) / (norm(a)*norm(b))))) * 180 / Math.PI;
}
function rng(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state,1664525) + 1013904223) >>> 0; return state / 4294967296; };
}

// Sampling uses the renderer. Fitting receives the resulting pixel arrays only.
// Patch locations and the candidate dimension are supplied, not discovered.
function sampleImages(center, radiusFraction, count, seed) {
  const random = rng(seed);
  return Array.from({length:count}, () => render(center.map((v,k) =>
    v + (2 * random() - 1) * radiusFraction * (RANGES[k][1] - RANGES[k][0]))));
}

function symmetricEigen(matrix) {
  const n = matrix.length;
  const a = matrix.map(row => Float64Array.from(row));
  const v = Array.from({length:n}, (_,i) => Float64Array.from({length:n}, (_,j) => +(i===j)));
  for (let sweep = 0; sweep < 35; sweep++) {
    let maxOff = 0;
    const scale = a.reduce((sum,row,i) => sum + Math.abs(row[i]), 0);
    for (let p = 0; p < n; p++) for (let q = p+1; q < n; q++) {
      const off = a[p][q]; maxOff = Math.max(maxOff,Math.abs(off));
      if (Math.abs(off) < scale * 1e-14) continue;
      const tau = (a[q][q] - a[p][p]) / (2 * off);
      const t = (tau>=0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1+tau*tau));
      const c = 1 / Math.sqrt(1+t*t), s = t*c;
      a[p][p] -= t * off; a[q][q] += t * off; a[p][q] = a[q][p] = 0;
      for (let k = 0; k < n; k++) {
        if (k !== p && k !== q) {
          const kp = a[k][p], kq = a[k][q];
          a[k][p] = a[p][k] = c*kp-s*kq;
          a[k][q] = a[q][k] = s*kp+c*kq;
        }
        const kp = v[k][p], kq = v[k][q];
        v[k][p] = c*kp-s*kq; v[k][q] = s*kp+c*kq;
      }
    }
    if (maxOff < scale * 1e-13) break;
  }
  const ids = Array.from({length:n}, (_,i) => i).sort((i,j) => a[j][j]-a[i][i]);
  return {
    values: ids.map(i => a[i][i]),
    vectors: ids.map(i => Float64Array.from(v,row => row[i]))
  };
}

function imagePCA(images) {
  const n = images.length, mean = new Float64Array(PIXELS), variance = new Float64Array(PIXELS);
  for (const image of images) for (let p=0;p<PIXELS;p++) mean[p] += image[p]/n;
  for (const image of images) for (let p=0;p<PIXELS;p++) variance[p] += (image[p]-mean[p])**2/n;
  const maxVariance = Math.max(...variance);
  const active = Array.from({length:PIXELS}, (_,i)=>i).filter(i=>variance[i]>maxVariance*1e-16);
  const m = active.length, covariance = Array.from({length:m},()=>new Float64Array(m));
  for (const image of images) {
    const delta = active.map(p=>image[p]-mean[p]);
    for (let i=0;i<m;i++) for(let j=0;j<=i;j++) covariance[i][j] += delta[i]*delta[j]/n;
  }
  for(let i=0;i<m;i++) for(let j=0;j<i;j++) covariance[j][i]=covariance[i][j];
  const eigen = symmetricEigen(covariance);
  const encode = (image, dimension) => Array.from({length:dimension}, (_,j)=>
    active.reduce((sum,p,i)=>sum+(image[p]-mean[p])*eigen.vectors[j][i],0)/Math.sqrt(eigen.values[j]));
  return {mean,active,...eigen,encode};
}

function polynomialTerms(z, quadratic) {
  const result = [1,...z];
  if(quadratic) for(let i=0;i<z.length;i++) for(let j=i;j<z.length;j++) result.push(z[i]*z[j]);
  return result;
}

function solveMany(a,b) {
  const n=a.length, m=b[0].length, rows=a.map((row,i)=>[...row,...b[i]]);
  for(let p=0;p<n;p++) {
    let best=p;
    for(let i=p+1;i<n;i++) if(Math.abs(rows[i][p])>Math.abs(rows[best][p])) best=i;
    [rows[p],rows[best]]=[rows[best],rows[p]];
    const pivot=rows[p][p];
    if(Math.abs(pivot)<1e-12) throw new Error('Singular polynomial fit');
    for(let j=p;j<n+m;j++) rows[p][j]/=pivot;
    for(let i=0;i<n;i++) if(i!==p) {
      const multiplier=rows[i][p];
      for(let j=p;j<n+m;j++) rows[i][j]-=multiplier*rows[p][j];
    }
  }
  return rows.map(row=>row.slice(n));
}

// Linear encoder plus quadratic decoder: a graph over the local PCA plane.
// No generator controls or their values enter this fitting function.
function fitPixelChart(train,pca,dimension,quadratic) {
  const features=train.map(image=>polynomialTerms(pca.encode(image,dimension),quadratic));
  const q=features[0].length,m=pca.active.length;
  const a=Array.from({length:q},()=>new Float64Array(q));
  const b=Array.from({length:q},()=>new Float64Array(m));
  for(let i=0;i<train.length;i++) for(let j=0;j<q;j++) {
    for(let k=0;k<q;k++) a[j][k]+=features[i][j]*features[i][k]/train.length;
    for(let k=0;k<m;k++) b[j][k]+=features[i][j]*(train[i][pca.active[k]]-pca.mean[pca.active[k]])/train.length;
  }
  const weights=solveMany(a,b);
  const decode=z=>{
    const f=polynomialTerms(z,quadratic),out=Float64Array.from(pca.mean);
    for(let k=0;k<m;k++) out[pca.active[k]]+=f.reduce((sum,v,j)=>sum+v*weights[j][k],0);
    return out;
  };
  return {decode,predict:image=>decode(pca.encode(image,dimension))};
}

function metrics(test,predict,mean) {
  let sse=0,total=0,worstL2=0,worstPixel=0;
  for(const image of test) {
    const error=sub(image,predict(image)),delta=sub(image,mean);
    sse+=dot(error,error); total+=dot(delta,delta);
    worstL2=Math.max(worstL2,norm(error)); worstPixel=Math.max(worstPixel,maxabs(error));
  }
  return {
    reconstructedVariancePercent:100*(1-sse/total),
    rmsPixelL2:Math.sqrt(sse/test.length),
    worstPixelL2:worstL2,
    worstAbsolutePixelError:worstPixel
  };
}

function runInteractions() {
  const base=[14.43,14.62,19.73,3.17,17.3],original=render(base),leanWidth=[];
  for(const factor of [1,.5,.1,.01,.001,.0001]) {
    const widthStep=.2*factor,leanStep=5*factor;
    const width=render(moved(base,3,widthStep)),lean=render(moved(base,4,leanStep));
    const jointImage=render(moved(moved(base,3,widthStep),4,leanStep));
    const dw=sub(width,original),da=sub(lean,original),joint=sub(jointImage,original);
    const mixed=sub(joint,plus(dw,da));
    const viaWidth=plus(dw,sub(jointImage,width)),viaLean=plus(da,sub(jointImage,lean));
    leanWidth.push({
      factor,widthStep,leanStepDegrees:leanStep,jointPixelL2:norm(joint),
      frozenSumResidualL2:norm(mixed),residualOverJoint:norm(mixed)/norm(joint),
      leanChangeDirectionAngleDegrees:angle(da,sub(jointImage,width)),
      updatedOrderResidualL2:Math.max(norm(sub(viaWidth,joint)),norm(sub(viaLean,joint))),
      mixedPixelMax:maxabs(mixed)
    });
  }
  const widthHeight=[];
  for(const factor of [1,.1,.01,.001,.0001]) {
    const widthStep=.2*factor,heightStep=.3*factor;
    const width=render(moved(base,3,widthStep)),height=render(moved(base,2,heightStep));
    const both=render(moved(moved(base,3,widthStep),2,heightStep));
    const mixed=sub(sub(both,original),plus(sub(width,original),sub(height,original)));
    widthHeight.push({factor,widthStep,heightStep,mixedPixelL2:norm(mixed),mixedOverStepProduct:norm(mixed)/(widthStep*heightStep)});
  }
  // Constructed counterexample: tangent, independent motions whose step sizes
  // depend on the other coordinate. They do not commute as parametrized flows.
  const leanStep=5,widthStep=.2,alpha=4;
  let q=moved(base,4,leanStep);
  q=moved(q,3,widthStep*(1+alpha*(q[4]-base[4])/45));
  q=moved(q,4,-leanStep);
  q=moved(q,3,-widthStep*(1+alpha*(q[4]-base[4])/45));
  const ordinary=moved(moved(moved(moved(base,4,leanStep),3,widthStep),4,-leanStep),3,-widthStep);
  return {base,leanWidth,widthHeight,loop:{
    ordinaryKnobLoopPixelL2:norm(sub(render(ordinary),original)),
    rescaledTangentLoopPixelL2:norm(sub(render(q),original)),
    rescaledTangentLoopWidthDrift:q[3]-base[3]
  }};
}

function runLocalCharts() {
  const center=[14.43,14.62,19.73,3.17,17.3],results=[];
  for(const radiusFraction of [.03,.01,.003,.001,.0001,.00001]) {
    const train=sampleImages(center,radiusFraction,256,20261003);
    const test=sampleImages(center,radiusFraction,128,20261004);
    const pca=imagePCA(train);
    const linear=fitPixelChart(train,pca,5,false);
    const quadratic=fitPixelChart(train,pca,5,true);
    const four=fitPixelChart(train,pca,4,true);
    results.push({
      radiusFraction,activePixelCount:pca.active.length,
      eigenvalueRatios:pca.values.slice(0,9).map(v=>v/pca.values[0]),
      linear5:metrics(test,linear.predict,pca.mean),
      quadratic5:metrics(test,quadratic.predict,pca.mean),
      quadratic4:metrics(test,four.predict,pca.mean)
    });
  }
  return {trainingImagesPerPatch:256,testImagesPerPatch:128,
    trainingUsesKnobValues:false,dimensionFiveSupplied:true,
    patchesProvidedRatherThanDiscovered:true,results};
}

function denseRender(p) {
  const [cx,cy,height,width,angle]=p,slope=Math.tan(angle*Math.PI/180);
  const out=new Float64Array(PIXELS);
  for(let row=0;row<28;row++) for(let col=0;col<28;col++) {
    let total=0;
    for(let subrow=0;subrow<16;subrow++) {
      const low=row+subrow/16,midpoint=low+1/32;
      const vertical=Math.max(0,Math.min(low+1/16,cy+height/2)-Math.max(low,cy-height/2))*16;
      const center=cx-slope*(midpoint-cy);
      const horizontal=Math.max(0,Math.min(center+width/2,col+1)-Math.max(center-width/2,col));
      total+=horizontal*vertical;
    }
    out[row*28+col]=total/16;
  }
  return out;
}

function validate() {
  const random=rng(761904);
  const states=Array.from({length:100},()=>RANGES.map(([lo,hi])=>lo+(hi-lo)*random()));
  for(let bits=0;bits<32;bits++) states.push(RANGES.map(([lo,hi],k)=>bits&(1<<k)?hi:lo));
  let maxPixelDifference=0,float32PixelMismatches=0,totalInkError=0,fullRowError=0;
  for(const state of states) {
    const fast=render(state),dense=denseRender(state);
    for(let p=0;p<PIXELS;p++) {
      maxPixelDifference=Math.max(maxPixelDifference,Math.abs(fast[p]-dense[p]));
      if(Math.fround(fast[p])!==Math.fround(dense[p])) float32PixelMismatches++;
    }
    totalInkError=Math.max(totalInkError,Math.abs(fast.reduce((a,b)=>a+b,0)-state[2]*state[3]));
    for(let row=6;row<=22;row++) fullRowError=Math.max(fullRowError,Math.abs(fast.slice(row*28,(row+1)*28).reduce((a,b)=>a+b,0)-state[3]));
  }
  if(maxPixelDifference>1e-12 || float32PixelMismatches || totalInkError>1e-10 || fullRowError>1e-12) throw new Error('Renderer validation failed');
  return {randomStates:100,parameterBoxCorners:32,
    independentDenseMaximumPixelDifference:maxPixelDifference,float32PixelMismatches,
    maximumTotalInkIdentityError:totalInkError,maximumFullRowWidthIdentityError:fullRowError,
    repositoryPythonExecuted:false};
}

function run() {
  const validation=validate(),interactions=runInteractions(),localCharts=runLocalCharts();
  if(interactions.leanWidth.some(row=>row.updatedOrderResidualL2>1e-12)) throw new Error('Composition identity failed');
  if(!(interactions.leanWidth[0].residualOverJoint>.1)) throw new Error('Joint-change counterexample missing');
  if(!(localCharts.results[1].quadratic5.reconstructedVariancePercent>99.99)) throw new Error('Local chart check failed');
  return {experiment:'Knob composition and local pixel-only quadratic charts',
    date:'2026-10-02',implementation:'JavaScript float64 mirror of the previously read 16-subrow renderer',
    validation,interactions,localCharts};
}

const fs = await import('node:fs/promises');
const output = new URL('./knob_path_results.json', import.meta.url);
const result = run();
await fs.writeFile(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));

