
function runExperiments(g,an,ch) {
 const {ranges,render,dot,norm,sub,add,scale,dist,basis,perp,singular,eigSym,rng,quantiles,breaks,patch,jac}=g;
 const output={scope:"Only grey_ones.py's controlled five-knob generated family; float64 mathematical coverage rule unless stated otherwise.",provenance:{generator:"grey_ones.py",generatorGitBlob:"d59685738e7303017503b2cd56bac0706c9dcfd9",runtime:"Dependency-free JavaScript; renderer port; local Python reference validation not run in original session."}};
 let r=rng(90210),patchRows=[],errors=[],massError=0;
 for(let i=0;i<500;i++){
  const q=ranges.map(([a,b])=>a+(b-a)*(.001+.998*r())),p=patch(q),u=.37,v=.61;
  const w=p.cell.w[0]+u*(p.cell.w[1]-p.cell.w[0]),h=p.cell.h[0]+v*(p.cell.h[1]-p.cell.h[0]),im=p.at(w,h);
  const fu=add(p.B,p.D,v),fv=add(p.C,p.D,u),normal=norm(perp(p.D,basis([fu,fv]))),det=dot(fu,fu)*dot(fv,fv)-dot(fu,fv)**2;
  errors.push(dist(im,p.predict(u,v)));massError=Math.max(massError,Math.abs(Array.from(im).reduce((a,b)=>a+b,0)-w*h));
  patchRows.push({q,rank:p.Q.length,K:-normal*normal/det,wSpan:p.cell.w[1]-p.cell.w[0],hSpan:p.cell.h[1]-p.cell.h[0],normalFraction:normal/Math.max(1e-30,norm(p.D))});
 }
 output.widthHeight={count:500,ranks:patchRows.reduce((a,x)=>(a[x.rank]=(a[x.rank]||0)+1,a),{}),maxMassError:massError,error:quantiles(errors),curvature:quantiles(patchRows.map(x=>x.K)),widthSpan:quantiles(patchRows.map(x=>x.wSpan)),heightSpan:quantiles(patchRows.map(x=>x.hSpan)),mixedNormalFraction:quantiles(patchRows.map(x=>x.normalFraction))};
 output.widthHeightRaw=patchRows;
 r=rng(77);let derivativeErrors=[],imageError=0;
 for(let i=0;i<100;i++){
  const q=ranges.map(([a,b])=>a+(b-a)*r()),a=an.analytic(q),fd=jac(q,1e-7);
  imageError=Math.max(imageError,dist(a.im,render(q)));for(let j=0;j<5;j++)derivativeErrors.push(norm(sub(a.J[j],fd[j]))/norm(a.J[j]));
 }
 output.derivatives={states:100,imageMaxError:imageError,jacRelativeError:quantiles(derivativeErrors)};
 // Exact nine-pattern chart and independent evaluation of its polynomial.
 r=rng(345);const chartRanks={},chartErrors=[],chartExamples={};
 for(let i=0;i<2000;i++){
  const q=ranges.map(([a,b])=>a+(b-a)*(.0001+.9998*r())),e=ch.exactChart(q),vs=e.vs.map(v=>scale(v,1/Math.max(1e-30,norm(v)))),rank=basis(vs,1e-9).length;
  chartRanks[rank]=(chartRanks[rank]||0)+1;if(!chartExamples[rank])chartExamples[rank]={q,sv:singular(vs)};
  if(i<100){
   const [t,b,a,w,s]=e.z;let ym=Infinity,xm=Infinity;
   for(const y of [t,b]){const f=y*16-Math.floor(y*16);ym=Math.min(ym,f/16,(1-f)/16);}
   for(let k=Math.floor(t*16);k<Math.ceil(b*16);k++)for(const sign of [-1,1]){const x=a-s*(k+.5)/16+sign*w/2,f=x-Math.floor(x);xm=Math.min(xm,f,1-f);}
   const steps=[ym/4,ym/4,xm/8,xm/4,xm/(8*28)],d=steps.map(x=>x*(2*r()-1));
   let factor=1,qq;
   for(;;){
    const z=e.z.map((x,j)=>x+factor*d[j]),cy=(z[0]+z[1])/2;
    qq=[z[2]-z[4]*cy,cy,z[1]-z[0],z[3],Math.atan(z[4])*180/Math.PI];
    if(qq.every((x,j)=>x>=ranges[j][0]&&x<=ranges[j][1]))break;
    factor*=.5;
   }
   chartErrors.push(dist(render(qq),e.predict(scale(d,factor))));
  }
 }
 output.fullCharts={n:2000,ranks:chartRanks,errors:quantiles(chartErrors),examples:chartExamples};
 const kinkQ=[14.43,14.2,19.6,3.2,17.3],kinks=[];
 for(const eps of [.01,.001,.0001,.00001,.000001]){
  const lo=kinkQ.slice(),hi=kinkQ.slice();lo[2]-=eps;hi[2]+=eps;
  const a=an.analytic(lo),b=an.analytic(hi),Qa=basis(a.J),Qb=basis(b.J);
  const values=eigSym(Qb.map(v=>Qb.map(w=>Qa.reduce((s,u)=>s+dot(v,u)*dot(w,u),0))));
  kinks.push({heightHalfStep:eps,pixelDistance:dist(a.im,b.im),principalAngles:values.map(x=>Math.acos(Math.sqrt(Math.max(0,Math.min(1,x.value))))*180/Math.PI),rankBefore:Qa.length,rankAfter:Qb.length});
 }
 output.kink={q:kinkQ,tests:kinks};
 const q=[14.43,14.62,19.77,3.2,17.3],origin=render(q),lo=q.slice(),hi=q.slice();lo[2]=19;lo[3]=1.8;hi[2]=20.5;hi[3]=4.6;
 const low=render(lo),high=render(hi),active=Array.from({length:784},(_,i)=>i).filter(i=>Math.abs(high[i]-low[i])>1e-12);
 r=rng(654);let Q=[],trace=[];
 for(let i=0;i<3000;i++){
  const p=q.slice();p[2]=19+1.5*r();p[3]=1.8+2.8*r();const im=render(p),v=Float64Array.from(active,i=>im[i]-origin[i]),vperp=perp(perp(v,Q),Q),n=norm(vperp);
  if(n>1e-9)Q.push(scale(vperp,1/n));if([9,49,99,199,499,999,1999,2999].includes(i))trace.push({samples:i+1,rank:Q.length});
 }
 const heldout=[];for(let i=0;i<1000;i++){const p=q.slice();p[2]=19+1.5*r();p[3]=1.8+2.8*r();const im=render(p);heldout.push(norm(perp(Float64Array.from(active,i=>im[i]-origin[i]),Q)));}
 const br=breaks(q);let maxVertexResidual=0,worst=null;
 for(const h of br.height)for(const w of br.width){
  const p=q.slice();p[2]=h;p[3]=w;const im=render(p),res=norm(perp(perp(Float64Array.from(active,i=>im[i]-origin[i]),Q),Q));
  if(res>maxVertexResidual){maxVertexResidual=res;worst={w,h};}
 }
 output.globalSlice={q,activeCoordinates:active.length,rank:Q.length,training:3000,heldout:1000,rankTrace:trace,heldoutResidual:quantiles(heldout),widthIntervals:br.width.length-1,heightIntervals:br.height.length-1,totalGridVertices:br.width.length*br.height.length,maxVertexResidual,worst};
 const precision=[];
 for(const eps of [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]){
  const ref=an.analytic(q).J,a=jac(q,eps,16,false),b=jac(q,eps,16,true);
  const denom=Math.sqrt(ref.reduce((s,v)=>s+dot(v,v),0));
  precision.push({normalizedStep:eps,float64Relative:Math.sqrt(a.reduce((s,v,j)=>s+norm(sub(v,ref[j]))**2,0))/denom,float32Relative:Math.sqrt(b.reduce((s,v,j)=>s+norm(sub(v,ref[j]))**2,0))/denom});
 }
 output.precision={q,rows:precision};
 if(output.widthHeight.error.max>1e-11||output.widthHeight.maxMassError>1e-10||output.fullCharts.errors.max>1e-10||output.globalSlice.maxVertexResidual>1e-8)throw new Error("Geometry validation failed");
 return output;
}

if (typeof module !== "undefined") module.exports={runExperiments};
if (typeof require !== "undefined" && require.main === module) {
 const fs=require("node:fs"),path=require("node:path");
 const result=runExperiments(require("./generated_geometry_core.cjs"),require("./generated_geometry_analytic.cjs"),require("./generated_geometry_charts.cjs"));
 const filename=path.join(__dirname,"generated_geometry_reproduced.json");
 fs.writeFileSync(filename,JSON.stringify(result,null,2)+"\n");
 console.log(JSON.stringify({output:path.basename(filename),widthHeight:result.widthHeight,fullCharts:result.fullCharts,globalSlice:result.globalSlice},null,2));
}

