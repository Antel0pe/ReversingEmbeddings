"""Independent reproduction checker; no learner or renderer imports."""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated_one_data_only'


def measured(images):
    y,x=np.indices(images.shape[-2:]);x=x+.5;y=y+.5
    result=[]
    for image in images:
        im=np.asarray(image,float);mass=im.sum()
        cx=np.sum(im*x)/mass;cy=np.sum(im*y)/mass
        result.append([mass,cx,cy,np.sum(im*x*x)/mass-cx*cx,
            np.sum(im*y*y)/mass-cy*cy,np.sum(im*x*y)/mass-cx*cy])
    return np.array(result)


def poly(u,degree):
    columns=[np.ones(len(u))]
    for d in range(1,degree+1):
        for term in itertools.combinations_with_replacement(range(u.shape[1]),d):
            col=np.ones(len(u))
            for k in term:col*=u[:,k]
            columns.append(col)
    return np.stack(columns,axis=1)


def independently_predict(x,y,query,degree,neighbors):
    center=x.mean(0);scale=np.maximum(np.std(x,axis=0),1e-12)
    u=(x-center)/scale;v=(query-center)/scale
    distance,index=cKDTree(u).query(v,k=neighbors)
    output=[]
    flat=y.reshape(len(y),-1).astype(float)
    all_constant=np.all(flat==flat[0],axis=0)
    for point,dist,indices in zip(v,distance,index):
        radius=max(float(dist[-1]),1e-12)
        a=poly((u[indices]-point)/radius,degree)
        root=np.exp(-2*(dist/radius)**2)
        # Separate implementation: first row of the weighted design pseudoinverse.
        weights=np.linalg.pinv(a*root[:,None],rcond=1e-10)[0]*root
        value=weights@flat[indices]
        # Exact channels were stored without regression in the primary program.
        value[all_constant]=flat[0,all_constant]
        output.append(value.reshape(y.shape[1:]))
    return np.array(output)


def simple_errors(true,pred):
    delta=np.asarray(pred,float)-np.asarray(true,float)
    norms=np.linalg.norm(delta.reshape(len(delta),-1),axis=1)
    return dict(max_pixel=float(np.max(abs(delta))),max_image_l2=float(np.max(norms)))


def main():
    frozen=json.loads((OUT/'frozen_model.json').read_text())
    result=json.loads((OUT/'results.json').read_text())
    assert hashlib.sha256((OUT/'frozen_model.json').read_bytes()).hexdigest()==result['model_sha256']
    for rel,value in frozen['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value,rel
    # Information audit checks imports and the absence of filesystem access in
    # both discovery modules; it does not claim a researcher was source-blind.
    allowed={'numpy','scipy','itertools','learn'}
    audits={}
    for name in ['learn.py','crease.py']:
        tree=ast.parse((Path(__file__).parent/name).read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(x.name.split('.')[0] for x in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module.split('.')[0])
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                assert node.func.id not in {'open','exec','eval','__import__'},name
        assert set(imports)<=allowed,(name,imports)
        audits[name]=imports
    with np.load(OUT/'evaluation_public.npz') as data:
        test=data['images'];coordinates=data['coordinates']
    stored=np.load(OUT/'predictions.npz')
    rng=np.random.default_rng(3001)
    indices=np.unique(np.r_[rng.choice(len(test),64,replace=False),
        [result['counts']['32768'][k]['worst_l2_index'] for k in ['images_only','supplied_coordinates']]])
    checks={}
    for condition in ['images_only','supplied_coordinates']:
        if condition=='images_only':
            with np.load(OUT/'training_images.npz') as d:train_images=d['images']
            keep=frozen['selected_measurements']
            x=measured(train_images)[:,keep];query=measured(test[indices])[:,keep]
            params=frozen['image_only_hyperparameters']
        else:
            with np.load(OUT/'training_with_coordinates.npz') as d:
                train_images=d['images'];x=d['coordinates']
            query=coordinates[indices];params=frozen['supplied_coordinate_hyperparameters']
        predicted=independently_predict(x,train_images,query,**params)
        agreement=simple_errors(stored[condition][indices],predicted)
        assert agreement['max_pixel']<2e-7,agreement
        m=simple_errors(test,stored[condition])
        for key in ['max_pixel','max_image_l2']:
            expected=result['counts']['32768'][condition][key]
            assert abs(m[key]-expected)<1e-10
        assert m['max_pixel']>2e-6 and m['max_image_l2']>1e-5
        assert np.min(stored[condition])<-2e-6 or np.max(stored[condition])>1+2e-6
        checks[condition]=dict(recomputed_states=len(indices),prediction_agreement=agreement,
                               full_stored_errors=m,scientific_accuracy_goal_met=False)
    # Independently evaluate each frozen hinge equation, including all pixels.
    hinge=json.loads((OUT/'crease_results.json').read_text())
    for rel,value in hinge['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value,rel
    checks['local_crease']={}
    for label,record in hinge['conditions'].items():
        model_path=OUT/f'crease_model_{label}.npz'
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==record['model_sha256']
        with np.load(model_path) as model,np.load(OUT/f'crease_test_{label}_public.npz') as data:
            u=(data['coordinates']-model['center'])/model['scale']
            ramp=np.maximum(0,u@model['normal']-model['offset'])
            a=np.column_stack([poly(u,3),ramp[:,None]*poly(u,2)])
            pred=(a@model['coefficient']).reshape(data['images'].shape)
            agree=simple_errors(data['predicted'],pred)
            assert agree['max_pixel']<1e-10,agree
            metric=simple_errors(data['images'],pred)
            assert metric['max_pixel']<2e-6 and metric['max_image_l2']<1e-5
            checks['local_crease'][label]=metric
    answer=dict(reproduction_pass=True,full_family_explanation_recovered=False,
                information_audit=audits,checks=checks)
    (OUT/'checker_results.json').write_text(json.dumps(answer,indent=2)+'\n')
    print(json.dumps(answer,indent=2))

if __name__=='__main__':main()
