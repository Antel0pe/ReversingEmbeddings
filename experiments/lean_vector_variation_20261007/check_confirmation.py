"""Independent confirmation after fixing the vector dictionary size at 16."""
import json
import numpy as np
from run import BASE,RANGES,OUT,images,observed_arrows,scores

model=np.load(OUT/'width_model.npz')
rng=np.random.default_rng(10312026)
lo,hi=json.loads((OUT/'metrics.json').read_text())['width_model']['refinement_interval_discovered_from_vectors']
ws=np.r_[rng.uniform(*RANGES[3],1000),np.linspace(lo,hi,1001)+1.7321e-7]
p=np.tile(BASE,(len(ws),1));p[:,3]=ws
xx=images(p);target=observed_arrows(p)
q=(xx-model['origin'])@model['coordinate_vector']
B=model['variation_basis'][:,:16];coeff=model['variation_coefficients'][:,:16]
vals=np.column_stack([np.interp(q,model['coefficient_knots'],coeff[:,j]) for j in range(16)])
pred=model['base_lean_vector']+vals@B.T
report={'fixed_rank':16,'new_random_states':1000,'new_transition_states':1001,
        'random':scores(pred[:1000],target[:1000]),'transition_stress':scores(pred[1000:],target[1000:])}
(OUT/'confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
