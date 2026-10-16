"""Generic numeric-coordinate rules. No image geometry or generator imports.

Inputs: normalized coordinates in [0,1]^d and vectors returned by an oracle.
Output channels have no adjacency, layout, names, or physical meaning.
"""
from itertools import product
import numpy as np
from scipy.linalg import svd


class Observations:
    def __init__(self, oracle):
        self.oracle = oracle
        self.cache = {}

    def __call__(self, points):
        points = np.atleast_2d(points)
        keys = [tuple(x) for x in points]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            values = self.oracle(np.asarray(missing))
            self.cache.update(zip(missing, values))
        return np.asarray([self.cache[k] for k in keys])


def discover_axis_knots(observations, dimension, axis, contexts, tolerance,
                        max_knots, max_depth=36):
    """Bisect the worst failing interval; prune numerically redundant knots.

    Only max absolute output error is used, hence output-permutation invariant.
    Quarter/mid/three-quarter probes reduce (do not eliminate) missed features.
    Contexts are numeric coordinates of the other controls, supplied as samples.
    """
    traces = {}

    def at(t):
        key = float(t)
        if key not in traces:
            points = contexts.copy()
            points[:, axis] = t
            traces[key] = observations(points)
        return traces[key]

    def error(a, b):
        errors = []
        for f in [.25, .5, .75]:
            errors.append(np.max(np.abs(at(a + f*(b-a)) -
                                ((1-f)*at(a) + f*at(b)))))
        return float(max(errors))

    knots = [0., 1.]
    segments = {(0., 1.): (error(0., 1.), 0)}
    while len(knots) < max_knots:
        candidates = [(err, a, b, depth) for (a,b),(err,depth) in segments.items()
                      if err > tolerance and depth < max_depth]
        if not candidates:
            break
        _, a, b, depth = max(candidates)
        midpoint = (a+b)/2
        knots.append(midpoint)
        segments.pop((a,b))
        segments[(a,midpoint)] = (error(a,midpoint), depth+1)
        segments[(midpoint,b)] = (error(midpoint,b), depth+1)
    knots.sort()
    before_pruning = len(knots)
    # Removal must preserve BOTH the old knot and new quarter-point probes.
    changed = True
    while changed:
        changed = False
        for i in range(1, len(knots)-1):
            a, t, b = knots[i-1:i+2]
            f = (t-a)/(b-a)
            old_error = np.max(np.abs(at(t)-((1-f)*at(a)+f*at(b))))
            if old_error <= tolerance and error(a,b) <= tolerance:
                knots.pop(i)
                changed = True
                break
    # Near a detected kink, intersect the measured affine traces on its sides.
    # This removes bisection roundoff WITHOUT assuming what any output means.
    def intersection(a, left_end, right_start, b):
        l1, l2 = a+.25*(left_end-a), a+.75*(left_end-a)
        r1, r2 = right_start+.25*(b-right_start), right_start+.75*(b-right_start)
        sl=(at(l2)-at(l1))/(l2-l1)
        sr=(at(r2)-at(r1))/(r2-r1)
        il=at(l1)-l1*sl
        ir=at(r1)-r1*sr
        ds=sl-sr; di=ir-il
        # Sorting makes the reductions bitwise independent of channel order.
        norm=np.sum(np.sort((ds*ds).ravel()))
        if norm < 1e-20:
            return None
        t=float(np.sum(np.sort((ds*di).ravel()))/norm)
        if not a < t < b or np.max(np.abs(ds*t-di)) > tolerance:
            return None
        return t

    i=1
    while i < len(knots)-2:
        if knots[i+1]-knots[i] < 1e-6:
            t=intersection(knots[i-1],knots[i],knots[i+1],knots[i+2])
            if t is not None and knots[i]-1e-6 < t < knots[i+1]+1e-6:
                knots[i:i+2]=[t]
                continue
        i+=1
    for i in range(1,len(knots)-1):
        a,t,b=knots[i-1:i+2]
        candidate=intersection(a,t,t,b)
        if candidate is not None and abs(candidate-t)<1e-6:
            if max(error(a,candidate),error(candidate,b)) <= tolerance:
                knots[i]=candidate
    final_error = max(error(a,b) for a,b in zip(knots[:-1],knots[1:]))
    return np.asarray(knots), dict(before_pruning=before_pruning,
        after_pruning=len(knots), worst_context_probe_error=float(final_error),
        cap_reached=before_pruning == max_knots,
        trace_locations=len(traces))


class CoordinateRule:
    def __init__(self, oracle, dimension, contexts, tolerance=1e-9,
                 max_knots=129, svd_tolerance=1e-12):
        self.observations = Observations(oracle)
        self.knots = []
        self.knot_diagnostics = []
        for axis in range(dimension):
            knots, diag = discover_axis_knots(self.observations, dimension, axis,
                                              contexts, tolerance, max_knots)
            self.knots.append(knots)
            self.knot_diagnostics.append(diag)
        self.grid = np.asarray(list(product(*self.knots)))
        self.table = self.observations(self.grid)
        self.anchor = self.table[0].copy()
        differences = self.table - self.anchor
        # Work on active output channels only; selection has no spatial meaning.
        self.active = np.flatnonzero(np.max(np.abs(differences),axis=0) > 0)
        left, singular, right = svd(differences[:,self.active],
                                    full_matrices=False, lapack_driver='gesdd')
        self.singular_values = singular
        self.rank = int(np.sum(singular > (singular[0]*svd_tolerance)))
        self.patterns = np.zeros((self.rank, self.table.shape[1]))
        self.patterns[:,self.active] = right[:self.rank]
        self.weights = (left[:,:self.rank]*singular[:self.rank])
        self.shape = tuple(len(k) for k in self.knots)
        reconstruction = self.anchor + self.weights @ self.patterns
        self.factor_error = float(np.max(np.abs(reconstruction-self.table)))
        self.weights = self.weights.reshape(*self.shape, self.rank)
        self.table_tensor = self.table.reshape(*self.shape, self.table.shape[-1])

    def interpolate(self, points, factored=True):
        points = np.atleast_2d(points)
        if np.any(points < -1e-12) or np.any(points > 1+1e-12):
            raise ValueError('No extrapolation outside the supplied coordinate box.')
        n, dimension = points.shape
        lower=[]; fractions=[]
        for j, knots in enumerate(self.knots):
            index = np.clip(np.searchsorted(knots,points[:,j],side='right')-1,
                            0,len(knots)-2)
            lower.append(index)
            fractions.append((points[:,j]-knots[index])/(knots[index+1]-knots[index]))
        array = self.weights if factored else self.table_tensor
        result = np.zeros((n,array.shape[-1]))
        for bits in product((0,1),repeat=dimension):
            scale = np.ones(n)
            indices=[]
            for j, bit in enumerate(bits):
                indices.append(lower[j]+bit)
                scale *= fractions[j] if bit else 1-fractions[j]
            result += scale[:,None] * array[tuple(indices)]
        return self.anchor + result @ self.patterns if factored else result

    def change(self, points, axis, amount):
        target = np.atleast_2d(points).copy()
        target[:,axis] += amount
        return self.interpolate(target)-self.interpolate(points)

    def sparse_edits(self, point, axis, amount, threshold=1e-7):
        target = np.asarray(point).copy()
        target[axis] += amount
        old = self.interpolate(point)[0]
        new = self.interpolate(target)[0]
        delta = new-old
        indices = np.flatnonzero(np.abs(delta)>threshold)
        return indices, new[indices], delta[indices]

    def save(self, path):
        arrays = dict(anchor=self.anchor, patterns=self.patterns,
                      weights=self.weights, singular_values=self.singular_values)
        arrays.update({f'knots_{j}':k for j,k in enumerate(self.knots)})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls,path):
        arrays=np.load(path)
        obj=cls.__new__(cls)
        obj.anchor=arrays['anchor']
        obj.patterns=arrays['patterns']
        obj.weights=arrays['weights']
        obj.singular_values=arrays['singular_values']
        obj.knots=[arrays[f'knots_{j}'] for j in range(obj.weights.ndim-1)]
        obj.shape=obj.weights.shape[:-1]
        obj.rank=obj.patterns.shape[0]
        obj.table_tensor=obj.anchor+obj.weights @ obj.patterns
        return obj
