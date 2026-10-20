"""Restore a frozen scalar function without refitting."""
from common import Features


def feature_from_archive(archive,prefix):
    f=Features.__new__(Features)
    f.active=archive[prefix+'_active'];f.mean=archive[prefix+'_mean']
    f.projection=archive[prefix+'_projection'];f.centers=archive[prefix+'_centers']
    f.kind=str(archive[prefix+'_kind']);f.h=float(archive[prefix+'_h'])
    f.terms=[tuple(int(a) for a in row if a>=0) for row in archive[prefix+'_terms']]
    f.bandwidth_multiplier=1.
    return f


def reading(archive,prefix,x):
    f=feature_from_archive(archive,prefix)
    return f.values(x)@archive[prefix+'_coefficient']+archive[prefix+'_center']
