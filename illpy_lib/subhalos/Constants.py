"""
"""
import collections


class SNAPSHOT():
    IDS                  = "ParticleIDs"
    POS                  = "Coordinates"

    POT                  = "Potential"
    DENS                 = "Denity"
    SFR                  = "StarFormationRate"
    VEL                  = "Velocities"
    EINT                 = "InternalEnergy"
    MASS                 = "Masses"

    HSML                 = "SmoothingLength"      # 2x max triangle radius

    SUBF_HSML            = "SubfindHsml"
    SUBF_VDISP           = "SubfindVelDisp"

    '''
    BH_MASS                  = "BH_Mass"
    BH_HSML                  = "BH_Hsml"
    BH_MDOT                  = "BH_Mdot"
    STELLAR_PHOTOS           = "GFM_StellarPhotometrics"
    FORM_TIME                = "GFM_StellarFormationTime"

    PARENT                   = "ParentID"
    NPART                    = "npart_loaded"
    '''


class SUBHALO():

    POS                  = "SubhaloPos"
    COM                  = "SubhaloCM"
    SUBH_PARENT          = "SubhaloParent"
    MOST_BOUND           = "SubhaloIDMostbound"
    NUM_GROUP            = "SubhaloGrNr"

    NUM_PARTS            = "SubhaloLen"
    NUM_PARTS_TYPE       = "SubhaloLenType"

    PHOTOS               = "SubhaloStellarPhotometrics"

    METZ_GAS             = "SubhaloGasMetallicity"
    METZ_STAR            = "SubhaloStarMetallicity"

    BH_MASS              = "SubhaloBHMass"
    BH_MDOT              = "SubhaloBHMdot"

    VEL                  = "SubhaloVel"
    VMAX                 = "SubhaloVmax"
    VDISP                = "SubhaloVelDisp"
    VMAX                 = "SubhaloVmax"
    SPIN                 = "SubhaloSpin"

    SFR                  = "SubhaloSFR"
    SFR_HALF_RAD         = "SubhaloSFRinHalfRad"
    SFR_IN_MAX_RAD       = "SubhaloSFRinMaxRad"
    SFR_IN_RAD           = "SubhaloSFRinRad"

    MASS                 = "SubhaloMass"
    MASS_TYPE            = "SubhaloMassType"
    MASS_HALF_RAD        = "SubhaloMassInHalfRad"
    MASS_IN_HALF_RAD_TYPE = "SubhaloMassInHalfRadType"
    MASS_IN_MAX_RAD      = "SubhaloMassInMaxRad"
    MASS_IN_MAX_RAD_TYPE = "SubhaloMassInMaxRadType"
    MASS_IN_RAD          = "SubhaloMassInRad"
    MASS_IN_RAD_TYPE     = "SubhaloMassInRadType"
    MASS_WIND            = "SubhaloWindMass"

    RAD_HALF_MASS        = "SubhaloHalfmassRad"
    RAD_HALF_MASS_TYPE   = "SubhaloHalfmassRadType"
    RAD_VMAX             = "SubhaloVmaxRad"
    RAD_PHOTOS           = "SubhaloStellarPhotometricsRad"

    # SubhaloGasMetallicityMaxRad
    # SubhaloStarMetallicityMaxRad
    # SubhaloStellarPhotometricsMassInRad
    # SubhaloStarMetallicityHalfRad
    # SubhaloGasMetallicitySfrWeighted
    # SubhaloGasMetallicityHalfRad
    # SubhaloStarMetallicity
    # SubhaloGasMetallicitySfr

    @staticmethod
    def PROPERTIES():
        return [getattr(SUBHALO, it) for it in vars(SUBHALO)
                if not it.startswith('_') and not isinstance(getattr(SUBHALO, it), collections.Callable)]
