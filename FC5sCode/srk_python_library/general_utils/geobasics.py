import numpy as np

def calc_q (s1_eff,s3_eff):
    return s1_eff-s3_eff

def calc_p_eff (s1_eff,s3_eff):
    return (s1_eff + 2 * s3_eff)/3

def calc_su (q):
    return q/2

def calc_eta (q,p_eff):
    return q/p_eff

def calc_phi_mob (eta):
    return np.degrees(np.arcsin( 3 * eta / (6 + eta) ))

def calc_s (s1_eff,s3_eff):
    return (s1_eff+s3_eff)/2

def calc_t (s1_eff,s3_eff):
    return (s1_eff-s3_eff)/2

def calc_G (tau,gamma,steps):
    gamma_interp = np.linspace(gamma.min(),gamma.max(),num=steps)
    gamma_float = np.array(gamma.values.tolist())*1
    tau_float = np.array(tau.values.tolist())*1
    tau_interp = np.interp(gamma_interp,gamma_float,tau_float)
    G_interp = np.gradient(tau_interp,gamma_interp,edge_order=2)
    G = np.interp(gamma_float,gamma_interp,G_interp)
    return G
    # return (tau.diff()/gamma.diff())

def normalize (values,value0=None):
    if value0 is None:
        return values/values.values[0]
    else:
        return values/value0

def calc_CSR (tau,s1_eff):
    return tau/s1_eff

def calc_Eoed(Eoed0,c,phi,sig3,m,pref): # Following the HSS formulation
    '''
    Oedometric modulus formulation from HSS, function of Eoed0, phi, c, sig3, m and pref
    
    Eoed0 [kPa]: Reference Eoed, corresponding to a σ'3 = pref
    phi [°]: Friction angle
    c [kPa]: Effective cohession
    sig3 [kPa]: Third effective principal stress. Compression is positive
    m [-]: Janbu coefficient
    pref [kPa]: Reference pressure
    '''
    
    # To skip c<0 values
    if isinstance(c,np.ndarray) or isinstance(c,list):
        for i,ci in enumerate(c): # Skip c<0
            if ci<0:
                c[i]=0
    else:
        if c < 0:
            c = 0
    phi = np.radians(phi)
    return Eoed0*((c*np.cos(phi)+sig3*np.sin(phi))/(c*np.cos(phi)+pref*np.sin(phi)))**m
    
def calc_Tv(uave):
    '''
    Dimensionless time factor

    uave: Degree of one-dimensional consolidation
    '''
    if 0 < uave < 0.6:
        return np.pi/4 * uave**2
    elif 1 > uave >= 0.6:
        return (-0.9333*np.log10(1-uave)-0.085)
    else:
        return print("Uave has to be in the range [0,1)")
    
def calc_Uave(Tvv):
    '''
    Instantaneous degree of consolidation
    
    Tvv: Time in days
    '''
    
    if Tvv < calc_Tv(0.6):
        return np.sqrt(Tvv*4/np.pi)
    elif Tvv >= calc_Tv(0.6):
        return (1-10**((Tvv+0.085)/(-0.933)))

def calc_Tu(uave,Ht,Cv):
    '''
    One-dimensional consolidation time as a function of degree of consolidation

    uave: (float, np.ndarray) Degree of one-dimensional consolidation [0,1)
    Ht: (float, np.ndarray) Stratum height in meters
    Cv: (float, np.ndarray) Consolidation coefficient in cm²/s
    '''
    Cv = Cv * 0.01**2 * (60*60*24) # Cambio de unidades (a metros y días)
    return (calc_Tv(uave)*(Ht/2)**2) / Cv

def calc_Uu(t,Ht,Cv):
    '''
    Degree of one-dimensional consolidation as a function of time

    t: (float, np.ndarray) Time in days
    Ht: (float, np.ndarray) Stratum height in meters
    Cv: (float, np.ndarray) Consolidation coefficient in cm²/s
    '''
    Cv = Cv * 0.01**2 * (60*60*24) # Cambio de unidades (a metros y días)
    return calc_Uave((t*Cv)/(Ht/2)**2)

def calc_Ur(t,D,Cv,d=0.15):
    '''
    Degree of radial consolidation as a function of time

    t: (float, np.ndarray) Time in days
    D: (float, np.ndarray) Distance between drains in meters
    Cv: (float, np.ndarray) Consolidation coefficient in cm²/s
    d: (float, np.ndarray) Diameter of drains in meters
    '''
    Cv = Cv * 0.01**2 * (60*60*24) # Cambio de unidades (a metros y días)
    return 1-np.e**((-8*(Cv*t/D**2))/(np.log(D/d)-3/4))

def calc_Tur(ur,D,Cv,d=0.15):
    '''
    Time as a function of the degree of radial consolidation

    ur: (float, np.ndarray) Degree of radial consolidation [0,1)
    D: (float, np.ndarray) Distance between drains in meters
    Cv: (float, np.ndarray) Consolidation coefficient in cm²/s
    d: (float, np.ndarray) Diameter of drains in meters 
    '''
    Cv = Cv * 0.01**2 * (60*60*24) # Cambio de unidades (a metros y días)
    return (-np.log(1 - ur) * (np.log(D/d) - 3/4)) / (8*Cv/D**2)

def calc_Uru(t,Ht,D,Cv,d=0.15):
    '''
    Degree of radial-vertical consolidation as a function of time

    t: (float, np.ndarray) (float, np.ndarray) Time in days
    Ht: (float, np.ndarray) (float, np.ndarray) Stratum height in meters
    D: (float, np.ndarray) (float, np.ndarray) Distance between drains in meters
    Cv: (float, np.ndarray) (float, np.ndarray) Consolidation coefficient in cm²/s
    d: (float, np.ndarray) (float, np.ndarray) Diameter of drains in meters
    '''
    return 1-((1-calc_Uu(t,Ht,Cv))*(1-calc_Ur(t,D,Cv,d)))

def calc_dSigz(q,r,z,load_type,
               label_circular = 'Circular',label_strip = 'Strip',label_rectangular='Rectangular',label_none = 'None'): # Following USACE Settlement Analysis. Table C-1. https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf
    '''
    Δσ calculation following USACE Settlement Analysis. Table C-1. https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf
    
    Parameters:
        q [kPa]: (float, np.ndarray) Load magnitude
        r [m]: (float, np.ndarray) Radius of circular load, half of width of strip load or array of two dimensions of rectangular load (Half of width and Long)
        z [m]: (float, np.ndarray) Depth where Δσq is calculated
        load_type: (str) Must be label_circular, label_strip, label_rectangular or label_none
        label_circular: (str) Label used for circular load
        label_strip: (str) Label used for strip load
        label_rectangular: (str) Label used for rectangular load
        label_none: (str) Label used for none load
        
    Return:
        Δσ_q [kPa]: (float, np.ndarray) Increase of effective stresses due to the load q, at a z depth
    '''
    if load_type == label_circular:
        S = np.sqrt(r**2+z**2)
        return q*r**2*(S**2+2*z**2)/(2*S**4) # Δσq
    
    elif load_type == label_strip: # Strip load. Following USACE Settlement Analysis. Table C-1. https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf
        beta = np.arctan(-r/z)
        alpha = np.arctan(r/z)-beta
        return q/np.pi*(alpha+np.sin(alpha)*np.cos(alpha+2*beta)) # Δσq
    
    elif load_type == label_rectangular: # Corner load x4. Following USACE Settlement Analysis. Table C-1. https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf
        a = r[0]
        b = r[1]
        A=np.sqrt(a**2+z**2)
        B=np.sqrt(b**2+z**2)
        C=np.sqrt(a**2+b**2+z**2)
        return 4*q/(2*np.pi)*(np.arctan((a*b)/(z*C))+(a*b*z)/C*(1/A**2+1/B**2)) # Δσq
    
    elif load_type == label_none:
        return 0
    
    else:
        return print('None valid type of load was introduced. The valids are "{label_circular}", "{label_strip}", "{label_rectangular}" or "{label_none}"'.format(label_circular=label_circular,
                                                                                                                                                                label_strip=label_strip,
                                                                                                                                                                label_rectangular=label_rectangular,
                                                                                                                                                                label_none=label_none))