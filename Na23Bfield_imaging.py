'''
Refactored code by Hongyin Liu Dec.23, 2023

Compute the imaging frequency of Na23 at a magnetic field
Used for calculating high field imaging frequency for molecule formation

'''

import numpy as np

try:
    import numericalunits as units
except ImportError:
    print("numericalunits module not found, using simple unit system")
    class Units:
        pass
    units = Units()
    units.MHz = 1. 
    units.G = 1.

muB = 1.399624604*units.MHz/units.G #Bohr magneton / h


###----------------------------------------------------------------------------###
#-----------------------------------Classes------------------------------------###
###----------------------------------------------------------------------------###

def _getN(J):
    #Find the number of states and check J
    if not(2*J == int(2*J)):
        raise ValueError("J must be half of an integer")
    return int(2*J+1)
    
def AngIndex(J,m):
    """
    Finds the index of m in the standard basis for spin J
    """
    ind_f = m+J #formula to go from m to index, but potentially gives floats
    ind_i = int(ind_f) #make it an int
    assert ind_f == ind_i #make sure we were given valid inputs
    return ind_i


#Angular momentum operators
def AngLadder(J,sign):
    """
    Finds the ladder operator for spin J
    sign = 1: raising
    sign =-1: lowering
    
    assumes the basis |J m_J> order starting with m_J = -J
    """    
    if sign != 1 and sign != -1:
        raise ValueError("sign must be +- 1")
    
    N = _getN(J)
    Mat = np.zeros((N,N))
    mRange = np.arange(-J,J) if sign == 1 else np.arange(-J+1,J+1)
    for m in mRange:
        Mat[AngIndex(J,m+sign),AngIndex(J,m)] = np.sqrt(J*(J+1)-m*(m+sign))
    return Mat

def AngZ(J):
    """
    Operator matrix for z component of angular momentum in standard basis
    """
    N = _getN(J)
    Mat = np.zeros((N,N))
    for m in np.arange(-J,J+1):
        Mat[AngIndex(J,m),AngIndex(J,m)] = m
    return Mat

def AngX(J):
    return 0.5*(AngLadder(J,1) + AngLadder(J,-1))

def AngY(J):
    return -1j*0.5*(AngLadder(J,1) - AngLadder(J,-1))    

def Dot(I,J):
    """
    dot product operator for spins with magnitudes I and J
    """
    return np.kron(AngX(I), AngX(J)) + np.kron(AngY(I), AngY(J))+np.kron(AngZ(I), AngZ(J))


class AtomIJ(object):
    """    
    Models an atom as a nuclear spin I and electron spin J
    Assumes we have chosen n,L
    """
    def __init__(self,name,L_label,I,J,gI, gJ, A, B=0, muBref=muB):
        self.name = name
        self.L_label = L_label
        self.I = I
        self.J = J
        self.gJ = gJ #electron Lande g factor
        self.gI = gI #nuclear g factor
        self.A = A #hyperfine constant A
        self.B = B #hyperfine constant B
        self.muB = muBref #Bohr magneton
        self._init()
        
    def _init(self):
        #initialize the Hamiltonian
        I = self.I; J=self.J
        NI = _getN(I)
        NJ = _getN(J)
        eyeI = np.eye(NI); eyeJ = np.eye(NJ)
        eye = np.kron(eyeI, eyeJ)
        
        IdotJ = np.kron(AngX(I), AngX(J)) + np.kron(AngY(I), AngY(J))+np.kron(AngZ(I), AngZ(J))
        if J != .5 and I != .5 and self.B != 0:
            Bterm = self.B*(3*np.dot(IdotJ,IdotJ)+ 1.5*IdotJ  - eye*I*(I+1)*J*(J+1))/(2*I*(2*I-1)*J*(2*J-1))
        else:
            Bterm = 0*IdotJ
            
        self.HF = self.A*IdotJ + Bterm #hyperfine Hamiltonian / h
        
        mJ = np.kron(eyeI, AngZ(J))
        mI = np.kron(AngZ(I), eyeJ)
        self.mu = self.muB*(self.gJ*mJ + self.gI*mI) #total magnetic moment operator / h
        
        if J == int(J):
            Jlabel = "$_{%d}$" % int(J)
        else:
            Jlabel = "$_{%d/%d}$" % (int(2*J), 2)
        self.term = self.L_label + Jlabel
        
    def eig(self,Bz):
        """
        find the energies at field Bz
        returns vals, vecs
        """
        H = self.HF + self.mu*Bz 
        vals, vecs = np.linalg.eig(H)
        #sort:
        vecs =  np.array([x for (y,x) in sorted(zip(vals,np.transpose(vecs)), key=lambda pair: pair[0].real)])     
        vals = np.array(sorted(vals))
        
        return vals.real, vecs
       

def landeG(L,L1,L2,g1,g2):
    def term(g,a,b):
        return g*(L*(L+1.)-b*(b+1.)+a*(a+1.))/(2.*L*(L+1.))
    return term(g1,L1,L2) + term(g2,L2,L1)
 




###------------------------------------------------------------------------------###
#-----------------------------------Functions------------------------------------###
#-----------------------------------Important------------------------------------###
###------------------------------------------------------------------------------###

def findDiffFreq(atom_gnd, atom_exc, ind_g, ind_e, Bz):
    """
    Find the transition frequency between ground and excited states
    Gives the answer relative to the difference between the lines' centers of mass
    i.e. don't make any correction for overall energy offset
    """
    vals_g, vecs_g = atom_gnd.eig(Bz)
    vals_g = sorted(vals_g)
    
    vals_e, vecs_e = atom_exc.eig(Bz)
    vals_e = sorted(vals_e)
    return vals_e[ind_e] - vals_g[ind_g]


def imgProgram(gnd, exc, ref_g, ref_e, ref_offset_freq, ind_g, ind_e, Bz, filepath=None, Bmax = 1000*units.G, Bref=0,doplot=True,divide=1):
    
    nucyc = findDiffFreq(gnd, exc, ind_g = ref_g, ind_e = ref_e, Bz=Bref)
    print("Transition frequency with Bz = 0G: " + str(nucyc))
    ref = nucyc + ref_offset_freq
    
    dE = findDiffFreq(gnd, exc, ind_g, ind_e, Bz) - ref
    msg = "Image Frequency at %g G:" % (Bz/units.G)
    if divide !=1:
        msg += str(divide) + "x"
    msg += "%g MHz at the HF imaging double-pass AOM" % (dE/units.MHz/divide)
    print(msg)



###------------------------------------------------------------------------------------------###
#---------------------------------Constants and Variables------------------------------------###
###------------------------------------------------------------------------------------------###

#Sodium Na values (steck)
I = 1.5                     #Nuclear Spin
S = 0.5                     #Electron Spin
gL = 0.99997613             #Electron Orbital g-factor
gS = 2.0023193043622        #Electron Spin g-factor
gI = -0.00080461080         #Nuclear g-factor

#Ground State S_1/2
L_g=0; J_g = 0.5
gJ_g = landeG(J_g, L_g, S, gL, gS) 
A_gnd = 885.81306440*units.MHz       #Magnetic Dipole Constant
B_gnd = 0                   #Electric Quadrupole Constant
Na_gnd = AtomIJ("$^{23}$Na","S",I, J_g, gI, gJ_g, A_gnd, B_gnd)

#Excited State P_1/2
L_e1=1; J_e1 = 0.5
gJ_e1 = landeG(J_e1, L_e1, S, gL, gS) 
A_D1 = 94.44*units.MHz
B_D1 = 0
Na_exc1 = AtomIJ("$^{23}$Na","P",I, J_e1, gI, gJ_e1, A_D1, B_D1)

#Excited State P_3/2
L_e2=1; J_e2 = 1.5
gJ_e2 = landeG(J_e2, L_e2, S, gL, gS) 
A_D2 = 18.534*units.MHz
B_D2 = 2.724*units.MHz
Na_exc2 = AtomIJ("$^{23}$Na","P",I, J_e2, gI, gJ_e2, A_D2, B_D2)




###------------------------------------------------------------------------------------------###
#-----------------------------------Execution of programs------------------------------------###
###------------------------------------------------------------------------------------------###
    
import os
folder = os.getcwd()
filepath = os.path.join(folder,"Na_23_imgfreq.csv")

'''
For molecule formation, we are using the transition F = 1-> F' = 0, and the Na state |1> = |F = 1, m_F = 1>
Note the excited state hyperfine levels are not resolvable within a natural linewidth 61 MHz
'''

#reference transition
ref_g = 0 #F = 1 state in the S1/2 ground state
ref_e = 0 #F = 0 state in the P3/2 excited state

ind_g = 0 #lowest energy ground state at high field |F = 1, m_F = 1>
ind_e = 0 #lowest energy excited state at high field

ref_offset = 765*units.MHz #detuning from reference 2-> 3' (MOT) transition before double-pass HF imaging AOM


Bref = 0*units.G #reference is defined at zero field
Bmax=750*units.G
#Bz = 745*units.G #For molecule formation, we are imaging at 745 Gauss
Bz = float(input("What is the Magnetic field B we are imaging at?  \n (Please put a number only. The standard for molecule formation is 745G) \n"))

#This program outputs the imaging frequency
imgProgram(Na_gnd, Na_exc2, ref_g, ref_e, ref_offset, ind_g, ind_e, Bz, filepath, Bmax, Bref,doplot=0, divide=2)
