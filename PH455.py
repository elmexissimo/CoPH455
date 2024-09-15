#*************************************************************
# Support functions for PH455/957 
# 
# You will probably need to use some of the functions in this file
# but you do not need to understand the source code. 
#
# Elmar Haller, Glasgow 2022
#*************************************************************
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

#***************************************************
# Print version of this file
#
# Usage:  version()
#***************************************************
def version():
    ver  = "0.3"
    date = "27/09/2023"    
    print("Version: %s \nDate: %s"% (ver,date))
    



#***************************************************
# For a Gaussian beam calculate Rayleigh range from q
#***************************************************
def calcBeamZ0_Q(q):
    #realQ = np.real(q)
    imagQ = np.imag(q)
    
    return imagQ

# from W0
def calcBeamZ0_W0(W0, lam):
    z0 = np.pi * W0**2 / lam
    return z0



#***************************************************
# For a Gaussian beam calculate q from W and R
#***************************************************
def calcBeamQ(R,W,lam):    
    # avoid division by zero
    if R==0:
        R = 1e-16
    if W==0:
        W = 1e-16
        
    invQ = 1/R - 1j * lam / (np.pi* W**2)
    return 1/invQ

#***************************************************
# For a Gaussian beam calculate W and R
#***************************************************
# from z0, z
def calcBeamR(z, z0):
    R = z * (1+ (z0/z)**2 )
    return R

# from W0, z0, z
def calcBeamW(z, W0, z0):
    W = W0* np.sqrt( 1+ (z/z0)**2 )
    return W

# from z0
def calcBeamW0_Z0(z0, lam):
    W0 = np.sqrt( lam * z0/np.pi )
    return W0


#***************************************************
# For a Gaussian beam calculate R and W from q
#***************************************************
def calcBeamRW(q,lam):
    if q==0:                     # prevent division by zero with small number
        q = 1e-16
    Oq = 1/q
    realOq = np.real(Oq)
    imagOq = np.abs(np.imag(Oq))
    
    if realOq == 0:              # prevent division by zero with small number
        realOq = 1e-16
    if imagOq == 0:
        imagOq = 1e-16
        
    R = 1/realOq
    W = np.sqrt( lam / (np.pi*imagOq ) )
    return R,W

    
#***************************************************
# ABCD matrix multiplication for Gaussa beams with q paramter
#***************************************************
def multiBeamMat(M,q1):
    
    # check matrix dimensions
    if len(M) != 2:
        print("mulitBeamMat: You must use a 2x2 matrix.")
        return           
    else:
        if len(M[0]) != 2:
            print("mulitBeamMat: You must use a 2x2 matrix.")
            return           

    q2 = (M[0][0]*q1 + M[0][1]) / (M[1][0]*q1 + M[1][1])
    return q2


#***************************************************
#  Create a simple plot for ray propagation
#  - zVect and rayVect are nparrays or normal arrays
#  - zVect and rayVect must have the same length
#  - rayVect contains rays of format: [y,theta]
#    e.g. rayVect = [ [y0,theta0], [y1,theta1], [y3,theta3]]
#  - assume all in units of mm
#***************************************************
def showBeam( zVect, bVect, lam):
    
    useString = "Usage: showBeam(zVect,bVect,lambda)\n"
    
    if len(zVect) != len(bVect):
        print("The arrays zVect and bVect must have the same length.")
        print(useString)
        return   
    
    def drawWavefront(z,R,W):    
        r = np.abs(R)
        # get angles for circle
        #print(W)
        #print(r)
        theta = np.abs(np.arcsin(W/r))
        angleVect = np.linspace(-theta, theta, 501 )        
        dz = r * np.cos(theta)
        
        # get points on circle
        xs = np.sign(R)*(r * np.cos(angleVect) - dz) + z
        ys = r * np.sin(angleVect)    
        # plot lines
        plt.plot(xs,ys, 'b-')
        plt.plot(z, W, 'ro')
        plt.plot(z, -W, 'ro')    

    #*****************************************
    # plot all rays but last
    # function has still a few unnecessary calculations
    #*****************************************
    # plot first beam
    q1 = bVect[0]                      # get two q values
    R1,W1 = calcBeamRW(q1,lam)         # get R,W values
    z1 = zVect[0]
    drawWavefront(z1,R1,W1)            # plot wavefronts
    
    for i1 in range(1,len(bVect)):
        # get two q values
        q2 = bVect[i1]
        #R1,W1 = calcBeamRW(q1,lam)
        R2,W2 = calcBeamRW(q2,lam)
        #z1 = zVect[i1]
        z2 = zVect[i1]
        
        # plot wavefronts
        drawWavefront(z2,R2,W2)
        # draw connecting lines
        plt.plot( [z1,z2], [W1,W2], 'r--')
        plt.plot( [z1,z2], [-W1,-W2], 'r--')
        
        # shuffle numbers 
        q1 = q2
        R1 = R2
        W1 = W2
        z1 = z2
        
    # plot optical axis and position of optical elements
    dz = plt.xlim()
    plt.plot([dz[0],dz[1]], [0,0], c='0.55', zorder=0)
    # used: c - color, zorder - bottom layer    
    plt.xlim(dz[0],dz[1])
       
    plt.xlabel('z-direction (mm)')
    plt.ylabel('x-position (mm)')


#***************************************************
# Show phasor diagram
#***************************************************
def showPhasor(sumU, phiVect, M):

    # store the real and imaginary compoenents of the calculated sumU values
    URealVect = [0]
    UImagVect = [0]

    fig, ax = plt.subplots()
        
    for idx in range(M):
        # calculate U for phi, M values
        Us = sumU(idx,phiVect)
        URealVect.append(np.real(Us))
        UImagVect.append(np.imag(Us))
        
    # plotting stuff for MVect
    for idx in range(M):
        plt.plot(URealVect[:(idx+1)],UImagVect[:(idx+1)],'b-')           # plot blue line
        plt.plot(URealVect[idx],UImagVect[idx],'ro')   # plot red dot for last value        

    # get max values for axis limits 
    RealMax = np.max(URealVect)
    RealMax = RealMax + RealMax * 0.1
    ImagMax = np.max(UImagVect)
    ImagMax = ImagMax + ImagMax * 0.1
    ImagMin = np.min(UImagVect)
    ImagMin = ImagMin - ImagMax * 0.1
    plt.xlim([-RealMax, RealMax])               # fix axis limits
    plt.ylim([ImagMin, ImagMax])

    # title with phi value
    plt.title( 'M={:.1f}, phi={:.2f}pi'.format(M,phiVect/np.pi) )
    plt.xlabel('real(U)')                       # labels  
    plt.ylabel('imag(U)')


#***************************************************
#  Create a simple plot for ray propagation
#  - zVect and rayVect are nparrays or normal arrays
#  - zVect and rayVect must have the same length
#  - rayVect contains rays of format: [y,theta]
#    e.g. rayVect = [ [y0,theta0], [y1,theta1], [y3,theta3]]
#  - assume all in units of mm
#***************************************************
def showRay( zVect, rayVect):
    
    useString = "Usage: showRay(zVect,rayVect)\n"
    
    #if not isinstance(zVect, np.ndarray) or not isinstance(rayVect, np.ndarray):
    #    print("The arguments zVect and rayVect must be numpy arrays.\n")
    #    print(useString)
    #    return   
    if len(zVect) != len(rayVect):
        print("The arrays zVect and rayVect must have the same length.")
        print(useString)
        return   
    
    plt.figure()
    
    #plot all rays but last
    for i1 in range(len(rayVect)-1):
        ray0 = rayVect[i1]
        ray1 = rayVect[i1+1]
        # plot circle for ray parameters
        plt.plot(zVect[i1], ray0[0], 'ro')
        # plot line to next ray parameters
        plt.plot( [zVect[i1],zVect[i1+1]], [ray0[0],ray1[0]], 'r--')
        
    # plot last ray with angle
    ray0 = rayVect[-1]
    plt.plot(zVect[-1], ray0[0], 'ro')
    # plot angle indicator, at least 0.1 length
    # or 10% of full length
    ddz = 0.1*np.max( [zVect[-1]-zVect[0], 10] ) 
    plt.plot( [zVect[-1], zVect[-1]+ddz] , [ray0[0],ray0[0]+ddz*ray0[1]], 'b--',zorder=0)

    # plot first ray angle
    ray0 = rayVect[0]
    plt.plot( [zVect[0]-ddz, zVect[0]] , [ray0[0]-ddz*ray0[1],ray0[0]], 'b--', zorder=0)

    # plot optical axis and position of optical elements
    dz = plt.xlim()
    plt.plot([dz[0],dz[1]], [0,0], c='0.55', zorder=0)
    # used: c - color, zorder - bottom layer    
    plt.xlim(dz[0],dz[1])
    
    # plot lines for optical elements
    dy = plt.ylim()
    mdy = max(np.abs(dy))
    plt.ylim(-mdy,mdy)
    for i1 in range(len(zVect)):
        plt.plot([zVect[i1],zVect[i1]], [-mdy,mdy], c='0.55', zorder=0)
        
    plt.xlabel('z-direction (mm)')
    plt.ylabel('x-position (mm)')
    
    

#***************************************************
#  A couple of support functions for the course
#***************************************************

def calcFreeSpectralRange(d,n=1):
    fsr = 299792458/(2*n*d)
    return fsr

# calculate a Lorentzian function
def Lorentzian(x, x0, b):
    L = 1/np.pi * b / ((x-x0)**2 + (b)**2 ) 
    return L
 
    
# calculate a Gaussian function
# b is the 1/sqrt(e) width
def Gaussian(x, x0, b):
    #c = 2*b/2.35482
    G = 1/(b*np.sqrt(2*np.pi)) * np.exp( -0.5*(x-x0)**2 /b**2 ) 
    return G