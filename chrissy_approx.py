# # Solution Exacte Saint-Venant

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,8)


# ===================================================================================
# INTRO ET FONCTIONS UTILES
# ===================================================================================

# --------------------------------------------------------
# Shift des vecteurs avec conditions de Neumann
# --------------------------------------------------------
v_plus  = lambda vv : np.concatenate( (vv[1:], vv[-1:]) ) 
v_moins = lambda vv : np.concatenate( (vv[:1], vv[:-1]) )

# ------------------------------------------
# ---     « Vraie » loi d'état du gaz    ---
# ------------------------------------------
p = lambda v : v # A definir !

# ------------------------------------------
# ---   « Nouvelle » loi d'état du gaz   ---
# ------------------------------------------
pi = lambda tau,v : p(v)+a**2*(v-tau)

# ------------------------------------------
# ---   Définition des vecteurs utiles   ---
# ------------------------------------------
# WW = [tau,u,v]        Vecteur solution (Y dans l'énoncé)
# QQ = [u,pi(tau,V),0]  Vecteur F(W)

# On exprime les composantes de Q en fonction de celles
# de W
Q0 = lambda WW : WW[1] 
Q1 = lambda WW : pi(WW[0],WW[2]) 
Q2 = lambda WW : 0

# ------------------------------------------
# ---   Définition des valeurs propres   ---
# ------------------------------------------
lambda_0 = lambda h,v : -a      # 1ère DDC
lambda_1 = lambda h,v :  0      # 2ème DDC
lambda_2 = lambda h,v :  a      # 3ème DDC

# On remarque que ce ne sont ques des constantes :
# les calculs seront plus simples par la suite !

# ===================================================================================
# ## APPROX
# ===================================================================================

# Petite présentation du schéma de Rusanov
#
#
#

# ------------------------------------------
# --- Définition du flux G 1ère equation ---
# ------------------------------------------
def g0(w0L, w1L, w2L, w0R, w1R, w2R):
    l0L = -a ## np.max(abs(lambda_0()))) = constante
    l0R = -a ## np.max(abs(lambda_0()))  = constante
    l1L =  0 ## np.max(abs(lambda_1()))) = constante
    l1R =  0 ## np.max(abs(lambda_1()))  = constante
    l2L =  a ## np.max(abs(lambda_2()))) = constante
    l2R =  a ## np.max(abs(lambda_2()))  = constante
    # il y a d'autres valeurs qui sont toujours égales à -a, 0, a
    A = max( [l0L,l0R,l1L,l1R,l2L,l2R] ) # = a

    # On ne calcule que suivant Q0 puisque c'est la 1ère equation
    return ( Q0([w0L, w1L, w2L])+Q0([w0R, w1R, w2R]) - (w0R-w0L)*A )/2 

# ------------------------------------------
# --- Définition du flux G 2ème equation ---
# ------------------------------------------
def g1(w0L, w1L, w2L, w0R, w1R, w2R):
    l0L = -a ## np.max(abs(lambda_0()))) = constante
    l0R = -a ## np.max(abs(lambda_0()))  = constante
    l1L =  0 ## np.max(abs(lambda_1()))) = constante
    l1R =  0 ## np.max(abs(lambda_1()))  = constante
    l2L =  a ## np.max(abs(lambda_2()))) = constante
    l2R =  a ## np.max(abs(lambda_2()))  = constante
    # il y a d'autres valeurs qui sont toujours égales à -a, 0, a
    A = max( [l0L,l0R,l1L,l1R,l2L,l2R] ) # = a

    # On ne calcule que suivant Q1 puisque c'est la 2ère equation
    return ( Q1([w0L, w1L, w2L])+Q1([w0R, w1R, w2R]) - (w1R-w1L)*A )/2 

# ------------------------------------------
# --- Définition du flux G 3ème equation ---
# ------------------------------------------
def g2(w0L, w1L, w2L, w0R, w1R, w2R):
    l0L = -a ## np.max(abs(lambda_0()))) = constante
    l0R = -a ## np.max(abs(lambda_0()))  = constante
    l1L =  0 ## np.max(abs(lambda_1()))) = constante
    l1R =  0 ## np.max(abs(lambda_1()))  = constante
    l2L =  a ## np.max(abs(lambda_2()))) = constante
    l2R =  a ## np.max(abs(lambda_2()))  = constante
    # il y a d'autres valeurs qui sont toujours égales à -a, 0, a
    A = max( [l0L,l0R,l1L,l1R,l2L,l2R] ) # = a

    # On ne calcule que suivant Q2 puisque c'est la 3ère equation
    return ( Q2([w0L, w1L, w2L])+Q2([w0R, w1R, w2R]) - (w2R-w2L)*A )/2 

# ------------------------------------------
# ---  Définition schéma VF 3 equations  ---
# ------------------------------------------
def sol_approx(ttau,uu,vv,dx,dt):

    # Vecteurs de travail
    WW0, WW1, WW2 = ttau, uu, vv
    WW0c, WW1c, WW2c= np.copy(WW0), np.copy(WW1), np.copy(WW2)
    WW0p, WW1p, WW2p= v_plus(WW0), v_plus(WW1), v_plus(WW2) 
    WW0m, WW1m, WW2m= v_moins(WW0), v_moins(WW1), v_moins(WW2) 

    # Première équation
    flux0p = g0(WW0c,WW1c,WW2c,WW0p,WW1p,WW2p)
    flux0m = np.roll(flux0p,1)
    flux0m[0] = g0(WW0m[0],WW1m[0],WW2m[0],WW0c[0],WW1c[0],WW2c[0])
    WW0_new = WW0c - (dt/dx) * (  flux0p - flux0m  )
    
    # Deuxième équation
    flux1p = g1(WW0c,WW1c,WW2c,WW0p,WW1p,WW2p)
    flux1m = np.roll(flux1p,1)
    flux1m[0] = g1(WW0m[0],WW1m[0],WW2m[0],WW0c[0],WW1c[0],WW2c[0])
    WW1_new = WW1c - (dt/dx) * (  flux1p - flux1m  )

    # Troisième équation
    flux2p = g2(WW0c,WW1c,WW2c,WW0p,WW1p,WW2p)
    flux2m = np.roll(flux1p,1)
    flux2m[0] = g2(WW0m[0],WW1m[0],WW2m[0],WW0c[0],WW1c[0],WW2c[0])
    WW2_new = WW2c - (dt/dx) * (  flux2p - flux2m  )

    return WW0_new , WW1_new, WW2_new

# ===================================================================================
# PROGRAMME MAIN
# ===================================================================================

# Bornes du domaine [x_L,x_R] et point de discontinuité x_0
x_L, x_0, x_R = -10, 0, 10
Nx = 400    # nb de mailles
a  = 10     # A definir !
cfl = 0.5   # condition cfl

# Espace
xx = np.linspace(x_L,x_R,Nx) 
dx = xx[1]-xx[0]

# Données Pbs de Riemann
CI = [(10,-3,8),(8,-3,3)] # vL et ensuite vR (il faut mettre quoi ?)
tau_L, u_L, v_L = CI[0]   # état gauche
tau_R, u_R, v_R = CI[1]   # état droit

# Initialisation
t = 0
dt = 0.999*dx/max(-a,0,a ) # max inutile puisque a>0 mais on reprend la même structure du code
ttau = np.zeros_like(xx)
uu = np.zeros_like(xx)
vv = np.zeros_like(xx)
mask_L = xx<0
mask_R = xx>=0
ttau[mask_L] , uu[mask_L], vv[mask_L] = tau_L , u_L, v_L
ttau[mask_R] , uu[mask_R], vv[mask_R] = tau_R , u_R, v_R

# Initialisation figure
fig,(ax0,ax1) = plt.subplots(nrows=1,ncols=2)

# Marche en temps
while t<1.0:
    # dt = 0.999*dx/max(-a,0,a ) # update inutile puisque dt dépend uniquement de constantes !

    # Update temps
    t += dt
    
    # Update solution approchée
    ttau,uu,vv = sol_approx(ttau,uu,vv,dx,dt)
    
    # Affichage figure
    ax0.cla()
    # ax0.set_title(f"${u_L=:g}$, ${uu=:g}$, ${u_R=:g}$") # Affichage à revoir
    ax0.plot(xx,uu,'-o')
    ax0.grid()    
    ax1.cla()    
    # ax1.set_title(f"${tau_L=:g}$, ${ttau=:g}$, ${tau_R=:g}$") # Affichage à revoir
    ax1.plot(xx,vv,'-o')
    ax1.grid()
    
    # On se repose un peu
    plt.pause(0.1)
