# #####################################################################
#
# PROJETS VOLUMES FINIS : Systèmes hyperboliques
#    Sujet n°8 :
#       Étude du système de la dynamique des gaz isentropiques
#       par relaxation en coordonnées Lagrangiennes
#
#   Codage de la solution exacte en suivant le calcul
#   développé dans le rapport PDF + solution approchée 
#   en utilisant le schéma de Rusanov, vu en cours (sv)
#
# ---------------------------------------------------------------------
#
#   À définir par l'utilisateur :
#       - définition loi d'état p(V) :      ligne  44
#       - valeur de a :                     ligne 327
#       - valeur états gauches/droites :    ligne 336
#
# #####################################################################

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::: Partie commune aux deux solutions : l. 21 --> 73              :::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# =====================================================================
# ## Importation des modules
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,5)

# =====================================================================
# ## Fonctions lambda utiles
# =====================================================================
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
# ---   pour la solution approchée       ---
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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::: Partie pour la solution exacte : l. 74 --> 175                :::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#=========================== Solution Exacte ==========================
#    ENTRÉES 
#       xx : espace
#        t : temps
#    tau_L : volume spécifique dans l'état gauche
#      u_L : vitesse du fluide dans l'état gauche
#      v_L : volume de relaxation dans l'état gauche   
#    tau_R : volume spécifique dans l'état droit
#      u_R : vitesse du fluide dans l'état droit
#      v_R : volume de relaxation dans l'état droit 
#        a : constante positive de l'énoncé
#    SORTIES 
#      tau : volume spécifique général
#        u : vitesse du fluide général
#        v : volume de relaxation général 
#======================================================================
def sol_exacte(xx, t, tau_L, u_L, v_L, tau_R, u_R, v_R, a):

    # -----------------------------------------
    # ---     Initialisation des sorties    ---
    # -----------------------------------------
    tau = np.zeros_like(xx)
    u = np.zeros_like(xx)  
    v = np.zeros_like(xx)   

    # -----------------------------------------
    # ---          Vitesses des DDC         ---
    # -----------------------------------------
    vit_1 = -a
    vit_2 = 0
    vit_3 = a

    # -----------------------------------------
    # ---  Affectation sorties avec masques ---
    # -----------------------------------------
    mask_L = (xx <= vit_1*t)
    tau[mask_L] = tau_L 
    u[mask_L] = u_L 
    v[mask_L] = v_L
    mask_1 = (vit_1*t < xx) & (xx < vit_2*t) 
    tau[mask_1] = tau_1
    u[mask_1] = u_1
    v[mask_1] = v_1
    mask_2 = (vit_2*t < xx) & (xx < vit_3*t) 
    tau[mask_2] = tau_2
    u[mask_2] = u_2
    v[mask_2] = v_2
    mask_R = (vit_3*t <= xx)
    tau[mask_R] = tau_R 
    u[mask_R] = u_R
    v[mask_R] = v_R

    # -----------------------------------------
    # ---         Retour des sorties        ---
    # -----------------------------------------               
    return tau, u, v

#=============== Solution dans l'état intermédiaire 1 =================
#    ENTRÉES 
#    tau_L : volume spécifique dans l'état gauche
#      u_L : vitesse du fluide dans l'état gauche
#      v_L : volume de relaxation dans l'état gauche   
#    tau_R : volume spécifique dans l'état droit
#      u_R : vitesse du fluide dans l'état droit
#      v_R : volume de relaxation dans l'état droit 
#        a : constante positive de l'énoncé
#    SORTIES 
#     tau_1 : volume spécifique dans l'état Y_1
#       u_1 : vitesse du fluide dans l'état Y_1
#       v_1 : volume de relaxation dans l'état Y_1 
#======================================================================
def y1(tau_L, u_L, v_L, tau_R, u_R, v_R, a):
    tau_1 = v_L - (pi(tau_L,v_L)+pi(tau_R,v_R)+a*(u_L-u_R)-2*p(v_L))/(2*a**2)
    u_1 = u_L + 0.5 * (pi(tau_L,v_L)/a + pi(tau_R,v_R)/a + u_L - u_R)
    v_1 = v_L

    return tau_1, u_1, v_1

#=============== Solution dans l'état intermédiaire 2 =================
#    ENTRÉES 
#    tau_L : volume spécifique dans l'état gauche
#      u_L : vitesse du fluide dans l'état gauche
#      v_L : volume de relaxation dans l'état gauche   
#    tau_R : volume spécifique dans l'état droit
#      u_R : vitesse du fluide dans l'état droit
#      v_R : volume de relaxation dans l'état droit 
#        a : constante positive de l'énoncé
#    SORTIES 
#     tau_2 : volume spécifique dans l'état Y_2
#       u_2 : vitesse du fluide dans l'état Y_2
#       v_2 : volume de relaxation dans l'état Y_2 
#======================================================================
def y2(tau_L, u_L, v_L, tau_R, u_R, v_R, a):
    tau_2 = v_R - (pi(tau_L,v_L)+pi(tau_R,v_R)+a*(u_L-u_R)-2*p(v_R))/(2*a**2)
    u_2 = u_R + 0.5 * (pi(tau_L,v_L)/a - pi(tau_R,v_R)/a + u_L - u_R)
    v_2 = v_R

    return tau_2, u_2, v_2

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::: Partie pour la solution approchée : l.176 --> 317             :::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ===================================================================================
#
# APPROXIMATION : SCHEMA DE RUSANOV
# _________________________________
#
# La fonction G d'un schéma VF s'écrit, dans le cas du schéma de Rusanov
# et pour un état gauche a et un état droit b :
#
#             F(a) + F(b)                                     b - a
#   G(a,b) = -------------  - max[lambda_k(a),lambda_k(b)] * -------
#                  2           k                                2
#
#     --> On décompose cette formule en trois équations
#
# ===================================================================================

#=============== Définition du flux G 1ère equation ==================
#    ENTRÉES 
#      w0L : 1ère composante du vecteur état gauche
#      w1L : 2ème composante du vecteur état gauche
#      w2L : 3ème composante du vecteur état gauche   
#      w0R : 1ère composante du vecteur état droit
#      w1R : 2ème composante du vecteur état droit
#      w2R : 3ème composante du vecteur état droit 
#    SORTIE 
#     Première composante du vecteur solution approchée 
#======================================================================
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

#=============== Définition du flux G 2ème equation ==================
#    ENTRÉES 
#      w0L : 1ère composante du vecteur état gauche
#      w1L : 2ème composante du vecteur état gauche
#      w2L : 3ème composante du vecteur état gauche   
#      w0R : 1ère composante du vecteur état droit
#      w1R : 2ème composante du vecteur état droit
#      w2R : 3ème composante du vecteur état droit 
#    SORTIE 
#     Deuxième composante du vecteur solution approchée 
#======================================================================
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

#=============== Définition du flux G 3ème equation ==================
#    ENTRÉES 
#      w0L : 1ère composante du vecteur état gauche
#      w1L : 2ème composante du vecteur état gauche
#      w2L : 3ème composante du vecteur état gauche   
#      w0R : 1ère composante du vecteur état droit
#      w1R : 2ème composante du vecteur état droit
#      w2R : 3ème composante du vecteur état droit 
#    SORTIE 
#     Troisième composante du vecteur solution approchée 
#======================================================================
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

#=============== Schéma VF avec 3 equations : Rusanov =================
#    ENTRÉES 
#     ttau : volume spécifique au temps t
#       uu : vitesse du fluide au temps t
#       vv : volume de relaxation au temps t  
#       dx : pas d'espace
#       dt : pas de temps
#    SORTIES 
#  WW0_new : 1ère composante du vecteur solution approchée à t+1 = tau
#  WW1_new : 2ème composante du vecteur solution approchée à t+1 = u
#  WW2_new : 3ème composante du vecteur solution approchée à t+1 = v
#======================================================================
def sol_approx(ttau,uu,vv,dx,dt):

    # ------------------------------------------
    # ---    Création vecteurs de travail    ---
    # ------------------------------------------
    WW0, WW1, WW2 = ttau, uu, vv
    WW0c, WW1c, WW2c= np.copy(WW0), np.copy(WW1), np.copy(WW2)
    WW0p, WW1p, WW2p= v_plus(WW0), v_plus(WW1), v_plus(WW2) 
    WW0m, WW1m, WW2m= v_moins(WW0), v_moins(WW1), v_moins(WW2) 

    # ------------------------------------------
    # ---         Première équation          ---
    # ------------------------------------------
    flux0p = g0(WW0c,WW1c,WW2c,WW0p,WW1p,WW2p)
    flux0m = np.roll(flux0p,1)
    flux0m[0] = g0(WW0m[0],WW1m[0],WW2m[0],WW0c[0],WW1c[0],WW2c[0])
    WW0_new = WW0c - (dt/dx) * (  flux0p - flux0m  )
    
    # ------------------------------------------
    # ---         Deuxième équation          ---
    # ------------------------------------------
    flux1p = g1(WW0c,WW1c,WW2c,WW0p,WW1p,WW2p)
    flux1m = np.roll(flux1p,1)
    flux1m[0] = g1(WW0m[0],WW1m[0],WW2m[0],WW0c[0],WW1c[0],WW2c[0])
    WW1_new = WW1c - (dt/dx) * (  flux1p - flux1m  )

    # ------------------------------------------
    # ---         Troisième équation         ---
    # ------------------------------------------
    flux2p = g2(WW0c,WW1c,WW2c,WW0p,WW1p,WW2p)
    flux2m = np.roll(flux1p,1)
    flux2m[0] = g2(WW0m[0],WW1m[0],WW2m[0],WW0c[0],WW1c[0],WW2c[0])
    WW2_new = WW2c - (dt/dx) * (  flux2p - flux2m  )

    # ------------------------------------------
    # ---         Troisième équation         ---
    # ------------------------------------------
    return WW0_new , WW1_new, WW2_new

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::: Partie MAIN commune aux deux solutions : l.318 --> fin        :::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# ------------------------------------------
# ---    Définition des paramètres       ---
# ------------------------------------------
x_L, x_0, x_R = -10, 0, 10      # Bornes du domaine [x_L,x_R] et point de discontinuité x_0
Nx = 400                        # Nb de mailles
a  = 10                         # A DEFINIR !
cfl = 0.5                       # condition cfl

xx = np.linspace(x_L,x_R,Nx)    # Espace
dx = xx[1]-xx[0]                # Pas d'espace

# ------------------------------------------
# ---   Données Problème de Riemann      ---
# ------------------------------------------
CI = [(10,-3,8),(8,-3,3)]       # vL et ensuite vR (A DEFINIR !)
tau_L, u_L, v_L = CI[0]         # Etat gauche
tau_R, u_R, v_R = CI[1]         # Etat droit

# ---------------------------------------------
# --- Calculs états intermédiaires (sol ex) ---
# ---------------------------------------------
tau_1, u_1, v_1 = y1(tau_L, u_L, v_L, tau_R, u_R, v_R, a)
tau_2, u_2, v_2 = y2(tau_L, u_L, v_L, tau_R, u_R, v_R, a)

# ---------------------------------------------
# ---  Initialisation paramètres (sol app)  ---
# ---------------------------------------------
t = 0                           # Temps initial
dt = 0.999*dx/max(-a,0,a )      # max inutile puisque a>0 mais on reprend la même structure du code vu en cours
ttau = np.zeros_like(xx)        # tau (approché) initial
uu = np.zeros_like(xx)          # u (approché) initial
vv = np.zeros_like(xx)          # v (approché) initial

# Travail avec les masques puis initialisation des tau, u, v initiaux
mask_L = xx<0   
mask_R = xx>=0
ttau[mask_L] , uu[mask_L], vv[mask_L] = tau_L , u_L, v_L
ttau[mask_R] , uu[mask_R], vv[mask_R] = tau_R , u_R, v_R

# Initialisation figure
fig,(ax0,ax1) = plt.subplots(nrows=1,ncols=2)

# ------------------------------------------
# ---           Marche en temps          ---
# ------------------------------------------
while t<1.0:

    # Update temps
    t += dt

    # Update solution approchée
    ttau,uu,vv = sol_approx(ttau,uu,vv,dx,dt)

    # Update solution exacte
    tau_ex, u_ex, v_ex = sol_exacte(xx, t,\
                                    tau_L, u_L, v_L,\
                                    tau_R, u_R, v_R,\
                                     a)


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