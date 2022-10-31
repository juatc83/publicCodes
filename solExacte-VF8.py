# #####################################################################
#
# PROJETS VOLUMES FINIS : Systèmes hyperboliques
#    Sujet n°8 :
#       Étude du système de la dynamique des gaz isentropiques
#       par relaxation en coordonnées Lagrangiennes
#
#   Codage de la solution exacte en suivant le calcul
#   développé dans le rapport PDF.
#
# ---------------------------------------------------------------------
#
#   À définir par l'utilisateur :
#       - définition loi d'état p(V) :      ligne  33
#       - valeur de a :                     ligne 163
#       - valeur états gauches/droites :    ligne 169
#
# #####################################################################

# =====================================================================
# ## Importation des modules
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,5)

# =====================================================================
# ## Fonctions lambda utiles
# =====================================================================
# ------------------------------------------
# ---     « Vraie » loi d'état du gaz    ---
# ------------------------------------------
p = lambda v : v # A definir !

# ------------------------------------------
# ---   « Nouvelle » loi d'état du gaz   ---
# ------------------------------------------
pi = lambda tau,v : p(v)+a**2*(v-tau)

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

#=====================================
# ## AFFICHAGE (prevu pour interact)
#===================================== 
def inter(fig,ax1,ax2,xx,t,tau_L, u_L, v_L, tau_R, u_R, v_R, \
            tau_1, u_1, v_1, tau_2, u_2, v_2, tau_ex, u_ex, v_ex):
    ax1.set_ylim([min([u_L,u_R,u_1,u_2])-0.1,max([u_L,u_R,u_1,u_2])+0.1])
    ax2.set_ylim([min([tau_L,tau_R,tau_1,tau_2])-0.1,max([tau_L,tau_R,tau_1,tau_2])+0.1])
    ax1.plot(xx,u_ex,'-o')
    ax2.plot(xx,tau_ex,'-o')
    ax1.set_title(f"${u_L=:g}$, ${u_1=:g}$, ${u_2=:g}$, ${u_R=:g}$")
    ax2.set_title(f"${tau_L=:g}$, ${tau_1=:g}$, ${tau_2=:g}$, ${tau_R=:g}$")
    ax1.grid()
    ax2.grid()

#===============================
#             MAIN
#===============================
# Bornes du domaine [x_L,x_R] et point de discontinuité x_0
x_L, x_0, x_R = -10, 0, 10

 # Nb de mailles
Nx = 100

# A definir !
a  = 10

# Espace
xx = np.linspace(x_L,x_R,Nx) 

# Données Pbs de Riemann
CI = [(10,-3,8),(8,-3,3)] # vL et ensuite vR (il faut mettre quoi ?)
tau_L, u_L, v_L = CI[0]   # état gauche
tau_R, u_R, v_R = CI[1]   # état droit

# Calculs états intermédiaires
tau_1, u_1, v_1 = y1(tau_L, u_L, v_L, tau_R, u_R, v_R, a)
tau_2, u_2, v_2 = y2(tau_L, u_L, v_L, tau_R, u_R, v_R, a)

# Boucle en temps et affichage
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
for t in np.linspace(0,1.5,11):    
    tau_ex, u_ex, v_ex = sol_exacte(xx, t, tau_L, u_L, v_L, tau_R, u_R, v_R, a)
    inter(fig,ax1,ax2,xx,t,tau_L, u_L, v_L, tau_R, u_R, v_R, \
            tau_1, u_1, v_1, tau_2, u_2, v_2, tau_ex, u_ex, v_ex)
    plt.pause(0.5)
    ax1.cla()
    ax2.cla()