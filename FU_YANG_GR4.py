import gurobipy as grb
import pandas as pd
import numpy as np
import copy
import math


# Extraction des donnees (c_names, d_, v_)
pf = pd.read_excel("villes.xls")
c_names = [c for c in pf["Ville"]]
n = len(c_names)
d_ = np.zeros((n, n))
for i in range(n):
    d_[:, i] = pf[c_names[i]]
for i in range(n):
    for j in range(i + 1, n):
        d_[i, j] = d_[j, i]
v_ = [v for i, v in enumerate(pf["Population"])]


# Programme lineaire 1.2
def pl_12(secteurs, alpha):
    
    """ list[str] * float -> solution pl
        hypothese : len(secteurs) > 0
    
        Retourne la solution optimale du programme lineaire 1.2. """
    
    k = len(secteurs)
    dn = {r : secteurs[r] for r in range(k)}
    nbcont = n + k
    nbvar = n * k
    a = np.zeros((nbcont, nbvar))    # x11 x21 x31 ... x12 x22 x32 ... 
    
    for i in range(n):
        for j in range(k):
            a[i, j * n + i] = 1
    for j in range(k):
        for i in range(n):
            a[n + j, j * n + i] = v_[i]
    
    charge = ((1 + alpha) / k) * sum(v_)
    b = []
    
    for i in range(n):
        b.append(1)
    for j in range(k):
        b.append(charge)
    
    c = [d_[i, c_names.index(dn[j])] for j in range(k) for i in range(n)]
    m = grb.Model("pl12")
    x = []
    
    for j in range(k):
        for i in range(n):
            x.append(m.addVar(vtype = grb.GRB.BINARY, name ="x%d%d" % (i, j)))
            
    m.update()
    obj = grb.LinExpr()
    obj = 0
    
    for l in range(nbvar):
        obj += c[l] * x[l]
    
    m.setObjective(obj, grb.GRB.MINIMIZE)
    
    for i in range(n):
        m.addConstr(grb.quicksum(a[i, l] * x[l] for l in range(nbvar)) >= b[i], "Contrainte%d" % i)
    for j in range(k):
        m.addConstr(grb.quicksum(a[n + j, l] * x[l] for l in range(nbvar)) <= b[n + j], "Contrainte%d" % (n + j))

    m.optimize()
    res = dict()
    j = 0
    ens = set()
    
    for l in range(nbvar):
        if x[l].x == 1:
            ens.add(c_names[l % n])
        if l != 0 and (l + 1) % n == 0:
            res[dn[j]] = ens
            j += 1
            ens = set()
    
    return res, m.objVal
    
        
# Programme lineaire 2.1
def pl_21(k, alpha):
    
    """ int * float -> solution pl
        hypothese : k > 0
    
        Retourne la solution optimale du programme lineaire 2.1. """
        
    charge = ((1 + alpha) / k) * sum(v_)
    m = grb.Model("pl21")
    x = []
    y = []
    
    for j in range(n):
        for i in range(n):
            x.append(m.addVar(vtype = grb.GRB.BINARY, name ="x%d%d" % (i, j)))
            
    for j in range(n):
        y.append(m.addVar(vtype = grb.GRB.BINARY, name ="y%d" % j))
        
    m.update()
    obj = grb.LinExpr()
    obj = 0
    
    for j in range(n):
        for i in range(n):
            obj += d_[i, j] * x[j * n + i]
            
    m.setObjective(obj, grb.GRB.MINIMIZE)
    
    for i in range(n):
        m.addConstr(grb.quicksum(x[j * n + i] for j in range(n)) >= 1, "Contrainte%d" % i)
    for j in range(n):
        m.addConstr(grb.quicksum(v_[i] * x[j * n + i] for i in range(n)) <= charge, "Contrainte%d" % (n + j))
    
    m.addConstr(grb.quicksum(y[j] for j in range(n)) == k, "Contrainte%d" % (n + n))
    
    for j in range(n):
        m.addConstr(grb.quicksum(x[j * n + i] for i in range(n)) <= y[j] * n, "Contrainte%d" % (n + n + 1 + j))
    
    m.optimize()
    res = dict()
    
    for j in range(n):
        if y[j].x == 1:
            ens = set()

            for i in range(n):
                if x[j * n + i].x == 1:
                    ens.add(c_names[i])
            
            res[c_names[j]] = ens
    
    return res, m.objVal
        

# Programme lineaire 2.2
def pl_22(k, alpha):
    
    """ int * float -> solution pl
        hypothese : k > 0
    
        Retourne la solution optimale du programme lineaire 2.1. """
        
    charge = ((1 + alpha) / k) * sum(v_)
    m = grb.Model("pl22")
    x = []
    y = []
    
    for j in range(n):
        for i in range(n):
            x.append(m.addVar(vtype = grb.GRB.BINARY, name ="x%d%d" % (i, j)))
            
    for j in range(n):
        y.append(m.addVar(vtype = grb.GRB.BINARY, name ="y%d" % j))
        
    r = m.addVar(vtype = grb.GRB.CONTINUOUS, name ="r")
        
    m.update()
    obj = grb.LinExpr()
    obj = r
            
    m.setObjective(obj, grb.GRB.MINIMIZE)
    
    for i in range(n):
        m.addConstr(grb.quicksum(x[j * n + i] for j in range(n)) >= 1, "Contrainte%d" % i)
    for j in range(n):
        m.addConstr(grb.quicksum(v_[i] * x[j * n + i] for i in range(n)) <= charge, "Contrainte%d" % (n + j))
    
    m.addConstr(grb.quicksum(y[j] for j in range(n)) == k, "Contrainte%d" % (n + n))
    
    for j in range(n):
        m.addConstr(grb.quicksum(x[j * n + i] for i in range(n)) <= y[j] * n, "Contrainte%d" % (n + n + 1 + j))
    
    for i in range(n):
        m.addConstr(r >= grb.quicksum(d_[i, j] * x[j * n + i] for j in range(n)), "Contrainte%d" % (n + n + n + 1 + i))
        
    
    m.optimize()
    res = dict()
    
    for j in range(n):
        if y[j].x == 1:
            ens = set()

            for i in range(n):
                if x[j * n + i].x == 1:
                    ens.add(c_names[i])
            
            res[c_names[j]] = ens
    
    return res, m.objVal
        

def find_maxd(res):
    
    """ dict{str : set{str}} -> float
    
        Retourne la distance maximale d'une ville a son secteur. """

    d = 0
    nom1 = ''
    nom2 = ''
    
    for c in res:
        for v in res[c]:
            i = c_names.index(v)
            j = c_names.index(c)
            
            if d_[i, j] > d:
                d = d_[i, j]
                nom1 = c_names[i]
                nom2 = c_names[j]
    
    return d, nom2, nom1
    

# Probleme du flot maximum a cout minimum
def create_unites(l):
    
    """ List[int] -> dict{int : str}
    
        Retourne les indicages des noeuds du graphe 3.1. """
        
    res = dict()
        
    for i in range(1, 2 * len(l) + 1):
        res[i] = c_names[l[(i - 1) % len(l)]]
        
    return res


def create_graphe(n_unites):
    
    """ int -> dict{int : set{int}}
    
        Retourne le graphe du probleme 3.1. """
    
    res = {0 : {i for i in range(1, n_unites + 1)}}
    
    for i in range(1, n_unites + 1):
        res[i] = {j for j in range(n_unites + 1, 2 * n_unites + 1)}
        
    for i in range(n_unites + 1, 2 * n_unites + 1):
        res[i] = {2 * n_unites + 1}
        
    res[2 * n_unites + 1] = set()
    
    return res


def create_capacite(p, c1, c2):
    
    """ list[int] * int * int -> dict{tuple(int, int) : int}
    
        Retourne les capacites du graphe 3.1. """
    
    res = dict()
    
    for i in range(1, len(p) + 1):
        res[(0, i)] = p[i - 1]
        
    for i in range(len(p) + 1, 2 * len(p) + 1):
        res[(i, 11)] = c2
    
    for i in range(1, len(p) + 1):
        for j in range(len(p) + 1, 2 * len(p) + 1):
            res[(i, j)] = c1
    
    return res


def create_cout(l):
    
    """ list[int] -> dict{tuple(int, int) : int} 
    
        Retourne les couts du graphe 3.1. """
    
    res = dict()
    
    for i in range(1, len(l) + 1):
        res[(0, i)] = 0
        
    for i in range(len(l) + 1, 2 * len(l) + 1):
        res[(i, 11)] = 0
        
    for i in range(1, len(l) + 1):
        for j in range(len(l) + 1, 2 * len(l) + 1):
            res[(i, j)] = d_[l[i - 1], l[j - 6]]
    
    return res


def find_chemin_aug(ge):
    
    """ dict{int : set{int}} -> bool
    
        Retourne True si on peut encore injecter du flot dans ge (graphe d'ecart du graphe 3.1), False sinon. """
    
    return len(ge[0]) != 0
    


def peres(ge, noeud):
    
    """ dict{int : set{int}} * int -> set{int} 
    
        Retourne les peres de noeud dans ge. """
    
    res = set()
    
    for u in ge:
        if noeud in ge[u]:
            res.add(u)
            
    return res


def plus_court_chemin(ge, co):
    
    """ dict{int : set{int}} * dict{tuple(int, int) : int} -> list[int]
    
        Retourne le plus court chemin dans ge en terme de cout co par l'algorithme de Bellman-Ford ainsi que son cout. """
    
    t_ge = len(ge)
    ite =[[0, -1]]
    
    for i in range(t_ge - 1):
        ite.append([float('inf'), None])
    
    
    for i in range(t_ge):
        for noeud in range(t_ge):
            d_noeud = ite[noeud][0]
            p_noeud = ite[noeud][1]
            pr = peres(ge, noeud)
            
            
            for u in pr:
                if ite[u][0] + co[(u, noeud)] < d_noeud:
                    d_noeud = ite[u][0] + co[(u, noeud)]
                    p_noeud = u
                    
            ite[noeud][0] = d_noeud
            ite[noeud][1] = p_noeud
        
    che = [t_ge - 1]
    
    while ite[che[0]][1] != -1:
        che = [ite[che[0]][1]] + che
        
    return che, ite[-1][0]


def goulot(che, cap):
    
    """ list[int] * dict{tuple(int, int) : int} -> int 
    
        Retourne le goulot d'etranglement de che en terme de capacite cap. """
    
    b = cap[(che[0], che[1])]
    
    for i in range(1, len(che) - 1):
        if cap[(che[i], che[i + 1])] < b :
            b = cap[(che[i], che[i + 1])]
    
    return b
        

def modif_ge(ge, cap, co, che, b):
    
    """ dict{int : set{int}} * dict{tuple(int, int) : int} * dict{tuple(int, int) : int} * list[int] * int -> graphe d'ecart, capacite, cout
    
        Retourne le graphe d'ecart, capacite et cout du graphe. """
    
    resge = copy.deepcopy(ge)
    rescap = copy.deepcopy(cap)
    resco = copy.deepcopy(co)
    
    for i in range(len(che) - 1):
        p = che[i]
        q = che[i + 1]
        rescap[(p, q)] = rescap[(p, q)] - b
        
        if rescap[(p, q)] == 0:
            del rescap[(p, q)]
            del resco[(p, q)]
            resge[p].remove(q)
        
        if (q, p) in rescap:
            rescap[(q, p)] = rescap[(q, p)] + b
        else:
            rescap[(q, p)] = b
            resco[(q, p)] = - co[(p, q)]
            resge[q] = resge[q] | {p}
    
    return resge, rescap, resco 


def flot_max_cout_min(gr, cap, co, unites):
    
    """ dict{int : set{int}} * dict{tuple(int, int) : int} * dict{tuple(int, int) : int} * dict{int : str} -> int, etapes, affecttions
    
        Retourne le cout min de l'algorithme du flot max cout min, les etapes de l'algorithme. """
    
    resgr = copy.deepcopy(gr)
    rescap = copy.deepcopy(cap)
    resco = copy.deepcopy(co)
    resc = 0
    etapes = []
    
    while find_chemin_aug(resgr):
        che, c = plus_court_chemin(resgr, resco)
        b = goulot(che, rescap)
        resche = [unites[che[i]] for i in range(1 ,len(che) - 1)]
        etapes.append((resche, b))
        resc += c * b 
        resgr, rescap, resco = modif_ge(resgr, rescap, resco, che, b)
        
    
    
    return resc, etapes
            

def rep_32(l, p, change = False):
    
    """ List[int] * List[int] -> flot max coup min
        hypothese : len(l) == len(p)
    
        Retourne le resultat de la question 3.2. """
        
    unites = create_unites(l)
    graphe = create_graphe(len(l))
    cs = sum(p) / len(p)
    if cs - math.floor(cs) != 0:
        cs = math.ceil(cs)
    if change:
        capacite = create_capacite(p, float('inf'), cs)
    else:
        capacite = create_capacite(p, float('inf'), 100)
    cout = create_cout(l)
    
    return flot_max_cout_min(graphe, capacite, cout, unites)


# fonction resultat du projet
def resultat():
    
    """ """
    
    choix = int(input('Veuillez entrer le numero pour obtenir les resultats :\n1 : question 1.2\n2 : question 2.1\n3 : question 2.2\n4 : question 3.2\n\n'))
    typ = int(input('Veuillez entrer le numero pour selectionner le type des reponses :\n1 : reponses obtenues dans le rapport\n2 : reponses selon vos propres parametres\n\n'))
    
    if choix == 1:
        if typ == 1:
            secteurs1 = [c_names[0], c_names[1], c_names[2]]
            secteurs2 = [c_names[0], c_names[1], c_names[2], c_names[3]]
            alpha1 = 0.1
            alpha2 = 0.2
            r1 = pl_12(secteurs1, alpha1)
            r2 = pl_12(secteurs1, alpha2)
            r3 = pl_12(secteurs2, alpha1)
            r4 = pl_12(secteurs2, alpha2)
            print('\n\n', r1, '\n')
            print(r2, '\n')
            print(r3, '\n')
            print(r4, '\n')
        if typ == 2:
            nb = int(input('Veuillez entrer le nombre de d unites a implanter :\n\n'))
            unites = []
            l = []
            dnames = {i : c_names[i] for i in range(len(c_names))}
            
            for i in range(nb):
                print('\nvos unites :', unites)
                print('indices :', l, '\n')
                print(dnames)
                u = int(input('Veuillez choisir une ville ci-dessus pour implanter une unite :\n\n'))
                unites.append(c_names[u])
                l.append(u)
            
            print('\nvos unites :', unites)
            print('indices :', l)
            alpha = float(input('Veuillez enter la valeur d alapha : \n\nalpha = '))
            
            return pl_12(unites, alpha)
            
    if choix == 2:
        if typ == 1:
            k1 = 3
            k2 = 4
            k3 = 5
            alpha1 = 0.1
            alpha2 = 0.2
            alpha3 = 0.3
            r1 = pl_21(k1, alpha1)
            r2 = pl_21(k1, alpha3)
            r3 = pl_21(k2, alpha1)
            r4 = pl_21(k2, alpha3)
            r5 = pl_21(k3, alpha1)
            r6 = pl_21(k3, alpha2)
            print('\n\n', r1, '\n')
            print(r2, '\n')
            print(r3, '\n')
            print(r4, '\n')
            print(r5, '\n')
            print(r6, '\n')
        if typ == 2:
            nb = int(input('Veuillez entrer le nombre de d unites a implanter :\n\n'))
            alpha = float(input('Veuillez enter la valeur d alapha : \n\nalpha = '))
            
            return pl_21(nb, alpha)
            
    if choix == 3:
        if typ == 1:
            k1 = 3
            k2 = 4
            k3 = 5
            alpha1 = 0.1
            alpha2 = 0.2
            alpha3 = 0.4
            r1 = pl_22(k1, alpha1)
            r2 = pl_22(k1, alpha3)
            r3 = pl_22(k2, alpha1)
            r4 = pl_22(k2, alpha2)
            r5 = pl_22(k3, alpha1)
            r6 = pl_22(k3, alpha2)
            print('\n\n', r1, '\n')
            print(r2, '\n')
            print(r3, '\n')
            print(r4, '\n')
            print(r5, '\n')
            print(r6, '\n')
        if typ == 2:
            nb = int(input('Veuillez entrer le nombre de d unites a implanter :\n\n'))
            alpha = float(input('Veuillez enter la valeur d alapha : \n\nalpha = '))
            
            return pl_22(nb, alpha)
        
    if choix == 4:
        if typ == 1:
            l = [0, 1, 8, 13, 14]
            p = [250, 200, 0, 20, 0]
            
            return rep_32(l, p)
        if typ == 2:
            unites = []
            l = []
            p = []
            rp = 500
            dnames = {i : c_names[i] for i in range(len(c_names))}
            
            for i in range(5):
                print('\nvos unites :', unites)
                print('indices :', l, '\n')
                print(dnames)
                u = int(input('Veuillez choisir une ville ci-dessus pour implanter une unite :\n\n'))
                print('\nVous pouvez encore repartir', rp, 'patients')
                pi = int(input('Veuillez entrer son p initial : \n\np = '))
                unites.append(c_names[u])
                l.append(u)
                p.append(pi)
                rp -= pi
            
            print('\nvos unites :', unites)
            print('indices :', l)
            
            if rp < 0:
                return
            
            return rep_32(l, p)
            
            

            
            
    
            
        
            
        
        
        
        
    
        
    
        
        
        
        
        
        


