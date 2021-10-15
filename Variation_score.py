
#######################variation score#####################"
def create_graph(max_samples) :
    #liste des itérations
    L = [i*10 for i in range(max_samples)]
    p = Pocket() 
    L_scores_train = []
    for elem in L:
        p.fit(X_train, y_train,elem)
        #calcul pour chaque itération de la liste L
        L_scores_train.append(p.score(X_train, y_train))    
    #Visualisation des résultats
    plt.plot(L, L_scores_train)

create_graph(10)


def create_graph2(max_samples) :
    #la liste des tailles des données utilisées dans l'apprentissage
    L = [i*10+1 for i in range(max_samples)]
    p = Pocket()
    L_scores_train = []
    for elem in L:
        p.fit(X_train[:elem], y_train[:elem],100)
        #calcul pour chaque taille de données de la liste L
        L_scores_train.append(p.score(X_train[:elem], y_train[:elem]))
    #Visualisation des résultats 
    plt.plot(L, L_scores_train,color='orange')
    
create_graph2(50) 
