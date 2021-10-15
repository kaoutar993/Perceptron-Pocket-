import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

class Perceptron:
    
 
        
    def fit(self, X, y, n_iter=100):
        
        n_samples = X.shape[0] #nbre des lignes de la matrice
        n_features = X.shape[1]#nbre des colonnes de la matrice
       
        #Initialisation du vecteur des poids par des zéros 
        self.weights = np.zeros((n_features+1,))
        
        # Ajouter une colonne des 1 et la concaténer avec X
        X = np.concatenate([X, np.ones((n_samples,1))], axis=1)
        
        for i in range(n_iter):
            for j in range(n_samples):
                
                #La condition de la mal classification
                if y[j]*np.dot(self.weights, X[j, :]) <= 0:
                    
                    #Ajouter la nouvelle valeur du poids au vecteur self.weights 
                    self.weights += y[j]*X[j, :]

    def predict(self, X):
        #Vérification si la phase d'apprentissage est déjà faite
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return 
        #la phase de prédiction
        n_samples = X.shape[0]
        X = np.concatenate([X, np.ones((n_samples,1))], axis=1)
        
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)
        return y
        
        
    
    def score(self, X, y):
        pred_y = self.predict(X)
        return np.mean(y == pred_y) 
    
    
############### Création de données ###############
X, y = make_classification(
        n_features=2, #represente le nombre de paramétres (colonnes de X)
        n_classes=2,  #les classifications de y
        n_samples=200, #nombre des données (les lignes de X )
        n_redundant=0, # les combinaisons linéaire entre les données  
        n_clusters_per_class=1, #le regroupement des données dans chaquue classe 
        random_state=55, # aide à generer les même données aleatoire
        class_sep=2 # pour separer les données lineairement 
    )
##################################################

#permuter les 0 par des -1 dans le vecteur y
y_=np.vectorize(lambda val: 1 if val > 0 else -1)(y)

# diviser les données en 2 parties ( train et test )        
X_train, X_test, y_train, y_test = train_test_split( X, y_, test_size=0.2, random_state=6)

p=Perceptron() #creation d'un objet p de type perceptron
p.fit(X_train, y_train,100) #executer la phase d'apprentissage des poids 

x1=np.min(X[:, 0]) - 1 #definir un min de la droite 
x2=np.max(X[:, 0]) + 1 #definir un max de la droite 

#definir l'equation de notre droite séparatrice 
xx=np.linspace(x1,x2,1000) 
#droite de l'aquation w2+w1*x2+w0*x1=0
L = [(-p.weights[2]-p.weights[0]*elem)/p.weights[1] for elem in xx]

### tracer les données et la droite 
fig, axs = plt.subplots(1,2, figsize=(11, 5))
#tracer les donneés d'apprentissage
axs[0].scatter(X_train[:,0],X_train[:,1],c=y_train)
#tracer les données de test 
axs[1].scatter(X_test[:,0],X_test[:,1],c=y_test)
#tracer la droite pour l'apprentissage et aussi pour le test 
axs[0].plot(xx,L)
axs[1].plot(xx,L)
#### ajout des titres pour chaque graphe
axs[0].set_title("les données d'apprentissage" )
axs[1].set_title("les données de test")
##ajouter le score de chaque graphe 
axs[0].set_xlabel("Score: "+str(p.score(X_train, y_train)*100)+"%")
axs[1].set_xlabel("Score: "+str(p.score(X_test, y_test)*100)+"%")
