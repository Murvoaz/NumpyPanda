import numpy as np
import pandas as pd

print("     Tableau d'entiers:")
x = np.array([1, 4, 2, 5, 3])
print(x)
print("     Tableau de flots:  ")
x2 = np.array([3.14, 4, 2, 3])
print(x2)
x3 = np.array([1, 2, 3, 4], dtype='float32')
print(x3)
print("     Une liste de listes est transformée en un tableau multi-dimensionnel :")
x4 = np.array([range(i, i + 3) for i in [2, 4, 6]])
print(x4)
print("   Matrice identité : ")
x5 = np.eye(3)
print(x5)

print(" Fonction sur tableau :")
np.random.seed(0)
x6 = np.random.randint(10, size=6)
print(x6)
print("nombre de dimensions de x6: ", x6.ndim)
print("forme de x6: ", x6.shape)
print("taille de x6: ", x6.size)
print("type de x6: ", x6.dtype)
# Pour accéder au premier élément
print("premier élément de x6: ", x6[0])

# Pour accéder au dernier élément
print("dernier élément de x6: ", x6[-1])

# On peut aussi modifier les valeurs
x6[0] = "1000"
print("1er élément en 1000 de x6: ",x6)

# Attention au type
x6[0] = 3.14
print("1er élément en en float de x6:",x6)

print("5 premier élément de x6:",x6[:5])  # Les cinq premiers éléments

print("à partir du 5e élément de x6:",x6[5:])  # Les éléments à partir de l'index 5

print("un élément sur deux de x6:",x6[::2])  # Un élément sur deux

print("x6 à l'envers:",x6[::-1])

x7 = np.random.randint(10, size=(3, 4))  # Tableau de dimension 3 par 4
print("premier element de la premiere ligne:",x7[0,0])

print("x7 tableau à 3 lignes:",x7)

print("premiere ligne de x7:",x7[0,:])

print("concatenation de tableau",np.concatenate([x2, x3]))

grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

print("x2 + grid",np.vstack([x2[:3], grid]))

# Il y a tout d'abord des opération mathématiques simples
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # Division avec arrondi
x = np.where(x > 0.5)
print("> 0.5",x)

x = [-2, -1, 1, 2]

print(x)
print("La valeur absolue: ", np.abs(x))
print("Exponentielle: ", np.exp(x))
print("Logarithme: ", np.log(np.abs(x)))

L = np.random.random(100)
print(np.sum(L))

M = np.random.random((3, 4))
print(M)
# Notez la syntax variable.fonction au lieu de 
# np.fonction(variable). Les deux sont possibles si
# la variable est un tableau Numpy.
print("La somme de tous les éléments de M: ", M.sum())
print("Les sommes des colonnes de M: ", M.sum(axis=0))

a = np.array([0, 1, 2])
print(a)
b = np.array([5, 5, 5])
print(b)
print(a + b)
M = np.ones((3, 3))
print("M vaut: \n", M)
print("M+a vaut: \n", M+a)
a = np.arange(3)
print(a)
# La ligne suivante crée une matrice de taille 3x1
# avec trois lignes et une colonne.
b = np.arange(3)[:, np.newaxis]
print(b)
print(a+b)

famille_panda = [
    [100, 5  , 20, 80],
    [50 , 2.5, 10, 40],
    [110, 6  , 22, 80],
]

famille_panda_df = pd.DataFrame(famille_panda,
                                index = ['maman', 'bebe', 'papa'],
                                columns = ['pattes', 'poil', 'queue', 'ventre'])
famille_panda_df

for ind_ligne, contenu_ligne in famille_panda_df.iterrows():
    print("Voici le panda %s :" % ind_ligne)
    print(contenu_ligne)
    print("--------------------")

print(famille_panda_df["ventre"] == 80)

masque = famille_panda_df["ventre"] == 80
pandas_80 = famille_panda_df[masque]
print(pandas_80)
famille_panda_df[~masque]
print(pandas_80)
quelques_pandas = pd.DataFrame([[105,4,19,80],[100,5,20,80]],      # deux nouveaux pandas
                               columns = famille_panda_df.columns) 
                               # même colonnes que famille_panda_df
tous_les_pandas = famille_panda_df.append(quelques_pandas)
print(tous_les_pandas)
tous_les_pandas = tous_les_pandas.drop_duplicates()
print(tous_les_pandas)
