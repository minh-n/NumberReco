# NumberReco

Voici un projet de reconnaissance d'image mené dans le cadre du cours de Big Data (ET5 Informatique).


## Introduction

L’objectif de ce projet est de créer un programme capable de classer des images de nombres, issues du monde réel. 

Nous utiliserons plusieurs algorithmes de classification tels que le classifieur à distance minimum (DMIN), les classifieurs de Scikit-Learn (SVM). Ces méthodes seront assistées d'un traitement préalable des données (pré-processing), ainsi qu'une réduction de la dimension des vecteurs (PCA).

Dans un deuxième temps, nous utiliserons des réseaux de neurones pour effectuer cette classification. 

À travers l’expérimentation de plusieurs méthodes de pré-processing et de classification, nous avons entraîné et amélioré notre modèle, jusqu’à atteindre autour de 18% de réussite avec DMIN et pré-processing, autour de 16% pour C-Support Vector Classification (SVC) et 46% pour le KNeighborsClassifier (KNN). Quant aux réseaux de neurones, le succès est de l'ordre de 80%.

Ce projet nous a permis d’expérimenter avec les bases du machine learning afin de gérer un gros volume de données. En effet, les classificateurs d’images sont une application répandue et très utile au monde des big data de nos jours. La reconnaissance de nombres est une étape basique mais nécessaire dans de nombreux domaines, comme la surveillance, la robotique ou encore la réalité augmentée. 

## Data

Nous utiliserons la bibliothèque SVHN, qui contient environ 600 000 numéros de maison extraits de Google Street View. Nous n’en utiliserons que 100 000 environ afin d’entraîner notre modèle et le tester. L’intérêt de SVHN comparé à d’autre datasets repose principalement sur la quantité très élevée de données issues du monde réel. Ces données possèdent toutes un label correspondant à chaque chiffre présent sur une image. 

Les données sont disponibles sur le site http://ufldl.stanford.edu/housenumbers sous la rubrique
Downloads → Format 2. Les fichiers d’entraînement (train_32x32.mat) et de test (test_32x32.mat)
contiennent respectivement 73257 et 26032 images de chiffres photographiés dans la rue et leurs  étiquettes associées. Les images ont une taille de 32 × 32 pixels codés en RVB avec 256 niveaux d’intensité.

## Résultats

Les résultats du projet sont disponibles dans les deux rapports en PDF dans le repository du projet.
