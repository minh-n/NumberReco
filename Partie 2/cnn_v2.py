import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.pyplot as plt
torch.manual_seed(0) #on set le seed a 0, pas d'aleatoire


#--------------------------------------------------
# Reseau de type convolutionnel classique
class CNN(nn.Module):

    #la fonction init permet d’initialiser le reseau de neurones (le modele).
    def __init__(self):
        super(CNN, self).__init__()
        
        #ici, on definit 4 couches de neurones : 2 convolutions et 2 lineaires
        # Les arguments de conv2D : conv2d(in_channels, out_channels, kernel_size)

        # Applies a 2D convolution over an input signal composed of several input planes.
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

        # Applies a linear transformation to the incoming data: y = xA^T + b
        self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 10)

    # Ist's the forward function that defines the network structure.
    # In the forward function, you define how your model is going to be run, from input to output.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(x.shape[0], -1) # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        # print les x pour voir la structure du réseau

        return x

#--------------------------------------------------
# Reseau de type LeNet
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
 
        # La premiere couche contient 6 filtres de convolution de taille 5 × 5 
        # avec une fonction d’activation de type ReLU et un max pooling de dimension 2 × 2.
        self.conv1 = nn.Conv2d(3, 6, 5)

        # La deuxieme couche contient 16 filtres de convolution de taille 5 × 5
        # avec une fonction d’activation de type ReLU et un max pooling de dimension 2 × 2.
        self.conv2 = nn.Conv2d(6, 16, 5)

        # La troisieme couche fully connected contient 120 neurones avec une fonction d’activation de type ReLU (defini dans forward()).
        # 16*5*5 = 16 input de 5*5 (a cause du flatten)
        self.fc1 = nn.Linear(16*5*5, 120)

        # La quatrieme couche fully connected contient 84 neurones avec une fonction d’activation de type ReLU (defini dans forward()).
        self.fc2 = nn.Linear(120, 84)

        # La derniere couche fully connected contient 10 neurones avec une fonction log-softmax (defini dans forward()).
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(x.shape[0], -1) # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

#--------------------------------------------------
# Reseau de type MultiLater Perceptron (pas de conv2D)
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 10)

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

#--------------------------------------------------
# Calcule le taux de succes
def computeSuccess(class_predicted, label, test_size):
    success = 0

    for i in range(0, len(class_predicted)):
        if(class_predicted[i] == label[i]):
            success += 1

    return success*100/test_size


#--------------------------------------------------
# Main du programme
if __name__ == '__main__':

    #--------------------------------------------------
    # Load the dataset, exactement comme la partie 1
    train_data = loadmat('train_32x32.mat')
    test_data = loadmat('test_32x32.mat')

    #--------------------------------------------------
    train_size = 1000
    test_size = 100

    # On recupere les train_size premieres etiquettes de donnees
    train_label = train_data['y'][:train_size]

    #Renvoie tous les elements de la liste avec idClass = 10
    train_label = np.where(train_label==10, 0, train_label)

    #On cast la train_label en int64 puis on le squeeze pour retirer une dimension de la liste (la dimension inutile qui restait)
    train_label = torch.from_numpy(train_label.astype('int64')).squeeze(1)
    
    #On permute les axes de données pour avoir l'ordre idClass, rgb, ligne, col
    train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)[:train_size]

    # memes operations que train_data, pour test_data 
    test_label = test_data['y'][:test_size]
    test_label = np.where(test_label==10, 0, test_label)
    test_label = torch.from_numpy(test_label.astype('int64')).squeeze(1)
    test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)[:test_size]

    #--------------------------------------------------
    # Hyperparameters
    epoch_nbr = 10
    batch_size = 10
    learning_rate = 1e-3 #a l'origine 1e1, mais il nous faut un learning rate < 1

    #--------------------------------------------------
    net = CNN() #indique quel genre de reseau on utilise (CNN, MLP ou LeNet)
    
    #--------------------------------------------------
    # stocke les performances du reseau, afin de les afficher
    train_success = []
    test_success = []

    #--------------------------------------------------
    #Implements stochastic gradient descent (optionally with momentum).
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)


    #--------------------------------------------------
    #on boucle sur le nombre d'epoch
    for e in range(epoch_nbr):

        print("Epoch numero : " + str(e))

        train_in_each_epoch = []
        test_in_each_epoch = []

        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            
            #net() appelle la fonction forward du reseau
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1)

            #calcul de la loss et retropropagation 
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()

            optimizer.step() # Perform the weights update

            #--------------------------------------------------
            # verifier la precision des donnes d'entrainement
            predictions_train = net(train_data[0:test_size])
            _, class_predicted = torch.max(predictions_train, 1)

            success = computeSuccess(class_predicted, train_label, test_size)
            train_in_each_epoch.append(success)
            print("Train success : " + str(success) + "%.")

            #--------------------------------------------------
            # verifier la precision des donnes de test
            predictions_test = net(test_data[0:test_size])
            _, class_predicted = torch.max(predictions_test, 1)

            success = computeSuccess(class_predicted, test_label, test_size)
            test_in_each_epoch.append(success)
            print("Test success : " + str(success) + "%.")

        #--------------------------------------------------
        # pourcentage de reussite total
        test_success.append(sum(test_in_each_epoch)/len(test_in_each_epoch))
        train_success.append(sum(train_in_each_epoch)/len(train_in_each_epoch))

    #--------------------------------------------------
    # plot le graphe des %
    plt.plot(test_success)
    plt.plot(train_success)

    plt.show()

