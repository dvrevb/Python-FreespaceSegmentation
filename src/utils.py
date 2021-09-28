#author: Burak Cevik

import matplotlib.pyplot as plt
def draw_graph(val_losses,train_losses,epochs):
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    epoch_numbers=list(range(1,epochs+1,1))
    plt.plot(epoch_numbers,norm_validation, label="validation",color="blue")
    plt.plot(epoch_numbers,norm_train,label="training",color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.show()

