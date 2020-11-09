import json
import numpy as np
import matplotlib.pyplot as plt

fname = "metrics-contrastive.txt"

with open(fname) as f:
    x=json.load(f)
    training_loss = np.array(x['training_loss'])
    plt.plot(training_loss)
    plt.title(fname + ": Training Loss vs Epoch")
    plt.show()
    training_acc = np.array(x['training_acc'])
    plt.plot(training_acc)
    plt.title(fname + ": Training Acc vs Epoch")
    plt.show()
    val_loss = np.array(x['val_loss'])
    plt.plot(val_loss)
    plt.title(fname + ": Validation Loss vs Epoch")
    plt.show()
    val_acc = np.array(x['val_acc'])
    plt.plot(val_acc)
    plt.title(fname + ": Validation Acc vs Epoch")
    plt.show()