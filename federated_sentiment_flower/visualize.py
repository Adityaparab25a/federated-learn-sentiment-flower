import matplotlib.pyplot as plt


rounds = [1, 2, 3]
accuracy = [0.76, 0.82, 0.85]
loss = [0.5, 0.4, 0.35]


plt.figure()
plt.plot(rounds, accuracy, marker='o')
plt.title('Federated Learning Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.savefig('accuracy_plot.png')


plt.figure()
plt.plot(rounds, loss, marker='o', color='red')
plt.title('Federated Learning Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.savefig('loss_plot.png')