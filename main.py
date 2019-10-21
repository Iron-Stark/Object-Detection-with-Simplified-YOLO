import train_model
import test_network
import precision_recall

def main():
    #train_model.start_train()
    test_network.test()
    precision_recall.plot_curve()

if __name__ == '__main__':
    main()
