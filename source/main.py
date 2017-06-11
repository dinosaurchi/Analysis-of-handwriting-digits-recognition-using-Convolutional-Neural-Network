from ConvNetPredictor import ConvNetPredictor
from LinearPredictor import LinearPredictor
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def plot_images(image):
    fig, ax = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(ax.flat):
        ax.imshow(image.reshape((28, 28)), cmap='binary')
    plt.show()


def main():
    # Auto downloading MNIST data
    data = input_data.read_data_sets("data/MNIST/", one_hot=True)

    """
        Test the Linear model
        If you want to test direct with a trained model, please download our pre-trained model
            Link:
                Linear model:
                    https://drive.google.com/file/d/0B3TTRMAPMT0DMTUwS1FUSFVxR1k/view?usp=sharing

                CNN-based mode:
                    https://drive.google.com/file/d/0B3TTRMAPMT0DeDVPOTBRN041bTQ/view?usp=sharing

            Then, restore the model:
                Linear model:
                    predictor.restore_model('./model_linear-i13800-b100-a92.9700016975')

                CNN-based model:
                    predictor.restore_model('./model_cnn-i33000-b128-a99.1299986839')

        Otherwise, you can train the model yourself
            Training code:

                predictor.train(data.train, data.test)
    """

    predictor = LinearPredictor()  # or LinearPredictor for linear model

    predictor.restore_model('./model_linear-i13800-b100-a92.9700016975')

    result = predictor.run_test(data.test)
    print '     Loss     = ' + str(result[0])
    print '     Accuracy = ' + str(result[1] * 100) + '%'


if __name__ == '__main__':
    main()
