{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOCQLEh90Oj7P79DkP0bg1D",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/vivek201102/ML-Labs/blob/master/Lab_12.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "_8uTydnyYjVl"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.autograd import Variable\n",
        "from torchvision.datasets import MNIST\n",
        "from torch.utils.data import DataLoader,TensorDataset\n",
        "import tensorflow\n",
        "from sklearn.model_selection import train_test_split\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mnist = MNIST(root='data/', train=True, download=True)"
      ],
      "metadata": {
        "id": "cgqlalFMZBxD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "features_train = mnist.data.float()\n",
        "targets_train = mnist.targets.long()\n",
        "features_train /= 255\n",
        "features_train = features_train.view(-1, 784)"
      ],
      "metadata": {
        "id": "5MMmtX7lZDf4"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "batch_size = 100\n",
        "n_iters = 5000\n",
        "num_epochs = n_iters / (len(features_train) / batch_size)\n",
        "num_epochs = int(num_epochs)"
      ],
      "metadata": {
        "id": "I3Y0-bKuZFz9"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_set = TensorDataset(features_train, targets_train)\n",
        "train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)\n",
        "features_train = torch.tensor(features_train, requires_grad=True)\n",
        "targets_train = torch.tensor(targets_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "97lxLZD0ZHev",
        "outputId": "d78c7615-064c-48c2-d334-deae1f858076"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-5-eb29c107d267>:3: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  features_train = torch.tensor(features_train, requires_grad=True)\n",
            "<ipython-input-5-eb29c107d267>:4: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  targets_train = torch.tensor(targets_train)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mnist_test = MNIST(root='data/', train=False, download=True)\n",
        "features_test = mnist_test.data.float()\n",
        "targets_test = mnist_test.targets.long()\n",
        "features_test /= 255\n",
        "features_test = features_test.view(-1, 784)"
      ],
      "metadata": {
        "id": "l_5ikA98ZJBr"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_set = TensorDataset(features_test, targets_test)\n",
        "test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)"
      ],
      "metadata": {
        "id": "LF2dU0ndZKbE"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.imshow(mnist.data[3], cmap='autumn')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "id": "skoucqlCZNlp",
        "outputId": "1a10c5e3-7cc9-445e-9454-f44c88bd5124"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAALuElEQVR4nO3db4gc9R3H8c+nVimotNGk1zQGYyVPQmljOULB0KZoRUMh8YmYBxKp9HygRcGCYgum0AehrYoUK5w1GotVBBXzILSmwTZIqXhKmj9GGxsi5rjkTkJrfGRjvn1wE3vGu53NzszOJt/3C47dne/s/L4M+WR2Z3b354gQgLPf59puAEB/EHYgCcIOJEHYgSQIO5DE5/s52Hw7lvRzQCCZg5Lej/BstUpht32tpIcknSPpdxGxsdP6SySNVRkQQEfDHWo9v4y3fY6khyVdJ2mZpHW2l/W6PQDNqvKefYWkdyLiQER8JOkZSWvqaQtA3aqEfZGk92Y8PlQs+xTbI7bHbI9NVRgMQDWNn42PiNGIGI6I4QVNDwZgTlXCPi5p8YzHlxTLAAygKmF/TdJS25fZPk/SjZK21NMWgLr1fOktIo7bvl3SnzR96W1TROytrTMAtap0nT0itkraWlMvABrEx2WBJAg7kARhB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkCDuQBGEHkiDsQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AEYQeSIOxAEoQdSIKwA0kQdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IotIsrkCjflZS/3lJvdOhbFXJc/9aUj8DVQq77YOSjkn6WNLxiBiuoykA9avjyP69iHi/hu0AaBDv2YEkqoY9JL1k+3XbI7OtYHvE9pjtsamKgwHoXdWX8SsjYtz2lyVts/1WROyYuUJEjEoalaRhOyqOB6BHlY7sETFe3E5KekHSijqaAlC/nsNu+3zbF568L+kaSXvqagxAvaq8jB+S9ILtk9v5Q0T8sZaukMPNJfW7S+onKoyd8A1lz2GPiAOSvlljLwAaxKU3IAnCDiRB2IEkCDuQBGEHkuArrmjPpSX1L/SlizQ4sgNJEHYgCcIOJEHYgSQIO5AEYQeSIOxAElxnR7Ou7lD7ccVtv1VS/0GH2pGKY5+BOLIDSRB2IAnCDiRB2IEkCDuQBGEHkiDsQBJcZ0c1K0vqj3eofbHi2L8qqb9bcftnGY7sQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AE19lRzfqS+lcrbPsvJfUnK2w7odIju+1Ntidt75mx7CLb22zvL27nNdsmgKq6eRn/hKRrT1l2j6TtEbFU0vbiMYABVhr2iNgh6egpi9dI2lzc3yxpbb1tAahbryfohiJiorh/WNLQXCvaHrE9ZntsqsfBAFRX+Wx8RISk6FAfjYjhiBheUHUwAD3rNexHbC+UpOJ2sr6WADSh17Bv0f8vuqyX9GI97QBoSul1dttPS1olab7tQ5Luk7RR0rO2b9H0t4ZvaLJJtGh+Sf2HJfUTHWr/LnnuL0rqOC2lYY+IdXOUrqq5FwAN4uOyQBKEHUiCsANJEHYgCcIOJMFXXLNbUlJ/rsGxf1NSf7nBsRPiyA4kQdiBJAg7kARhB5Ig7EAShB1IgrADSXCdPbtTf0r0VN+ouP3tHWoPVdw2TgtHdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IguvsZ7u1JfWNFbf/Skm905TO/6k4Nk4LR3YgCcIOJEHYgSQIO5AEYQeSIOxAEoQdSILr7GeDJR1qTf7uuyQdKKkfaXh8dK30yG57k+1J23tmLNtge9z2zuJvdbNtAqiqm5fxT2j23zN5MCKWF39b620LQN1Kwx4ROyQd7UMvABpU5QTd7bZ3FS/z5821ku0R22O2x6YqDAagml7D/oikyyUtlzQh6f65VoyI0YgYjojhBT0OBqC6nsIeEUci4uOIOCHpUUkr6m0LQN16CrvthTMeXi9pz1zrAhgMpdfZbT8taZWk+bYPSbpP0irbyyWFpIOSbm2uRZS6u0PtRMNjV/0+PPqmNOwRsW6WxY810AuABvFxWSAJwg4kQdiBJAg7kARhB5LgK65nguUl9WsaHPvFkvrbDY6NWnFkB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkuM5+JnippD7nj4J14e8l9ZsrbBsDhSM7kARhB5Ig7EAShB1IgrADSRB2IAnCDiTBdfYzwcUl9So/F/3bkvqHFbaNgcKRHUiCsANJEHYgCcIOJEHYgSQIO5AEYQeS4Dr7IHi8pN7kf8l/a3DbGCil/4xsL7b9su03be+1fUex/CLb22zvL26r/IQCgIZ1c8w4LumuiFgm6duSbrO9TNI9krZHxFJJ24vHAAZUadgjYiIi3ijuH5O0T9IiSWskbS5W2yxpbUM9AqjBab0btL1E0hWSXpU0FBETRemwpKE5njNie8z22FSVTgFU0nXYbV8g6TlJd0bEBzNrERGSYrbnRcRoRAxHxPCCSq0CqKKrsNs+V9NBfyoini8WH7G9sKgvlDTZTIsA6lB66c22JT0maV9EPDCjtEXSekkbi9uyyX3zWl5Sv7qkXvYV1o861B4uee6RkjrOGt1cZ79S0k2SdtveWSy7V9Mhf9b2LZLelXRDIx0CqEVp2CPiFUmeo3xVve0AaAoflwWSIOxAEoQdSIKwA0kQdiAJvuLaD18qqX+l4vbHO9R+UnHbOGtwZAeSIOxAEoQdSIKwA0kQdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEk+D57P7xVUi+bNnllXY0gM47sQBKEHUiCsANJEHYgCcIOJEHYgSQIO5BEN/OzL5b0pKQhSSFpNCIesr1B0o8kTRWr3hsRW5tq9Ix2uKT+3b50geS6+VDNcUl3RcQbti+U9LrtbUXtwYj4dXPtAahLN/OzT0iaKO4fs71P0qKmGwNQr9N6z257iaQrJL1aLLrd9i7bm2zPm+M5I7bHbI9NzbYCgL7oOuy2L5D0nKQ7I+IDSY9IulzSck0f+e+f7XkRMRoRwxExvKB6vwB61FXYbZ+r6aA/FRHPS1JEHImIjyPihKRHJa1ork0AVZWG3bYlPSZpX0Q8MGP5whmrXS9pT/3tAahLN2fjr5R0k6TdtncWy+6VtM72ck1fjjso6dYG+gNQk27Oxr8iybOUuKYOnEH4BB2QBGEHkiDsQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AEYQeSIOxAEoQdSIKwA0kQdiAJR0T/BrOnJL07Y9F8Se/3rYHTM6i9DWpfEr31qs7eLo2IWX8Brq9h/8zg9lhEDLfWQAeD2tug9iXRW6/61Rsv44EkCDuQRNthH215/E4GtbdB7Uuit171pbdW37MD6J+2j+wA+oSwA0m0Enbb19p+2/Y7tu9po4e52D5oe7ftnbbHWu5lk+1J23tmLLvI9jbb+4vbWefYa6m3DbbHi3230/bqlnpbbPtl22/a3mv7jmJ5q/uuQ1992W99f89u+xxJ/5T0fUmHJL0maV1EvNnXRuZg+6Ck4Yho/QMYtr8j6UNJT0bE14tlv5R0NCI2Fv9RzouIuwektw2SPmx7Gu9itqKFM6cZl7RW0s1qcd916OsG9WG/tXFkXyHpnYg4EBEfSXpG0poW+hh4EbFD0tFTFq+RtLm4v1nT/1j6bo7eBkJETETEG8X9Y5JOTjPe6r7r0FdftBH2RZLem/H4kAZrvveQ9JLt122PtN3MLIYiYqK4f1jSUJvNzKJ0Gu9+OmWa8YHZd71Mf14VJ+g+a2VEfEvSdZJuK16uDqSYfg82SNdOu5rGu19mmWb8E23uu16nP6+qjbCPS1o84/ElxbKBEBHjxe2kpBc0eFNRHzk5g25xO9lyP58YpGm8Z5tmXAOw79qc/ryNsL8maanty2yfJ+lGSVta6OMzbJ9fnDiR7fMlXaPBm4p6i6T1xf31kl5ssZdPGZRpvOeaZlwt77vWpz+PiL7/SVqt6TPy/5L00zZ6mKOvr0n6R/G3t+3eJD2t6Zd1/9X0uY1bJF0sabuk/ZL+LOmiAert95J2S9ql6WAtbKm3lZp+ib5L0s7ib3Xb+65DX33Zb3xcFkiCE3RAEoQdSIKwA0kQdiAJwg4kQdiBJAg7kMT/AOh2aimmSat1AAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class ANNModel(nn.Module):\n",
        "    def __init__(self, input_dim, hidden_dim, output_dim):\n",
        "        super(ANNModel, self).__init__()\n",
        "        self.fc1 = nn.Linear(input_dim, hidden_dim)\n",
        "        self.relu = nn.ReLU()\n",
        "        self.fc2 = nn.Linear(hidden_dim, output_dim)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        out = self.fc1(x)\n",
        "        out = self.relu(out)\n",
        "        out = self.fc2(out)\n",
        "        return out\n",
        "\n",
        "# instantiate ANN\n",
        "input_dim = 28*28\n",
        "hidden_dim = 100\n",
        "output_dim = 10\n",
        "\n",
        "# Create ANN\n",
        "model = ANNModel(input_dim, hidden_dim, output_dim)\n",
        "\n",
        "# Cross Entropy Loss\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "\n",
        "# SGD Optimizer\n",
        "learning_rate = 0.02\n",
        "optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)"
      ],
      "metadata": {
        "id": "VBVGI5dcZPVL"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs = 5\n",
        "count=0\n",
        "\n",
        "# Define empty lists to store loss, iteration and accuracy values\n",
        "loss_list = []\n",
        "iteration_list = []\n",
        "accuracy_list = []\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    for i, (images, labels) in enumerate(train_loader):\n",
        "\n",
        "        # Convert input and labels to Variables\n",
        "        train = (images.view(-1, 28*28))\n",
        "        labels = (labels)\n",
        "\n",
        "        # Clear gradients\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "        # Forward propagation\n",
        "        outputs = model(train)\n",
        "\n",
        "        # Calculate softmax and cross entropy loss\n",
        "        loss = criterion(outputs, labels)\n",
        "\n",
        "        # Calculating gradients\n",
        "        loss.backward()\n",
        "\n",
        "        # Update parameters\n",
        "        optimizer.step()\n",
        "\n",
        "        # Count iterations\n",
        "        count += 1\n",
        "\n",
        "        # Calculate accuracy every 50 iterations\n",
        "        if count % 50 == 0:\n",
        "            correct = 0\n",
        "            total = 0\n",
        "            for images, labels in test_loader:\n",
        "                test = (images.view(-1, 28*28))\n",
        "                outputs = model(test)\n",
        "                predicted = torch.max(outputs.data, 1)[1]\n",
        "                total += labels.size(0)\n",
        "                correct += (predicted == labels).sum()\n",
        "\n",
        "            accuracy = 100 * correct / float(total)\n",
        "\n",
        "            # Store loss and iteration values\n",
        "            loss_list.append(loss.data)\n",
        "            iteration_list.append(count)\n",
        "            accuracy_list.append(accuracy)\n",
        "\n",
        "        # Print loss and accuracy every 500 iterations\n",
        "        if count % 500 == 0:\n",
        "            print('Iteration: {} Loss: {:.4f} Accuracy: {:.2f} %'.\n",
        "                  format(count, loss.data, accuracy))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7O8-8BX3ZRnH",
        "outputId": "3d8669f5-95c2-4a9a-93a5-9685079bc3c2"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Iteration: 500 Loss: 0.6300 Accuracy: 86.91 %\n",
            "Iteration: 1000 Loss: 0.5344 Accuracy: 89.46 %\n",
            "Iteration: 1500 Loss: 0.4035 Accuracy: 90.38 %\n",
            "Iteration: 2000 Loss: 0.2331 Accuracy: 90.93 %\n",
            "Iteration: 2500 Loss: 0.3536 Accuracy: 91.72 %\n",
            "Iteration: 3000 Loss: 0.1998 Accuracy: 92.08 %\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(iteration_list,loss_list)\n",
        "plt.xlabel(\"Number of iteration\")\n",
        "plt.ylabel(\"Loss\")\n",
        "plt.title(\"ANN: Loss vs Number of iteration\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "wDQ9soZvZUX-",
        "outputId": "c1ac613b-233b-424b-83b2-034639f61ba5"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(iteration_list,accuracy_list,color = \"red\")\n",
        "plt.xlabel(\"Number of iteration\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.title(\"ANN: Accuracy vs Number of iteration\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "02r6zAd6ZWS0",
        "outputId": "bfec5e96-ac17-41db-c254-06a5698dde2a"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}
