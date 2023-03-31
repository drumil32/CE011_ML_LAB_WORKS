{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPIDhsKJrKrxv32r6nAqr4b",
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
        "<a href=\"https://colab.research.google.com/github/vivek201102/ML-Labs/blob/master/Lab_09.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import make_blobs\n",
        "X, y = make_blobs(n_samples = 100,centers=3,n_features=2,cluster_std=0.2,random_state=0)"
      ],
      "metadata": {
        "id": "nLLDXOYhPPsA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Scatter plot of the data points\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline"
      ],
      "metadata": {
        "id": "TihcQJpdPWxO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1Cdh_KPK-laj",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "0a51cd13-610f-4ba4-c1cc-1e72cdb4f52d"
      },
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
      ],
      "source": [
        "plt.scatter(X[:,0],X[:,1], c='white',marker='o',edgecolor='black',s=50)\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#using scikit-learn to perform K-means clustering\n",
        "from sklearn.cluster import KMeans"
      ],
      "metadata": {
        "id": "K6N4iJywPKK9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "km = KMeans(n_clusters=3, init='random',n_init=10,max_iter=300, tol=1e-04, random_state=0)\n",
        "y_km = km.fit_predict(X)"
      ],
      "metadata": {
        "id": "DZ7QwF7PPs1X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(\n",
        "    X[y_km == 0,0],X[y_km == 0,1],\n",
        "    s=50, c='lightgreen',\n",
        "    marker='s', edgecolor='black',\n",
        "    label='cluster 1'\n",
        ")\n",
        "\n",
        "plt.scatter(\n",
        "    X[y_km == 1,0],X[y_km == 1,1],\n",
        "    s=50, c='orange',\n",
        "    marker='o',edgecolor='black',\n",
        "    label='cluster 2'\n",
        ")\n",
        "\n",
        "plt.scatter(\n",
        "    X[y_km == 2,0],X[y_km == 2,1],\n",
        "    s=50,c='lightblue',\n",
        "    marker='v',edgecolor='black',\n",
        "    label='cluster 3'\n",
        ")\n",
        "\n",
        "#plot the centroids\n",
        "plt.scatter(\n",
        "    km.cluster_centers_[:,0],km.cluster_centers_[:,1],\n",
        "    s=250, marker='*',\n",
        "    c='red',edgecolor='black',\n",
        "    label='centroids'\n",
        ")\n",
        "\n",
        "plt.legend(scatterpoints=1)\n",
        "plt.grid()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "id": "EkIcLbbEPwqD",
        "outputId": "937f51ac-8438-44f4-90eb-a9a75e0f6b57"
      },
      "execution_count": null,
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
        "from sklearn.cluster import KMeans\n",
        "from sklearn import datasets\n",
        "from sklearn.metrics import silhouette_score"
      ],
      "metadata": {
        "id": "1Pd0I5cLPzuX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "score = silhouette_score(X,km.labels_,metric='euclidean')\n",
        "print('Silhouetter Score: %.3f' % score)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Zo8wlw4qP6eX",
        "outputId": "3b5d6e5e-8d01-475f-b3aa-59d401aec93c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Silhouetter Score: 0.882\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Elbow method"
      ],
      "metadata": {
        "id": "9-0HKNx3QD4v"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# from kneed import KneeLocator\n",
        "from sklearn.preprocessing import StandardScaler"
      ],
      "metadata": {
        "id": "UMj07NuxQGNP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "distortions = []\n",
        "for i in range(1,11):\n",
        "  km = KMeans(\n",
        "      n_clusters=i, init='random',\n",
        "      n_init=10,max_iter=300,\n",
        "      tol=1e-04, random_state=0\n",
        "  )\n",
        "  km.fit(X)\n",
        "  distortions.append(km.inertia_)\n",
        "\n",
        "#plot\n",
        "plt.plot(range(1,11),distortions,marker='o')\n",
        "plt.xlabel('Number of clusters')\n",
        "plt.ylabel('Distortion')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "5Cc5iwt8QLY3",
        "outputId": "474c3393-a81e-4dab-921f-b097670a8c51"
      },
      "execution_count": null,
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
        "# from kneed import KneeLocator\n",
        "!pip install --upgrade kneed"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tykZGSJhQOSz",
        "outputId": "f193033d-2d7a-4bfa-e37c-3f846428eb25"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting kneed\n",
            "  Downloading kneed-0.8.2-py3-none-any.whl (10 kB)\n",
            "Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.9/dist-packages (from kneed) (1.10.1)\n",
            "Requirement already satisfied: numpy>=1.14.2 in /usr/local/lib/python3.9/dist-packages (from kneed) (1.22.4)\n",
            "Installing collected packages: kneed\n",
            "Successfully installed kneed-0.8.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        " from kneed import KneeLocator"
      ],
      "metadata": {
        "id": "wwMWlWtYQUB7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Determining the elbow point in the SSE curve isn’t always straightforward. If you’re having trouble choosing the elbow point of the curve, then you could use a Python package, kneed, to identify the elbow point programmatically:\n",
        "kl = KneeLocator(range(1,11),distortions, curve='convex',direction='decreasing')\n",
        "kl.elbow"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NmyydvsdQWkj",
        "outputId": "47074a03-59a6-4a98-fbea-a5a45fffba45"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "silhouette_coefficients = []\n",
        "for k in range(2, 11):\n",
        "  kmeans = KMeans(n_clusters=k)\n",
        "  kmeans.fit(X)\n",
        "  score = silhouette_score(X, kmeans.labels_)\n",
        "  silhouette_coefficients.append(score)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EU6DaVfgQZYq",
        "outputId": "71a033b7-3bc3-48c4-d9e9-5bfbeead14f4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.style.use(\"fivethirtyeight\")\n",
        "plt.plot(range(2, 11), silhouette_coefficients)\n",
        "plt.xticks(range(2, 11))\n",
        "plt.xlabel(\"Number of Clusters\")\n",
        "plt.ylabel(\"Silhouette Coefficient\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 306
        },
        "id": "MRE7mfRkQb-c",
        "outputId": "16bec98d-a285-4dfe-964e-72dc2e368ec0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from sklearn.cluster import KMeans\n",
        "#x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])\n",
        "X1 = np.array([[2,2],[3,4],[6,7],[5,5],[3,3],[12,13]])\n",
        "#X1.shape()\n",
        "X2 = np.array([2,3,4])\n",
        "plt.scatter(X1[:,0], X1[:,1])\n",
        "centroid = np.array([[2,2],[3,4],[6,7]])\n",
        "kmeans = KMeans(n_clusters=3, init=centroid).fit(X1)\n",
        "kmeans.cluster_centers_\n",
        "kmeans.labels_\n",
        "#print(\"Number of iterations \" , kmeans.n_iter_)\n",
        "kmeans.predict([[8,8]])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 389
        },
        "id": "HAkoJ7ZjQe-C",
        "outputId": "e9a3aaee-135a-4d70-c18a-b20b483f0024"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:1362: RuntimeWarning: Explicit initial center position passed: performing only one init in KMeans instead of n_init=10.\n",
            "  super()._check_params_vs_input(X, default_n_init=10)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1], dtype=int32)"
            ]
          },
          "metadata": {},
          "execution_count": 18
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Hand Written Digit Recognition"
      ],
      "metadata": {
        "id": "Iw_5FZEKQi76"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.datasets import load_digits\n",
        "#digits dataset from scikit learn consists of 8x8 pixel images of digits\n",
        "#Data plotting and visualization libraries\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy.stats import mode\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix"
      ],
      "metadata": {
        "id": "hCSb_wIcQjxK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "digits = load_digits() #load the dataset in digits"
      ],
      "metadata": {
        "id": "4cFbCJEcQpzy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "digits.data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fseamwLgQsGX",
        "outputId": "18c0245b-9275-48e1-9c9a-36fbc50f3cac"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1797, 64)"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "digits.keys() #Dataset loaded is a dictionary\n",
        "# data : flattened arrays/tensors used for clustering\n",
        "# target : label associated with flattened array\n",
        "#print(digits.target)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_n5x8WWPQz2G",
        "outputId": "e0a32c22-8a30-4f28-88ff-11eab0523142"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "digits.data[0:3] #flattened data for 3 images of the dataset"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H6VjprS-Q2eK",
        "outputId": "37c78a58-5af2-44f6-a503-21a4c4b894e6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,\n",
              "        15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,\n",
              "        12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,\n",
              "         0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,\n",
              "        10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.],\n",
              "       [ 0.,  0.,  0., 12., 13.,  5.,  0.,  0.,  0.,  0.,  0., 11., 16.,\n",
              "         9.,  0.,  0.,  0.,  0.,  3., 15., 16.,  6.,  0.,  0.,  0.,  7.,\n",
              "        15., 16., 16.,  2.,  0.,  0.,  0.,  0.,  1., 16., 16.,  3.,  0.,\n",
              "         0.,  0.,  0.,  1., 16., 16.,  6.,  0.,  0.,  0.,  0.,  1., 16.,\n",
              "        16.,  6.,  0.,  0.,  0.,  0.,  0., 11., 16., 10.,  0.,  0.],\n",
              "       [ 0.,  0.,  0.,  4., 15., 12.,  0.,  0.,  0.,  0.,  3., 16., 15.,\n",
              "        14.,  0.,  0.,  0.,  0.,  8., 13.,  8., 16.,  0.,  0.,  0.,  0.,\n",
              "         1.,  6., 15., 11.,  0.,  0.,  0.,  1.,  8., 13., 15.,  1.,  0.,\n",
              "         0.,  0.,  9., 16., 16.,  5.,  0.,  0.,  0.,  0.,  3., 13., 16.,\n",
              "        16., 11.,  5.,  0.,  0.,  0.,  0.,  3., 11., 16.,  9.,  0.]])"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# km = KMeans(n_samples=1797,n_features=64,n_clusters=10, init='random',n_init=10,max_iter=300, tol=1e-04, random_state=0)\n",
        "# y_km = km.fit_predict(digits.data[0:3])\n",
        "X = digits.data"
      ],
      "metadata": {
        "id": "rS2Jrn_WQ5qH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.cluster import KMeans\n",
        "kmeans = KMeans(n_clusters=10, random_state=0)\n",
        "kmeans.fit(digits.data)\n",
        "print(kmeans.labels_)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sj5Q1gEARAfm",
        "outputId": "ff13a888-6e96-4de4-820a-0b69722eef95"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[5 7 7 ... 7 3 3]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fig, ax = plt.subplots(2, 5, figsize = (8,3)) #Create a figure and a set of␣subplots( 2 rows and 5 columns)\n",
        "centers = kmeans.cluster_centers_.reshape(10,8,8)\n",
        "#flattened image can't be viewed, re-transform/reshape/inverse transform it to␣original form to view matrix shaped image\n",
        "#reshape 10 rows of clusters (k_means.cluster_centers_ = 10,64) and 64 to 8 * 8␣matrix\n",
        "for axi, center in zip(ax.flat, centers): #ax.flat:flattening the image &␣plotting relevant centers\n",
        "  axi.set(xticks = [], yticks = [])\n",
        "  axi.imshow(center, interpolation='nearest',cmap = plt.cm.binary)\n",
        "#imshow(matplotlib method) to render the image in notebook"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "v1vfAJZKQ9Ha",
        "outputId": "f7e1e9b6-f5bd-4030-ca7e-b97b16e9bdfe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 576x216 with 10 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfwAAAC9CAYAAABWIc9uAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAANjklEQVR4nO3cS4gV9N/H8cbxOo6jkmGhOaaJZdEFKiswtVC0lFyUqaVFLcSioDTLWqTtpAtIaaBuRGghGViaZgWZdqMsijLIcjKsVNIZdbTxMvqsnmeX8+Fp/v7n9Hu9tufNGfXMmY9nMd+qpqamM+cBAP9qnf7bfwAA4D/P4ANAAQw+ABTA4ANAAQw+ABTA4ANAAQw+ABTA4ANAATonUe/evdvli508eTLqFi9eHHVr166NujfffDPq6uvro65Tp7b/n3To0KHoudpDW6/PsWPHoueZP39+1K1YsSLqampqou7pp5+OusceeyzqevTocdbHz+Vrc955bb8+Bw8ejJ5n9uzZUffxxx9HXXNzc9QNGDAg6l544YWomzRp0lkf70jvndbW1uh5lixZEnVLly6Nutra2qh79NFHo27atGnt9nU70utz+vTp6HnSrXjyySejbv/+/VE3cuTIqJszZ07UjR8//qyPV1VVnfVxn/ABoAAGHwAKYPABoAAGHwAKYPABoAAGHwAKEP1aXnvZvn171L300ktRN3PmzKjr1q1b1B0+fDjq0l+Z6Si2bt0adR988EHUzZo1K+p27twZdWvWrIm69FeL0l+v7Ci2bdsWdW+88UbUXXbZZVE3ZcqUqBs0aFDUDRkyJOoqyY4dO6Lu2Wefjbq77ror6tJfB0y/JyZPnhx1lfazrbGxMepWrlwZdRdeeGHUjRgxIurSn4Hpr063tWUnTpw46+M+4QNAAQw+ABTA4ANAAQw+ABTA4ANAAQw+ABTA4ANAAQw+ABTA4ANAAdrl0t7Ro0ejbuHChVGXXgqbMGFC1H344YdRl15ZGjVqVNR1FOm/54oVK9r1686dOzfqLr300qjr27fvP/njdFi///571HXt2jXqHn/88agbN25c1PXp0yfq6urqoq6S7N69O+r69esXdemlvfTC34YNG6IuveRWac6cORN1M2bMiLqRI0dG3VtvvRV1Bw4ciLr0Z2D37t3P+rhLewCAwQeAEhh8ACiAwQeAAhh8ACiAwQeAAhh8ACiAwQeAAhh8AChAu1zae/fdd9u1W758edR99913UffBBx9E3eTJk6Pu1ltvjbqOor6+PurSi2qLFi2Kup9++inqpk6dGnU9evSIukqTXqirrq6OuldeeSXqfvnll6h74IEHoq53795RV0nSC2hXXnll1D333HNRt3fv3qhLL/yl11ArTfqeaOtC3f/auHFj1K1evTrqUvv27Yu6qqqqf/R1fMIHgAIYfAAogMEHgAIYfAAogMEHgAIYfAAogMEHgAIYfAAogMEHgAK0y6W9TZs2tcfT/J9t27ZF3datW6Pu999/j7p777036irNmTNnoq5z5+zbYcyYMVH3ySefRN2qVauibtKkSVE3fPjwqOsobrrppqh75JFHou7nn3+Ouh07dkTdZ599FnUDBw6MupqamqjrCIYOHRp16QW9L7/8Muqam5ujbt26dVH366+/Rl16MbCjaGlpibotW7ZE3dtvvx11e/bsibpx48ZF3bm6UukTPgAUwOADQAEMPgAUwOADQAEMPgAUwOADQAEMPgAUwOADQAEMPgAUIDqt1talthEjRkRf7IYbboi67du3R11DQ0PUjR8/Pupuu+22qKs0J06ciLojR45EXfrv1K1bt6ibN29e1L333ntRV2mX9tILdem/U3qpcs2aNVGXXqpsbW2NukpSVVUVdemltIkTJ0ZdevXynXfeiboff/wx6m6//fao6yjSnzG33HJL1H399ddRl15gXLBgQdTdeOONUfdP+YQPAAUw+ABQAIMPAAUw+ABQAIMPAAUw+ABQAIMPAAUw+ABQAIMPAAXIzjm1YebMmVE3atSoqFu7dm3U7d69O+qeeOKJqBswYEDUVZqWlpaoW7lyZdTV1tZG3a5du6Ju3759UZdeH6u0i2/p67Nly5aoW79+fdSl19fuvPPOqOvatWvUVZK//vor6pYvXx51V1xxRdT16dMn6vbv3x916UW6tq6qdjR9+/aNumHDhkXdoUOHom727NlRd/PNN0dd+vr8Uz7hA0ABDD4AFMDgA0ABDD4AFMDgA0ABDD4AFMDgA0ABDD4AFMDgA0ABotNlVVVVZ338/PPPj75Yr169om7NmjVRV19fH3WXX3551P1bpVec0strq1evjrq6urqomzFjRtRNnDgx6k6dOhV1HcXJkyejLr2EmF7kmzt3btRNmDAh6s7VtbBz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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "clusters=digits.target\n",
        "labels = np.zeros_like(clusters) # blank labels\n",
        "print(f\"The labels are : {labels}\")\n",
        "print(f\"\\nThe size of labels is : {labels.shape}\")\n",
        "print(\"The mask values are : \")\n",
        "for i in range(10):\n",
        "  mask = (clusters == i)\n",
        "#if a specific digit belongs to/equivalent a specific cluster then its True␣else False\n",
        "  print(mask)\n",
        "  labels[mask] = mode(digits.target[mask])[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DxejcTHOVVwF",
        "outputId": "224e575c-1d71-464a-8018-dbf04c70c026"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The labels are : [0 0 0 ... 0 0 0]\n",
            "\n",
            "The size of labels is : (1797,)\n",
            "The mask values are : \n",
            "[ True False False ... False False False]\n",
            "[False  True False ... False False False]\n",
            "[False False  True ... False False False]\n",
            "[False False False ... False False False]\n",
            "[False False False ... False False False]\n",
            "[False False False ... False False False]\n",
            "[False False False ... False False False]\n",
            "[False False False ... False False False]\n",
            "[False False False ...  True False  True]\n",
            "[False False False ... False  True False]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-41-9dbf7b918b7a>:10: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.\n",
            "  labels[mask] = mode(digits.target[mask])[0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "4. Use the k-means algorithm in python to cluster the following 8 examples into 3 clusters: A1=(2,10), A2=(2,5), A3=(8,4), A4=(5,8), A5=(7,5), A6=(6,4), A7=(1,2), A8=(4,9). (a) Suppose that the centers of each cluster are A1, A4 and A7. Run the k-means algorithm for 3 epochs only. At the end of this epoch show:\n",
        "i. The new clusters (i.e. the examples belonging to each cluster)(mention the appro- priate attribute used to identify the clusters in sklearn)\n",
        "\n",
        "ii. The centers of the new clusters (mention the appropriate attribute used to identify the cluster centers in sklearn)"
      ],
      "metadata": {
        "id": "WnV0p7dlVahm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from sklearn.cluster import KMeans\n",
        "\n",
        "X = np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])\n",
        "centroid = np.array([[2,10],[5,8],[1,2]])\n",
        "\n",
        "kmeans = KMeans(n_clusters=3, init=centroid, max_iter=1).fit(X)\n",
        "print(\"cluster center for 1st iteration:\\n\", kmeans.cluster_centers_)\n",
        "\n",
        "c1 = kmeans.cluster_centers_\n",
        "plt.scatter(X[:,0], X[:,1])\n",
        "plt.scatter(c1[:,0], c1[:,1])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 458
        },
        "id": "VMwOtxzRVcGk",
        "outputId": "901092dc-983c-4bb5-ecc9-b29310edf8c4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "cluster center for 1st iteration:\n",
            " [[ 2.  10. ]\n",
            " [ 6.   6. ]\n",
            " [ 1.5  3.5]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:1362: RuntimeWarning: Explicit initial center position passed: performing only one init in KMeans instead of n_init=10.\n",
            "  super()._check_params_vs_input(X, default_n_init=10)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f81f9dfbd90>"
            ]
          },
          "metadata": {},
          "execution_count": 42
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZsAAAEJCAYAAABCNoqwAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAWyElEQVR4nO3dfVBU973H8Q9BIKjbLCKupcIquEqw2ghO5Wby0LQVdJxUY5qq07l2NNqCcVom9QlLkoZLqomMtRkN9XFiorYxkdqoHem0tS3EGI2kg2OSZrkojUN9AIMuilEX7h/JpeJTWDg/Dgffr5n+wY/wO9/TdX27hwMb1tDQ0CIAAAy6w+4BAAA9H7EBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAcY6Mjd/vt3uETuMcugfOwX5On1/iHNrDkbEBADgLsQEAGEdsAADGtSs2b731lqZNm6a7775bbrdbW7ZsafP5lpYWLV26VCkpKRo4cKAmTpyoDz74wMjAAADnaVdszp8/r9TUVC1btkzR0dHXff5Xv/qVVq9ereeff15/+ctfFBcXp0ceeUSBQMDygQEAztOrPf9RZmamMjMzJUlz585t87mWlhYVFxcrNzdXkyZNkiQVFxfL5/PpjTfe0MyZMy0e2dlqj/5Lp15ZpzvP1ukfd/XXgBlzFD8k0e6xAMCoTn/PpqamRidPntQ3v/nN1rXo6Gjde++9eueddzq7fY9Se/Rfinphge6rLtOY+g90X3WZol5YoNqj/7J7NAAwqtOxOXnypCQpLi6uzXpcXJxOnTrV2e17lFOvrJP3wsk2a94LJ3XqlXU2TQQAXaNdl9FM6OwPEDnxh6juPFt303Unno/kzMfhWpyD/Zw+v8Q5+Hy+W36+07HxeDySpNOnTyshIaF1/fTp0xowYECHB7sVv9/fqa+3yz/u6i/VX79+8a7+useB5+PUx+FqnIP9nD6/xDm0R6cvo3m9Xnk8Hu3du7d17eLFi3r77bc1duzYzm7fowyYMUc1vT1t1mp6ezRgxhybJgKArtGuVzaNjY2qrq6WJDU3N+v48eOqrKxUTEyMEhISlJOToxUrVsjn82no0KEqKipSnz599N3vftfo8E4TPyRRtQuXq/zzu9EucjcagNtEu2Lz3nvv6eGHH279eOnSpVq6dKmmT5+u4uJi/eQnP1FTU5MWLFighoYGpaenq6SkRC6Xy9jgThU/JFHxz/yP/H6/Iy+dAUBHtCs2999/vxoaGm76+bCwMOXl5SkvL8+quQAAPQi/Gw0AYByxAQAYR2wAAMYRGwCAccQGAGAcsQEAGEdsAADGERsAgHHEBgBgHLEBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAccQGAGBcu96pE+hpagKXVVgRUHVdlJJqzyg/zSWvK8LusYAei9jgtlMTuKzJpfU6GghKCtehc0169/Ql7ciKJTiAIVxGw22nsCLweWj+42ggqMKKgE0TAT0fscFt598XgjdcP3GTdQCdR2xw2/ly7/Abrg+8yTqAziM2uO3kp7k0xNU2LENc4cpPc9k0EdDzERvcdryuCO3IitVjSdFKvyuox5KiuTkAMIy70XBb8roitO7BfvL76+XzJdo9DtDj8coGAGAcsQEAGEdsAADGERsAgHHEBgBgHLEBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAccQGAGAcsQEAGEdsAADGERsAgHHEBgBgnCWxCQaDKiws1KhRo+TxeDRq1CgVFhbqypUrVmwPAHA4S94WeuXKlVq/fr2Ki4uVmpqqI0eOKCcnR5GRkVq4cKEVhwAAOJglsTlw4IDGjx+vCRMmSJK8Xq8mTJigQ4cOWbE9gBuoCVxWYUVA1XVRSqo9o/w0l7yuCLvHAm7IkstoGRkZKi8v10cffSRJ+vDDD1VWVqZx48ZZsT2Aa9QELmtyab1er27SoXPher26SZNL61UTuGz3aMANWfLKJjc3V42NjRo7dqzCw8N15coVzZ8/X7Nnz7ZiewDXKKwI6Ggg2GbtaCCowoqA1j3Yz6apgJsLa2hoaOnsJtu3b9fTTz+tgoICpaSk6PDhw1q8eLEKCgo0Y8aMG36N3+/v7GGB21Z2ZZQOnQu/bj39rqB+PfJTGybC7c7n893y85bEZsSIEZo3b55ycnJa15YvX66tW7fqvffe6+z21/H7/V94Yt0d59A9OPUc5vztjF6vbrpu/bGkaMe9snHqY3A1zuGLWfI9mwsXLig8vO2/ssLDw9Xc3GzF9gCukZ/m0hBX2+fcEFe48tNcNk0E3Jol37MZP368Vq5cKa/Xq5SUFFVWVmr16tWaNm2aFdsDuIbXFaEdWbGf3Y1W36ik2L7cjYZuzZLYvPDCC3ruuef005/+VHV1dfJ4PPrBD37Az9gABnldEVr3YD/5/fXy+RLtHge4JUti43K5tGzZMi1btsyK7QAAPQy/Gw0AYByxAQAYR2wAAMYRGwCAccQGAGAcsQEAGEdsAADGERsAgHHEBgBgHLEBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAccQGAGAcsQEAGEdsAADGERsAgHHEBgBgHLEBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAccQGAGAcsQEAGEdsAADGERsAgHHEBgBgHLEBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAccQGAGAcsQEAGGdZbE6cOKHs7GwlJyfL4/Fo7NixKi8vt2p7AICD9bJik4aGBmVlZSkjI0Pbtm1TbGysampqFBcXZ8X2AACHsyQ2L774ogYOHKg1a9a0rg0ePNiKrQH0UGGnahVZslFDaz9WVHyCLk2ZpZYB8XaPBUMsuYy2e/dupaena+bMmRo6dKjuu+8+rV27Vi0tLVZsD6CHCTtVq+jl8xXx9p/kqvmnIt7+k6KXz1fYqVq7R4MhlsTm2LFj2rBhgwYPHqzt27crOztbzz77rNatW2fF9gB6mMiSjbrjmrDc8fkrHfRMYQ0NDZ1++REXF6fRo0frj3/8Y+taQUGBdu3apQMHDtzwa/x+f2cPC8Chhr5aJFfNP69bD3iHq+q/59swETrL5/Pd8vOWfM/G4/Fo+PDhbdaGDRum48ePd3iwW/H7/Z36+u6Ac+geOAd7RMUnSDeIzZ3xCY47F8mZj8G1TJ+DJZfRMjIyVFVV1WatqqpKCQkJVmwPoIe5NGWWmq+5GaB5QLwuTZll00QwzZLYzJ07VwcPHlRRUZGqq6u1Y8cOrV27VrNnz7ZiewA9TMuAeDUtKNLl//q2At7huvxf31bTgiLuRuvBLLmMlpaWpi1btqigoEDLly/XoEGDtGTJEmID4KZaBsTr0+x8VfWAS1D4YpbERpKysrKUlZVl1XYAgB6E340GADCO2AAAjCM2AADjiA0AwDhiAwAwjtgAAIwjNgAA44gNAMA4YgMAMI7YAACMIzYAAOOIDQDAOGIDADCO2AAAjCM2AADjiA0AwDhiAwAwzrJ36kT71AQuq7AioOq6KCXVnlF+mkteV4TdYwHoAJ7P7UdsulBN4LIml9braCAoKVyHzjXp3dOXtCMrlj+ggMPwfA4Nl9G6UGFF4PM/mP9xNBBUYUXApokAdBTP59AQmy707wvBG66fuMk6gO6L53NoiE0X+nLv8BuuD7zJOoDui+dzaIhNF8pPc2mIq+0fxCGucOWnuWyaCEBH8XwODbHpQl5XhHZkxeqxpGil3xXUY0nRfDMRcCiez6HhbrQu5nVFaN2D/eT318vnS7R7HACdwPO5/XhlAwAwjtgAAIwjNgAA44gNAMA4YgMAMI7YAACMIzYAAOOIDQDAOGIDADCO2AAAjCM2AADjiA0AwDhiAwAwjtgAAIwjNgAA44gNAMA4YgMAMM5IbFasWCG3260FCxaY2B4A4DCWx+bgwYN6+eWXNWLECKu3BgA4lKWxOXv2rObMmaNVq1bJ7XZbuTWAHqYmcFlz/nZG2ZVRmvO3M6oJXLZ7pNtSVz0OlsYmNzdXkyZN0gMPPGDltgB6mJrAZU0urdfr1U06dC5cr1c3aXJpPcHpYl35OFgWm02bNqm6ulr5+flWbQmghyqsCOhoINhm7WggqMKKgE0T3Z668nHoZcUmfr9fBQUF2rNnjyIiItr9NZ09ptNxDt0D59D1quuiJIVfv17fKL+/vusHsoDTHgPJ2sfB5/Pd8vOWxObAgQOqr69XRkZG61owGNS+ffu0ceNG1dbWKioqKqTBbsXv93fq67sDzqF74BzskVR7RofONV2/HttXPl+iDRN1jhMfA6lrHwdLYjNx4kSNHj26zdoTTzyh5ORkPfnkk4qMjLTiMAB6iPw0l949fanNJZwhrnDlp7lsnOr205WPgyWxcbvd19191rt3b8XExCg1NdWKQwDoQbyuCO3IilVhRUDV9Y1Kiu2r/DSXvK72XYaHNbrycbAkNgAQKq8rQuse7Ce/v96Rl856iq56HIzFZvfu3aa2BgA4DL8bDQBgHLEBABhHbAAAxhEbAIBx3I12jbBTtYos2aiwT+rUEtNfl6bMUsuAeLvHAgBHIzZXCTtVq+jl83XHqdrWtfD/fV9NC4oIDgB0ApfRrhJZsrFNaCTpjs9f6QAAOo7YXCXsk7obrzc48xcDAkB3QWyu0hLT/8br7tgungQAehZic5VLU2ap+ZrvzTQPiNelKbNsmggAegZuELhKy4B4NS0o+uxutIZ6tbhjuRsNACxAbK7RMiBen2bzbqMAYCUuowEAjCM2AADjiA0AwDhiAwAwjtgAAIwjNgAA44gNAMA4YgMAMI7YAACMIzYAAOOIDQDAOGIDADCO2AAAjCM2AADjiA0AwDhiAwAwjtgAAIwjNgAA44gNAMA4YgMAMI7YAACMIzYAAOOIDQDAOGIDADCO2AAAjCM2AADjiA0AwDhiAwAwjtgAAIwjNgAA4yyJzYoVK/TQQw8pISFBycnJmjp1qt5//30rtgYA9ACWxKa8vFyPP/64SktL9eabb6pXr16aPHmyPvnkEyu2BwA4XC8rNikpKWnz8Zo1a5SYmKj9+/drwoQJVhxCklQTuKzCioCq66KUVHtG+WkueV0Rlu0PADDDkthcq7GxUc3NzXK73ZbtWRO4rMml9ToaCEoK16FzTXr39CXtyIolOADQzRm5QWDx4sUaOXKkvv71r1u2Z2FF4PPQ/MfRQFCFFQHLjgEAMCOsoaGhxcoNlyxZopKSEu3Zs0eDBw++6X/n9/tD2je7MkqHzoVft55+V1C/HvlpqGMCACzk8/lu+XlLL6Pl5eWppKREO3fuvGVopC8e7FpJtWd06FzT9euxfeXzJYa0V3fg9/tD/v+gu+Ecugenn4PT55c4h/aw7DLaokWLtH37dr355psaNmyYVdu2yk9zaYir7SubIa5w5ae5LD8WAMBalryymT9/vl577TVt3rxZbrdbJ0+elCT16dNHffv2teIQ8roitCMr9rO70eoblRTbl7vRAMAhLInN+vXrJUmTJk1qs75o0SLl5eVZcQhJnwVn3YP95PfXO/LSGQDcriyJTUNDgxXbAAB6KH43GgDAOGIDADCO2AAAjLP8hzoBALgWr2wAAMYRGwCAccQGAGAcsQEAGEdsAADGOSo2b731lqZNm6a7775bbrdbW7ZssXukkKxYsUIPPfSQEhISlJycrKlTp+r999+3e6yQrFu3Tvfee68SEhKUkJCgcePGqbS01O6xOmzFihVyu91asGCB3aO029KlS+V2u9v8z8QvvzXtxIkTys7OVnJysjwej8aOHavy8nK7x2q3kSNHXvc4uN1ufe9737N7tHYLBoMqLCzUqFGj5PF4NGrUKBUWFurKlSuWH8vIO3Wacv78eaWmpmr69OnKzs62e5yQlZeX6/HHH1daWppaWlr0i1/8QpMnT9Y777yjmJgYu8drl/j4eD377LNKTk5Wc3OzfvOb3+j73/++/vrXv+qrX/2q3eOF5ODBg3r55Zc1YsQIu0cJmc/n065du1o/Dg+//r2eurOGhgZlZWUpIyND27ZtU2xsrGpqahQXF2f3aO22d+9eBYP/eUPHEydO6Bvf+IYmT55s31AhWrlypdavX6/i4mKlpqbqyJEjysnJUWRkpBYuXGjpsRwVm8zMTGVmZkqS5s6da/M0oSspKWnz8Zo1a5SYmKj9+/drwoQJNk0VmokTJ7b5+KmnntKGDRt08OBBR8Xm7NmzmjNnjlatWqXnn3/e7nFC1qtXL3k8HrvH6LAXX3xRAwcO1Jo1a1rXvug9sLqb/v37t/n41Vdflcvl0iOPPGLTRKE7cOCAxo8f3/r3j9fr1YQJE3To0CHLj+Woy2g9TWNjo5qbm+V2u+0epUOCwaC2b9+u8+fPW/oW4F0hNzdXkyZN0gMPPGD3KB1y7NgxpaSkaNSoUZo1a5aOHTtm90gh2b17t9LT0zVz5kwNHTpU9913n9auXauWFmf+jHlLS4teffVVTZ06VdHR0XaP024ZGRkqLy/XRx99JEn68MMPVVZWpnHjxll+LEe9sulpFi9erJEjRzruL+ojR44oMzNTFy9eVJ8+fbR582ZHXYratGmTqqurtXbtWrtH6ZAxY8bopZdeks/nU11dnZYvX67MzEzt379f/fr1s3u8djl27Jg2bNiguXPnKjc3V4cPH9aiRYskST/84Q9tni50e/fuVU1NjWbMmGH3KCHJzc1VY2Ojxo4dq/DwcF25ckXz58/X7NmzLT8WsbHJkiVLtH//fu3Zs8dx19t9Pp/Kysp07tw5/f73v1dOTo527dql1NRUu0f7Qn6/XwUFBdqzZ48iIpz5xnvX/qtzzJgxuueee7R161bNmzfPpqlC09zcrNGjR+uZZ56RJH3ta19TdXW11q9f78jYbNq0SWlpaRo5cqTdo4SkpKREv/3tb7V+/XqlpKTo8OHDWrx4sRITEy0PJ7GxQV5enkpKSrRz507HXaeWpMjISCUlJUmS7rnnHlVUVOill17SqlWrbJ7six04cED19fXKyMhoXQsGg9q3b582btyo2tpaRUVF2Thh6Pr27auUlBRVV1fbPUq7eTweDR8+vM3asGHDdPz4cZsm6rjTp0/rD3/4g4qKiuweJWRPP/205s2bp0cffVSSNGLECH388cf65S9/SWycbtGiRfrd736nnTt3OvJ21Rtpbm7WpUuX7B6jXSZOnKjRo0e3WXviiSeUnJysJ598UpGRkTZN1nEXL16U3+/X/fffb/co7ZaRkaGqqqo2a1VVVUpISLBpoo7bunWroqKiWv/CdpILFy5cd2UlPDxczc3Nlh/LUbFpbGxs/ddbc3Ozjh8/rsrKSsXExDjiD+n8+fP12muvafPmzXK73Tp58qQkqU+fPurbt6/N07XPz3/+c2VmZuorX/mKGhsb9cYbb6i8vFzbtm2ze7R2+f+fhbha7969FRMT44jLgJKUn5+v8ePHa9CgQa3fs7lw4YKmT59u92jtNnfuXGVmZqqoqEhTpkxRZWWl1q5dq6eeesru0ULS0tKiV155RVOmTHHMc/hq48eP18qVK+X1epWSkqLKykqtXr1a06ZNs/xYjnqLgbKyMj388MPXrU+fPl3FxcU2TBSam911tmjRIuXl5XXtMB2Uk5OjsrIynTp1Sl/60pc0YsQI/fjHP9a3vvUtu0frsIkTJyo1NVXLly+3e5R2mTVrlvbt26f6+nr1799fY8aM0c9+9jOlpKTYPVpISktLVVBQoKqqKg0aNEhz5szRj370I4WFhdk9Wrv9/e9/13e+8x39+c9/Vnp6ut3jhCwQCOi5557Trl27VFdXJ4/Ho0cffVQLFy7UnXfeaemxHBUbAIAz8XM2AADjiA0AwDhiAwAwjtgAAIwjNgAA44gNAMA4YgMAMI7YAACMIzYAAOP+D+Oezd9hC35cAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#2nd iteration\n",
        "kmeans = KMeans(n_clusters=3, init=c1, max_iter=1).fit(X)\n",
        "print(\"\\ncluster center after 2nd iteration:\\n\", kmeans.cluster_centers_)\n",
        "\n",
        "c2 = kmeans.cluster_centers_\n",
        "plt.scatter(X[:,0], X[:,1])\n",
        "plt.scatter(c2[:,0], c2[:,1])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 476
        },
        "id": "0fMEYGe2VhFo",
        "outputId": "efeee8d5-0af6-44f0-bf25-b984be9b759b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "cluster center after 2nd iteration:\n",
            " [[3.   9.5 ]\n",
            " [6.5  5.25]\n",
            " [1.5  3.5 ]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:1362: RuntimeWarning: Explicit initial center position passed: performing only one init in KMeans instead of n_init=10.\n",
            "  super()._check_params_vs_input(X, default_n_init=10)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f81f9d08250>"
            ]
          },
          "metadata": {},
          "execution_count": 43
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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AwcOKBgMKjc3t20tHA5r37592rBhgxobG5WYmGjjhLHr27evsrKyFAgE7B6lw7xer+655552a8OHD9eJEydsmuj2nTlzRn/6059UUVFh9ygxW7x4sebMmaPHH39ckjRixAh9+OGH+vWvf01snG7BggV68803tWPHDkdernojkUhEly5dsnuMDpk0aZJGjx7dbu2ZZ55RZmamnn32WSUkJNg02e27ePGi/H6/HnzwQbtH6bDc3FwdPXq03drRo0eVlpZm00S3b8uWLUpMTGz7C9tJLly4cN2ZFZfLpUgkYvltOSo2LS0tbf96i0QiOnHihOrr65WcnOyIP6Rz587Va6+9pk2bNsnj8ejUqVOSpD59+qhv3742T9cxP//5z5WXl6cvf/nLamlp0RtvvKGamhpt3brV7tE65P9/F+Jqd911l5KTkx1xGlCSSktLNWHCBN19991tr9lcuHBB06dPt3u0Dps9e7by8vJUUVGhKVOmqL6+XmvWrNFzzz1n92gxiUaj+t3vfqcpU6Y45jF8tQkTJmjFihVKT09XVlaW6uvrtWrVKk2bNs3y23LURwzs3btXjz766HXr06dPV2VlpQ0TxeZmV50tWLBAJSUlXTvMbSoqKtLevXt1+vRpffGLX9SIESP0ox/9SN/61rfsHu22TZo0SdnZ2Vq2bJndo3TIjBkztG/fPgWDQfXv319jx47VT3/6U2VlZdk9Wkyqq6tVVlamo0eP6u6779asWbP0wx/+UHFxcXaP1mF///vf9e1vf1t/+ctfNGbMGLvHiVkoFNILL7ygnTt3qqmpSV6vV48//rjmz5+vL3zhC5belqNiAwBwJn7PBgBgHLEBABhHbAAAxhEbAIBxxAYAYByxAQAYR2wAAMYRGwCAccQGAGDc/wHeTQqNW6JkHwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#3rd iteration\n",
        "kmeans = KMeans(n_clusters=3, init=c2, max_iter=1).fit(X)\n",
        "print(\"\\ncluster center after 3rd iteration:\\n\", kmeans.cluster_centers_)\n",
        "\n",
        "c3 = kmeans.cluster_centers_\n",
        "plt.scatter(X[:,0], X[:,1])\n",
        "plt.scatter(c3[:,0], c3[:,1])"
      ],
      "metadata": {
        "id": "qnH2ctSKVj6q",
        "outputId": "d33ff771-2bca-404e-ba42-906ff2003f30",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 476
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "cluster center after 3rd iteration:\n",
            " [[3.66666667 9.        ]\n",
            " [7.         4.33333333]\n",
            " [1.5        3.5       ]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/sklearn/cluster/_kmeans.py:1362: RuntimeWarning: Explicit initial center position passed: performing only one init in KMeans instead of n_init=10.\n",
            "  super()._check_params_vs_input(X, default_n_init=10)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f81f9c83580>"
            ]
          },
          "metadata": {},
          "execution_count": 44
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}
