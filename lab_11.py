{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOY/NMFk6XvBWneiISjY6F1",
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
        "<a href=\"https://colab.research.google.com/github/vivek201102/ML-Labs/blob/master/Lab_11.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Part A\n",
        "#### Basic SVM with kernel"
      ],
      "metadata": {
        "id": "sDLJ6596PX9b"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "G3b37VaGPUfD"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "from sklearn import svm\n",
        "from sklearn.model_selection import train_test_split\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# importing scikit learn with make_blobs\n",
        "from sklearn.datasets import make_blobs\n",
        "# creating datasets X containing n_samples\n",
        "# Y containing two classes\n",
        "X, Y = make_blobs(n_samples=40,n_features=2,centers=2)\n",
        "print(X.shape)\n",
        "print(Y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "erZ6YgHzPpZq",
        "outputId": "a91f5f49-4374-44c4-e05a-2f2fb886b9f0"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(40, 2)\n",
            "[0 0 0 0 1 1 1 1 0 0 1 0 1 0 0 0 1 1 1 0 1 0 0 0 1 1 1 0 1 1 0 1 1 1 0 1 1\n",
            " 0 0 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# plotting scatters\n",
        "plt.scatter(X[:,0],X[:,1],c=Y,s=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "EQ1R5eH8PrF0",
        "outputId": "6325d569-e1e9-4b43-86f4-df5354a3ffb1"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f0de48bd610>"
            ]
          },
          "metadata": {},
          "execution_count": 3
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXkAAAD4CAYAAAAJmJb0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAcUElEQVR4nO3deZwcdZnH8c/T3XMnJBlyEXJxI4cgDKgIIvfhEVFZAXcFwxJQ5BBZFkWz7CICBmFFRYwoAosEDwIsgpAsCB4ITDCBBBIcJJAEcpEQkszR3dPP/tGVsTP0JDPprq5Jzff9es0rXV2/rt8z1ZVvV/+qpsrcHRERiadE1AWIiEh4FPIiIjGmkBcRiTGFvIhIjCnkRURiLBV1AYWGDx/uEydOjLoMEZHtypw5c1a7+4hi8/pVyE+cOJHm5uaoyxAR2a6Y2Ws9zdNwjYhIjCnkRURiTCEvIhJjCnkRkRhTyIuIxJhCXkQkxhTyIhJr7s5d3/o1/7Lb+Vz56WlsfKc16pIqSiEvIrHW/MhcZlx3H8tfXckzv32O6ZfeEXVJFaWQF5FYW/n6ajyXv29GJp1l2SvLI66oshTyIhJrh006hNpBtdTvUEdNfTX/dOmkqEuqqH51WQMRkXIbNmooP3vpv3npqZfZec8xjN1jp6hLqiiFvIjE3g6Ng3n/Rw+OuoxIaLhGRCTGFPIiIjGmkBcRiTGFvIhIjCnkRURiTCEvIhJjCnkRkRhTyIuIxJhCXkQkxhTyIiJ9lM1kWbxgCRve3hh1KVulyxqIiPRB6/o2zj/0clYvewsz47pHp/Ke9+8RdVk90p68iEgv/fWxF7jyU9NY/uoK2jd00La+ndunzoi6rC3SnryISC8s+PMivvnxa+loS3c9l0wlGDSsIcKqtk578iIivfDCky+SzWS7phMJY+J+4znvhrOiK6oXQg95MzvRzBaZWYuZXR52fyIiYdjviPeQqsoPftTW13DBzedwy3PTGD6mMeLKtszcPbyFmyWBl4HjgKXAs8Dp7v5isfZNTU3e3NwcWj0iIqV47v9e4M/3Pc2IccN56emXqWuo4+xrPxd50JvZHHdvKjYv7DH5Q4EWd/97UMgMYBJQNORFRPqzg47Zn30P25PTdj6Xjes2kkgmeHnO3/npghujLq1HYQ/X7AwsKZheGjwnIrJdWrP8bTLpDO7Qmc2x9OU3oi5piyI/8GpmU8ys2cyaV61aFXU5IiJbNHL8cEZNGEF1XTW1DTW8/+SDSl5mZ2cnYQ2dhz1cswwYVzA9Nniui7tPB6ZDfkw+5HpEREqSTCa56alv89gv/khNXTVHn3H4Ni/L3fn+l2/lt9Nn0zCknmsevoK9Dtm9jNWGvyf/LLCHme1iZtXAacADIfcpIhKqhh3q+fh5x3P8mR/pOuNmWyx6toVZdzxBrjPH+jUbuH7yzWWsMi/UPXl3z5rZl4FHgCTwM3dfEGafIiLbi2ymE8y6pjPp7BZab5vQx+Td/SF339Pdd3P3q8PuT0Rke7HPB/fk4OPeS6o6RU19DRfefE7Z+9BlDUREIpJIJLjyN//G2pXrqB9cS01dTdn7UMiLiERs2MghoS078lMoRUT6u7Ur19G2oS3qMraJQl5EpAfuzrTJP+Rz48/jMyPP5g/3Pt01L5fL8dOv38U5+1/CLZf8nM5sZ4SV9kwhLyIDTmdnJ1effiMnVH2Ws/a6kBWvFf9DzMULlvDEL/9MJp0l3Z7h++f/pGveI7c9zsybHmLxgiU8OH0W9/3g4UqV3ycKeREZcP408xn+8uAccp053nxlOT/6ym1F21VVpyj8Q9RUVYo1y9dy7vsu5cYpP6ajNX9t+Y7WNEsWLiu6jKgp5EVkwCm88Ucu57Rt6CjabuyeYzj1qx8nmUrQMKSef7/zAn7y7//DawuWdl2GoLq2itr6Gk74wtEVqb2vFPIiMuAc8ekPMGb30dTUVVO/Qx1nX3NGj23P+q/T+G3bL5i55ucccOS+bFzX2jX+Xl1TxdFnHMEtc6f12/u86hRKERlwautruLn5Ola+vpqhI4dQ11C7xfbJZLLr8Zn/+Vmef+JFOrOd7DimkfO++3kahuRvAbjitVVU11YxbNTQMMvvE4W8iAxIyWSSnXYZ1efX7XbARO5ecgtvvbGW0buM7Lp2zY3n/pjZdz6Bu3Pu9Wcy6fwTy13yNtFwjYhIH9UNqmPsnmO6An7V0reYdecTpNszZDqyTL/szogr/AftyYvIgJFJZ1j07Cs0jh7KmN1Gl7w8d2f2nU+yqLkF7/zHaTg1ddUlL7tcFPIiMiCkOzJc+MGv80bLcnKdOS66ZQrH/cuRJS3zrm/9mhnX3U9Hawep6hTVdVXU1NXwjXsuKVPVpdNwjYgMCPP/uJA3XllO24Z2OtrS3P4f95S8zKceaKajNX/6ZaoqxUU3T+He1bdx0DH7l7zsctGevIgMCMNG7kCuMweAJYwddxpW8jIPOu69vPbSMjpaO3DPsdchuwHQ3trB98+/lYXPtnD8mR+huqaK3932GHsevBvn3zSZ2vryX22yJwp5ERkQdtl/Amdf8zlmXDOTHXdu5PL/ubDkZZ511Wk0jh7Kqy+8zvFnHcWEffJ3O73tG3fz+3v+RLo9k//G4JDpyLB00RvUNtRw/vcml9x3bynkRWTAOOWCkznlgpPLtrxkMskpF370Xc+/9uJS0u0ZADqznVhw96d0e4bF85eUrf/e0Ji8iEiZnXLhydTUV1M3uJba+hpq6qqpG1RLTX01n7zgpIrWoj15EZEye//JB/HDZ65l8YKl7H/E3rjDvN8vYMI+Y9ntgIkVrUUhLyISggn7jOsaowc4+vTDI6lDwzUiEhsrl6zmvh88zNMPPdd1lciBTnvyIhILa1e8zbkHXEq6PU0imeCfv/kZPnvZJ6MuK3LakxeRWHj+yZfIdeZIt2do39jBo3c8EXVJ/YJCXkRiYeK+Y+nsDK7zXlvFXk27RVxR/6CQF5FYmLDPOKb+6lIOPv4APnrucVx48zmh9PPg9FmcOvpfOWf/S3i9n97yr5D1p4MTTU1N3tzcHHUZIiJFLWt5kykHXEq6LY0ZTNxvPNPnfReAlr++ygM/eoRRE0ZwwheO4qkHmhkyfDCHf+r9JBLh7k+b2Rx3byo2TwdeRUR6ad3q9SSS+cB2z19HHmD1G2u45MiptG1op6o2xd3XzAR3LGH89bH5XBTSt4reCO3jxcymmdlCM3vezGaa2dCw+hIRqYS9mnajYXBd13TbhnZa5r7Kq8+/1hX+mfYs6bYOOtrStG/s4Pf3/CmqcoFwx+RnAfu5+3uBl4GvhdiXiAxw7s70y+5g0tDPc+6Bl7Ly9VVl7yOZStI45h9Xr6yqTvHK3MXsftCuuDtmRnVdFYngnrCpqiS77D++7HX0RWjDNe7+aMHkX4DPhNWXiMjcx+fzvz96lPaNHSye/zo3nvtjrnn4G2Xv55jPHcGShcvIdGQwM9575D4MGzmEHz5zLbPufIKR44Yz/j07c/c1Mxk2eijnTvt82Wvoi0qNyU8Gil6h38ymAFMAxo+P9hNPRLZfG9Zu7LraYy7nrFu1PpR+Pn3xx9hp11EsWfgGh01q6roZ+Ng9x/CFq07varf/EfuE0n9flRTyZjYbKHajxCvc/f6gzRVAFrir2DLcfTowHfJn15RSj4gMXIec9D5GTRzBisWryOWcs685I7S+DvvEIfCJ0BZfViWFvLsfu6X5ZnYW8DHgGO9P52qKSOzU1tfwoznfYfGCJew4ppFhI4dEXVK/ENpwjZmdCFwGHOnurWH1IyKySaoqxe4H7hJ1Gf1KmGfX/AAYDMwys7lmdkuIfYmISBFhnl2ze1jLFhGR3tG1a0REYkwhLyISYwp5EZEYU8iLiMSYQl5EJMYU8iIiMaaQFxGJMYW8iEiMKeRFRGJMIS8iEmMKeRGRGFPIi4jEmEJeRCTGFPIiIjGmkBcRiTGFvIhIjCnkRURiTCEvIhJjCnkRkRhTyIuIxJhCXkQkxhTyIiIxppAXEYkxhbyISIwp5EVEYkwhLyISY6GHvJl91czczIaH3ZeIiGwu1JA3s3HA8cDrYfYjIiLFhb0nfyNwGeAh9yMiIkWEFvJmNglY5u7zttJuipk1m1nzqlWrwipHRGRASpXyYjObDYwuMusK4Ovkh2q2yN2nA9MBmpqatMcvIlJGJYW8ux9b7Hkz2x/YBZhnZgBjgefM7FB3X15KnyIi0nslhXxP3P0FYOSmaTNbDDS5++ow+hMRkeJ0nryISIyFsiffnbtPrEQ/IiKyOe3Ji4jEmEJeRCTGFPIiIjGmkBcRiTGFvIhIjCnkRURiTCEvIhJjCnkRkRhTyIuIxJhCXkQkxhTyIiIxppAXEYkxhbyISIwp5EVEYkwhLyISYwp5EZEYU8iLiMSYQl5EJMYU8iIiMaaQFxGJMYW8iEiMKeRFRGJMIS8iEmMKeRGRGFPIi4jEmEJeRCTGQg15M7vAzBaa2QIz+06YfYmIyLulwlqwmR0FTAIOcPcOMxsZVl/duacBw6yqUl2KiPRLYe7JfxG41t07ANx9ZYh9dcltvANfcSC+4kByrb+pRJciIv1WmCG/J3CEmT1tZk+Y2SHFGpnZFDNrNrPmVatWldSh5zbC+uuALJCBd6binu25vafxtnvx1hn514qIxExJwzVmNhsYXWTWFcGyG4EPAIcAvzSzXd3dCxu6+3RgOkBTU5N3X1CYfO2XIPMsuEPrXbDj/ZjpWLSIxEdJIe/ux/Y0z8y+CNwbhPozZpYDhgOl7a5vgSUa8MFfhfXXAwY7TMWs+K/onoP0H4DgcyW7GHIrIbn5Z5Znl0LHY5CagNUcGVbpIiKhCO3AK3AfcBTwuJntCVQDq0PsD4BEw2S8/nTyB15re2xnlsCT46FzCeBgdZBo3KyNd67A35oE3gGWxAddRKJhcri/gIhIGYU5NvEzYFczmw/MAM7sPlQTFrO6LQZ8V7vGO6H2ZKg5FtvxF5hVb94g/Qx4J5AGb4O2+0KpV0QkLKHtyXv+PMZ/Dmv55WDJ0djQG3pukNoDyAUTNVD13kqUJSJSNmEO12z3rGpvGPo9vPVOSO2BDb4k6pJERPpEIb8VVnsUVntU1GWIiGwTnS8oIhJjCnkRkRhTyIuIxJhCvgfe8QS+8ad49pXyLzvzMt7+GJ5bX/Zli4gUivWBV8+tA+/Akn27AGZu492w/logAxu+n7/cQWpCWWrKtd4P73wTLAk2GIY/iCV2KMuyRUS6i+2efK71Xnzlh/BVR5NbN7VvL26/H2gDsuA5SD9VvsJabwXawTeCr4f0n8q3bBGRbmIb8qy/Ckjnf9pm4tklvX9t1cHApr+YNUi9p2gzz7WS23ArufU34Z29vGJDcjxdX6C8ExJjel+XiEgfxXi4phrYdPlgh+6XLNgCG3wxbjWQXYDVnYpVH1C0na89BzLzgBze9hsYMXurNyqxId/C374cOl+F+jN7XLaISDnENuRt6A342xeCt8Pgr2DJUb1/rVVhgy/cesPMHLoue+BvQ+dySI3b8rITw7DGH/e6FhGRUsR2uMZqPkRi1BwY+Ves6n149rXyd5LaF6gCDEjiNqj8fYiIlCC2IQ/gnoE1Z+BrJ+OrP06u9VfbsAzHO98K7hu7OWu8DareByTy4+trPoN7WxkqFxEpj1iHPJl50NkC3gq0w4bv9enl7ll87Vn4qg/jKz+Ap+dtNt8SO0Dna0An0Aa5NVDQxt2p0NWVRUSKinfIJxrzp0ACYJAY3rfXdzwZHFjNgG/A13/r3W2SE4Bk/rF3QjJ/tkxu4y/wFfvlbyjeNmtbfwMRkZLEOuQttSsMvhwSIyC1Nzb0v/u6gMIJih2ntqE3QvWHIbU3DJmGpcbn/5J1/dVABmiDdV8h99Zp5FZ/Ck8/t+2/kIhIH8X27JpNEg1nQMMZ2/bi6sOh5mho/y3YMEjtTW7VMZDaFxvybSwxCEuOKHK2TLbbdBoy+XD3tZNh5NOY1WxbTSIifRDrPflSmSVIDL0BGzUfhnwb2u7N3xO24zF8/fWbtc1tvJ3c8gPIrfwgZP8ODZPJn3mTIv8tIOBZyL1TyV9DRAaw2O/Jl4NZFeTexLtuBZiGzsVd873zTVh/PdABuTb87YtJjPwD3vAFIIm/czV0PJpvXHVA348NiIhsI4V8b9UcC3YTUAWexRr+9R/zvJXN99ZbAbBEY356yHWQngSehpojMCtoKyISIoV8L1lyFIz4HaTnQmrXza9KmdwVao+F9uAsmsGXb/5aM6j5UOWKFREJxCLkPbcWsq9Dancs0RBaP5YYBkXu92pmMOS7MGgJWAOW3DG0GkRE+mK7D3nPvIiv+RxgYHWw431YckTF6zAzSI2veL8iIluy3Z9d4xunB9dm3wC5t6H9gahLEhHpN7b7kM+fqbLp8r6p/PnsIiICxCDkbdBFUN2Uv5Ve7QlQNynqkkRE+o3QxuTN7EDgFvK3WMoCX3L3Z8reT2Iw1nh7uRcrIhILYe7Jfwf4T3c/EJgaTIuISAWFGfIO7BA8HgK8EWJfkXFPk3vnanJvnUGu9f6oyxER2UyYp1BeDDxiZteT/zA5rFgjM5sCTAEYP377OwXR10+D1hlAB2QW4KmxWPXBUZclIgKUuCdvZrPNbH6Rn0nAF4GvuPs44CvAT4stw92nu3uTuzeNGFH589tLlpkPdAQTDtm/RVmNiMhmStqTd/dje5pnZncAFwWTvwJuLaWvfqvus5B5EfD89edrjoi6IhGRLmEO17wBHAn8HjgaiOUubqL+k3hqLGRboPpwLLlz1CWJiHQJM+TPAb5nZimgnWDcPY6suil/rn4fuDv4O2CDMEuGVJmIDHShhby7/xGI/Aik59ZCxx8gOSYfxv2Aezu+5vP58fzEUGi8e/OrWoqIlMl2/xevW+K5d/DVH8PXTcXXnE1uYz/5o6m2/4XMQiALuTXvusuUiEi5xDrkST8b3MCjFWiD1ruirggA7/gTm52RE/O3QUSiE+90SY4H7wwmqiC1e6TlAHjH09DxOPlwB6jBBl8aZUkiEmOxDnmr2iN/673UflB7AjbkmqhLgs5lm09X7YWlxkVTi4jE3nZ/05CtSdSdBHUnRV3GP9R+BDZMA0/kv2XUnxN1RSISY7EP+f7GEo0w/GFIN0NyQv7bhohISBTyEbDE0PyNv0VEQhbrMXkRkYFOIS8iEmMKeRGRGFPIi4jEmEJeRCTGFPIiIjGmkBcRiTGFvIhIjCnkRURiTCEvIhJjCnkRkRhTyIuIxJhCXkQkxhTyIiIxppAXEYkxhbyISIwp5EVEYkwhLyISYwp5EZEYKynkzexUM1tgZjkza+o272tm1mJmi8zshNLKFBGRbVHqnvx84FPAk4VPmtk+wGnAvsCJwM1mliyxr37JvQPPvo57JupSRETeJVXKi939JQAz6z5rEjDD3TuAV82sBTgUeKqU/vobz76Ov3UqeDskG2HH32CJxqjLEhHpEtaY/M7AkoLppcFzseIbfwz+NtAGnSuh9ddRlyQispmt7smb2WxgdJFZV7j7/aUWYGZTgCkA48ePL3VxlWV1QBLIAgmw2ogLEhHZ3FZD3t2P3YblLgPGFUyPDZ4rtvzpwHSApqYm34a+ImODvoynn4PsS1DdBPWfjbokEZHNlDQmvwUPAL8wsxuAMcAewDMh9RUZSwzFht8bdRkiIj0q9RTKU8xsKfBB4Ldm9giAuy8Afgm8CPwOON/dO0stVkRE+qbUs2tmAjN7mHc1cHUpyxcRkdLoL15FRGJMIS8iEmMKeRGRGFPIi4jEmEJeRCTGzL3//P2Rma0HFkVdxxYMB1ZHXcRW9PcaVV9pVF9p4lrfBHcfUWxGWH8Mta0WuXvT1ptFw8ya+3N90P9rVH2lUX2lGYj1abhGRCTGFPIiIjHW30J+etQFbEV/rw/6f42qrzSqrzQDrr5+deBVRETKq7/tyYuISBkp5EVEYqziIW9mp5rZAjPLmVlTt3lfM7MWM1tkZif08PpdzOzpoN09ZlYdYq33mNnc4Gexmc3tod1iM3shaNccVj099H2lmS0rqPPkHtqdGKzXFjO7vIL1TTOzhWb2vJnNNLOhPbSr6Drc2vows5rg/W8JtreJYddU0Pc4M3vczF4M/q9cVKTNR8xsXcH7PrVS9QX9b/H9srybgvX3vJkdVMHa9ipYL3PN7B0zu7hbm4quPzP7mZmtNLP5Bc81mtksM/tb8O+wHl57ZtDmb2Z2Zp87d/eK/gDvAfYCfg80FTy/DzAPqAF2AV4BkkVe/0vgtODxLcAXK1T3d4GpPcxbDAyv9LoM+r4SuHQrbZLB+twVqA7W8z4Vqu94IBU8vg64Lup12Jv1AXwJuCV4fBpwTwXf052Ag4LHg4GXi9T3EeDBKLa53rxfwMnAw4ABHwCejqjOJLCc/B8LRbb+gA8DBwHzC577DnB58PjyYv83gEbg78G/w4LHw/rSd8X35N39JXcv9letk4AZ7t7h7q8CLcChhQ3MzICjgU13zL4d+GSI5Rb2+0/A3WH3FZJDgRZ3/7u7p4EZ5Nd36Nz9UXfPBpN/IX8ryKj1Zn1MIr99QX57OybYDkLn7m+6+3PB4/XAS8DOlei7jCYBd3jeX4ChZrZTBHUcA7zi7q9F0HcXd38SWNPt6cJtrKcsOwGY5e5r3H0tMAs4sS9996cx+Z2BJQXTS3n3hr0j8HZBaBRrE4YjgBXu/rce5jvwqJnNCW5MXmlfDr4S/6yHr3y9WbeVMJn83l0xlVyHvVkfXW2C7W0d+e2vooJhovcBTxeZ/UEzm2dmD5vZvpWtbKvvV3/Z5k6j552zKNcfwCh3fzN4vBwYVaRNyesxlMsamNlsYHSRWVe4+/1h9Lmtelnr6Wx5L/5wd19mZiOBWWa2MPjkDr1G4EfAVeT/011Fflhpcrn67o3erEMzuwLIAnf1sJhQ1+H2yMwGAb8BLnb3d7rNfo78EMSG4DjMfeTvpVwp/f79Co7XfQL4WpHZUa+/zbi7m1ko57OHEvLufuw2vGwZMK5gemzwXKG3yH/tSwV7V8Xa9MnWajWzFPAp4OAtLGNZ8O9KM5tJfjigbBt8b9enmf0EeLDIrN6s223Wi3V4FvAx4BgPBhqLLCPUddhNb9bHpjZLg21gCPntryLMrIp8wN/l7u+6W3xh6Lv7Q2Z2s5kNd/eKXHyrF+9XqNtcL50EPOfuK7rPiHr9BVaY2U7u/mYwlLWySJtl5I8fbDKW/PHMXutPwzUPAKcFZzXsQv5T9ZnCBkFAPA58JnjqTCDsbwbHAgvdfWmxmWbWYGaDNz0mf6BxfrG2Yeg2znlKD30/C+xh+TOTqsl/hX2gQvWdCFwGfMLdW3toU+l12Jv18QD57Qvy29tjPX1AlVsw9v9T4CV3v6GHNqM3HSMws0PJ/1+uyIdQL9+vB4DPB2fZfABYVzA0USk9fgOPcv0VKNzGesqyR4DjzWxYMBR7fPBc71Xq6HLB0eJTyI8rdQArgEcK5l1B/qyHRcBJBc8/BIwJHu9KPvxbgF8BNSHX+3PgvG7PjQEeKqhnXvCzgPwQRSXX553AC8DzwUazU/cag+mTyZ+l8UolawzepyXA3ODnlu71RbEOi60P4L/IfxgB1AbbV0uwve1awXV2OPnht+cL1tvJwHmbtkXgy8G6mkf+gPZhFayv6PvVrT4Dfhis3xcoOJOuQjU2kA/tIQXPRbb+yH/YvAlkgvw7m/wxnv8D/gbMBhqDtk3ArQWvnRxshy3AF/raty5rICISY/1puEZERMpMIS8iEmMKeRGRGFPIi4jEmEJeRCTGFPIiIjGmkBcRibH/B9QhLGNEQMpvAAAAAElFTkSuQmCC\n"
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
        "# Split data to train and test on 80-20 ratio\n",
        "X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.8)\n",
        "print(X_train.shape)\n",
        "print(X_test.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PNySchKDPtLn",
        "outputId": "bf604d08-8eab-4dc9-b4c4-cd82130d4048"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(32, 2)\n",
            "(8, 2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a linear SVM classifier\n",
        "clf = svm.SVC(kernel='linear')\n",
        "# Train classifier\n",
        "clf.fit(X_train,y_train)\n",
        "\n",
        "# # Plot decision function on training and test data\n",
        "# plot_decision_function(X_train, y_train, X_test, y_test, clf)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "6b0-6NgQP0Am",
        "outputId": "bed8afa3-7196-4abb-9f87-2c25db1198a5"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVC(kernel='linear')"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC(kernel=&#x27;linear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC(kernel=&#x27;linear&#x27;)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "clf_predictions = clf.predict(X_test)\n",
        "print(\"Accuracy: {}%\".format(clf.score(X_test, y_test) * 100 ))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9y0VJqu8P1Vc",
        "outputId": "6fea7727-b9cc-4495-c74e-cf8e71ea6d34"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 100.0%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "print(\"Accuracy: \",metrics.accuracy_score(y_test, clf_predictions))\n",
        "print(\"Precision: \",metrics.precision_score(y_test, clf_predictions))\n",
        "print(\"Recall: \",metrics.recall_score(y_test, clf_predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Aa4jB_gMP5ke",
        "outputId": "08d20d6d-845d-4ed0-902b-02d6f3ad3784"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy:  1.0\n",
            "Precision:  1.0\n",
            "Recall:  1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, cmap=plt.cm.Paired)\n",
        "from sklearn.inspection import DecisionBoundaryDisplay\n",
        "# plot the decision function\n",
        "ax = plt.gca()\n",
        "DecisionBoundaryDisplay.from_estimator(\n",
        "    clf,\n",
        "    X_train,\n",
        "    plot_method=\"contour\",\n",
        "    colors=\"k\",\n",
        "    levels=[-1, 0, 1],\n",
        "    alpha=0.5,\n",
        "    linestyles=[\"--\", \"-\", \"--\"],\n",
        "    ax=ax,\n",
        ")\n",
        "# plot support vectors\n",
        "ax.scatter(\n",
        "    clf.support_vectors_[:, 0],\n",
        "    clf.support_vectors_[:, 1],\n",
        "    s=100,\n",
        "    linewidth=1,\n",
        "    facecolors=\"none\",\n",
        "    edgecolors=\"k\",\n",
        ")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 266
        },
        "id": "PpLeiKpbP6I5",
        "outputId": "240a8330-c6b6-41c2-b119-8431db0e045c"
      },
      "execution_count": 8,
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
        "# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_train, s=20, cmap=plt.cm.Paired)\n",
        "from sklearn.inspection import DecisionBoundaryDisplay\n",
        "# plot the decision function\n",
        "ax = plt.gca()\n",
        "DecisionBoundaryDisplay.from_estimator(\n",
        "    clf,\n",
        "    X_test,\n",
        "    plot_method=\"contour\",\n",
        "    colors=\"k\",\n",
        "    levels=[-1, 0, 1],\n",
        "    alpha=0.5,\n",
        "    linestyles=[\"--\", \"-\", \"--\"],\n",
        "    ax=ax,\n",
        ")\n",
        "# plot support vectors\n",
        "ax.scatter(\n",
        "    clf.support_vectors_[:, 0],\n",
        "    clf.support_vectors_[:, 1],\n",
        "    s=100,\n",
        "    linewidth=1,\n",
        "    facecolors=\"none\",\n",
        "    edgecolors=\"k\",\n",
        ")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "id": "0IH4P4AmP-HM",
        "outputId": "b1b64853-929a-451a-98a6-01fa85292965"
      },
      "execution_count": 13,
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
        "print(\"Accuracy: {}%\".format(clf.score(X_test, y_test) * 100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o2k_toqrP-22",
        "outputId": "0cbc84ba-6a25-4923-ec3b-ffc97bb00a9b"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 100.0%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def make_meshgrid(x, y, h=.02):\n",
        "  x_min, x_max = x.min() - 1, x.max() + 1\n",
        "  y_min, y_max = y.min() - 1, y.max() + 1\n",
        "  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\n",
        "  return xx, yy\n",
        "\n",
        "def plot_contours(ax, clf, xx, yy, **params):\n",
        "  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])\n",
        "  Z = Z.reshape(xx.shape)\n",
        "  out = ax.contourf(xx, yy, Z, **params)\n",
        "  return out\n",
        "\n",
        "fig, ax = plt.subplots()\n",
        "# title for the plots\n",
        "title = ('Decision surface of linear SVC ')\n",
        "# Set-up grid for plotting.\n",
        "X0, X1 = X[:, 0], X[:, 1]\n",
        "xx, yy = make_meshgrid(X0, X1)\n",
        "plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)\n",
        "\n",
        "5\n",
        "\n",
        "ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20,edgecolors='k')\n",
        "ax.set_ylabel('y label here')\n",
        "ax.set_xlabel('x label here')\n",
        "ax.set_xticks(())\n",
        "ax.set_yticks(())\n",
        "ax.set_title(title)\n",
        "ax.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "kRuIC_ngQAXa",
        "outputId": "642efdcf-a1de-4768-db71-fd70039780aa"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:matplotlib.legend:No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.\n"
          ]
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
      "cell_type": "code",
      "source": [
        "from sklearn import datasets\n",
        "#Load dataset\n",
        "cancer = datasets.load_breast_cancer()"
      ],
      "metadata": {
        "id": "s00RvFu7RPBQ"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(cancer.feature_names)\n",
        "print(cancer.target_names)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DyZYP33gRQn5",
        "outputId": "77ba6460-5ea8-48d1-987d-6959c65bbf5f"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['mean radius' 'mean texture' 'mean perimeter' 'mean area'\n",
            " 'mean smoothness' 'mean compactness' 'mean concavity'\n",
            " 'mean concave points' 'mean symmetry' 'mean fractal dimension'\n",
            " 'radius error' 'texture error' 'perimeter error' 'area error'\n",
            " 'smoothness error' 'compactness error' 'concavity error'\n",
            " 'concave points error' 'symmetry error' 'fractal dimension error'\n",
            " 'worst radius' 'worst texture' 'worst perimeter' 'worst area'\n",
            " 'worst smoothness' 'worst compactness' 'worst concavity'\n",
            " 'worst concave points' 'worst symmetry' 'worst fractal dimension']\n",
            "['malignant' 'benign']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target_names']))\n",
        "df_cancer.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "bUV2WVtWRRT3",
        "outputId": "116f8664-3935-4f22-d36e-3dcb27550e83"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "0        17.99         10.38          122.80     1001.0          0.11840   \n",
              "1        20.57         17.77          132.90     1326.0          0.08474   \n",
              "2        19.69         21.25          130.00     1203.0          0.10960   \n",
              "3        11.42         20.38           77.58      386.1          0.14250   \n",
              "4        20.29         14.34          135.10     1297.0          0.10030   \n",
              "\n",
              "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "0           0.27760          0.3001              0.14710         0.2419   \n",
              "1           0.07864          0.0869              0.07017         0.1812   \n",
              "2           0.15990          0.1974              0.12790         0.2069   \n",
              "3           0.28390          0.2414              0.10520         0.2597   \n",
              "4           0.13280          0.1980              0.10430         0.1809   \n",
              "\n",
              "   mean fractal dimension  ...  worst texture  worst perimeter  worst area  \\\n",
              "0                 0.07871  ...          17.33           184.60      2019.0   \n",
              "1                 0.05667  ...          23.41           158.80      1956.0   \n",
              "2                 0.05999  ...          25.53           152.50      1709.0   \n",
              "3                 0.09744  ...          26.50            98.87       567.7   \n",
              "4                 0.05883  ...          16.67           152.20      1575.0   \n",
              "\n",
              "   worst smoothness  worst compactness  worst concavity  worst concave points  \\\n",
              "0            0.1622             0.6656           0.7119                0.2654   \n",
              "1            0.1238             0.1866           0.2416                0.1860   \n",
              "2            0.1444             0.4245           0.4504                0.2430   \n",
              "3            0.2098             0.8663           0.6869                0.2575   \n",
              "4            0.1374             0.2050           0.4000                0.1625   \n",
              "\n",
              "   worst symmetry  worst fractal dimension  target_names  \n",
              "0          0.4601                  0.11890           0.0  \n",
              "1          0.2750                  0.08902           0.0  \n",
              "2          0.3613                  0.08758           0.0  \n",
              "3          0.6638                  0.17300           0.0  \n",
              "4          0.2364                  0.07678           0.0  \n",
              "\n",
              "[5 rows x 31 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-c228fa7c-9cdb-4d1b-bf56-db81ee210360\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "      <th>target_names</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>0.07871</td>\n",
              "      <td>...</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>0.05667</td>\n",
              "      <td>...</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>0.05999</td>\n",
              "      <td>...</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>0.09744</td>\n",
              "      <td>...</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>0.05883</td>\n",
              "      <td>...</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 31 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-c228fa7c-9cdb-4d1b-bf56-db81ee210360')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-c228fa7c-9cdb-4d1b-bf56-db81ee210360 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-c228fa7c-9cdb-4d1b-bf56-db81ee210360');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = df_cancer.drop(['target_names'], axis=1)\n",
        "X.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "zW6dKhxMRTKb",
        "outputId": "9f57a2a3-46d5-4e80-a668-7410329c6046"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "0        17.99         10.38          122.80     1001.0          0.11840   \n",
              "1        20.57         17.77          132.90     1326.0          0.08474   \n",
              "2        19.69         21.25          130.00     1203.0          0.10960   \n",
              "3        11.42         20.38           77.58      386.1          0.14250   \n",
              "4        20.29         14.34          135.10     1297.0          0.10030   \n",
              "\n",
              "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "0           0.27760          0.3001              0.14710         0.2419   \n",
              "1           0.07864          0.0869              0.07017         0.1812   \n",
              "2           0.15990          0.1974              0.12790         0.2069   \n",
              "3           0.28390          0.2414              0.10520         0.2597   \n",
              "4           0.13280          0.1980              0.10430         0.1809   \n",
              "\n",
              "   mean fractal dimension  ...  worst radius  worst texture  worst perimeter  \\\n",
              "0                 0.07871  ...         25.38          17.33           184.60   \n",
              "1                 0.05667  ...         24.99          23.41           158.80   \n",
              "2                 0.05999  ...         23.57          25.53           152.50   \n",
              "3                 0.09744  ...         14.91          26.50            98.87   \n",
              "4                 0.05883  ...         22.54          16.67           152.20   \n",
              "\n",
              "   worst area  worst smoothness  worst compactness  worst concavity  \\\n",
              "0      2019.0            0.1622             0.6656           0.7119   \n",
              "1      1956.0            0.1238             0.1866           0.2416   \n",
              "2      1709.0            0.1444             0.4245           0.4504   \n",
              "3       567.7            0.2098             0.8663           0.6869   \n",
              "4      1575.0            0.1374             0.2050           0.4000   \n",
              "\n",
              "   worst concave points  worst symmetry  worst fractal dimension  \n",
              "0                0.2654          0.4601                  0.11890  \n",
              "1                0.1860          0.2750                  0.08902  \n",
              "2                0.2430          0.3613                  0.08758  \n",
              "3                0.2575          0.6638                  0.17300  \n",
              "4                0.1625          0.2364                  0.07678  \n",
              "\n",
              "[5 rows x 30 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-3636fc8a-41cb-4694-b4f1-d97f48bf5f9e\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst radius</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>0.07871</td>\n",
              "      <td>...</td>\n",
              "      <td>25.38</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>0.05667</td>\n",
              "      <td>...</td>\n",
              "      <td>24.99</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>0.05999</td>\n",
              "      <td>...</td>\n",
              "      <td>23.57</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>0.09744</td>\n",
              "      <td>...</td>\n",
              "      <td>14.91</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>0.05883</td>\n",
              "      <td>...</td>\n",
              "      <td>22.54</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 30 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-3636fc8a-41cb-4694-b4f1-d97f48bf5f9e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-3636fc8a-41cb-4694-b4f1-d97f48bf5f9e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-3636fc8a-41cb-4694-b4f1-d97f48bf5f9e');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y= df_cancer['target_names']\n",
        "y.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H1uRpqoTRU5f",
        "outputId": "a431b7a1-222d-46e6-b29e-1118bd7f9624"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    0.0\n",
              "1    0.0\n",
              "2    0.0\n",
              "3    0.0\n",
              "4    0.0\n",
              "Name: target_names, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)"
      ],
      "metadata": {
        "id": "JIQLhMKURWpi"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVC\n",
        "svc_model = SVC()"
      ],
      "metadata": {
        "id": "xDjBzkgLRYO3"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "svc_model.fit(X_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "kCporuPfRZ09",
        "outputId": "a9d55ba0-192b-48f7-f1bd-a9f4816913bb"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVC()"
            ],
            "text/html": [
              "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_predict = svc_model.predict(X_test)"
      ],
      "metadata": {
        "id": "rug6OzVCRdRF"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "print(\"Accuracy: \",metrics.accuracy_score(y_test, y_predict))\n",
        "print(\"Precision: \",metrics.precision_score(y_test, y_predict))\n",
        "print(\"Recall: \",metrics.recall_score(y_test, y_predict))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DwSnpmlFRfft",
        "outputId": "d49ff35a-d170-4f8a-848b-892795e6a627"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy:  0.9298245614035088\n",
            "Precision:  0.8918918918918919\n",
            "Recall:  1.0\n"
          ]
        }
      ]
    }
  ]
}
