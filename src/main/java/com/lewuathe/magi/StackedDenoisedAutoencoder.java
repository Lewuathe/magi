package com.lewuathe.magi;

import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class StackedDenoisedAutoencoder {
    private int nIns;
    private int[] hiddenLayerSize;
    private int nOuts;
    private int nLayers;
    private int[] nLayerSize;
    private HiddenLayer[] sigmoidLayers;
    private DenoisedAutoencoder[] dALayers;
    private LogisticsRegression logLayer;

    public StackedDenoisedAutoencoder(int nIns, int[] hiddenLayerSize, int nOuts) {
        this.nIns = nIns;
        this.hiddenLayerSize = hiddenLayerSize;
        this.nOuts = nOuts;

        this.nLayers = 1 + hiddenLayerSize.length + 1;
        this.nLayerSize = new int[nLayers];
        for (int i = 0; i < nLayers; i++) {
            if (i == 0) {
                nLayerSize[i] = nIns;
            } else if (i == nLayers - 1) {
                nLayerSize[i] = nOuts;
            } else {
                nLayerSize[i] = hiddenLayerSize[i - 1];
            }
        }


        sigmoidLayers = new HiddenLayer[nLayers];
        dALayers = new DenoisedAutoencoder[nLayers];

        int inputSize;
        for (int i = 0; i < hiddenLayerSize.length; i++) {
            if (i == 0) {
                inputSize = nIns;
            } else {
                inputSize = hiddenLayerSize[i - 1];
            }

            this.sigmoidLayers[i] = new HiddenLayer(inputSize, hiddenLayerSize[i]);

            int[] numLayers = {inputSize, hiddenLayerSize[i], inputSize};
            this.dALayers[i] = new DenoisedAutoencoder(numLayers);
            this.dALayers[i].setCorruptionLevel(0.2);
        }

        this.logLayer = new LogisticsRegression(hiddenLayerSize[hiddenLayerSize.length - 1], nOuts);
    }

    public void pretrain(double[][] xs, double lr, double corrptionLevel, int epochs) {
        double[] layerInput = new double[0];
        double[] prevLayerInput;

        for (int i = 0; i < hiddenLayerSize.length; i++) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int n = 0; n < xs.length; n++) {
                    for (int l = 0; l <= i; l++) {
                        if (l == 0) {
                            layerInput = xs[n];
                        } else {
                            prevLayerInput = layerInput;
                            sigmoidLayers[l - 1].setWeight(dALayers[l - 1].weights[0]);
                            sigmoidLayers[l - 1].setBias(dALayers[l - 1].biases[0]);
                            layerInput = sigmoidLayers[l - 1].output(prevLayerInput);
                        }
                    }
                    dALayers[i].train(layerInput, layerInput, lr);
                }
            }
        }
    }

    public void finetune(double[][] xs, double[][] ys, double lr, int epochs) {
        assert xs.length == ys.length;
        // Init bias update matrix
        Matrix[] nablaB = new Matrix[nLayers - 1];
        for (int i = 0; i < nLayers - 1; i++) {
            nablaB[i] = Matrix.factory.zeros(nLayerSize[i + 1], 1);
        }

        // Init weight update matrix
        Matrix[] nablaW = new Matrix[nLayers - 1];
        for (int i = 0; i < nLayers - 1; i++) {
            nablaW[i] = Matrix.factory.zeros(nLayerSize[i + 1], nLayerSize[i]);
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Epoch
            for (int n = 0; n < xs.length; n++) {
                // Data
                Matrix xMat = Matrix.factory.zeros(nIns, 1);
                Matrix yMat = Matrix.factory.zeros(nOuts, 1);
                for (int j = 0; j < nIns; j++) {
                    xMat.setAsDouble(xs[n][j], j, 0);
                }
                for (int j = 0; j < nOuts; j++) {
                    yMat.setAsDouble(ys[n][j], j, 0);
                }
                Matrix[][] delta = this.backprod(xMat, yMat);

                for (int i = 0; i < nLayers - 1; i++) {
                    nablaB[i] = delta[0][i];
                    nablaW[i] = delta[1][i];
                    if (i == nLayers - 2) {
                        Matrix bias = logLayer.bias;
                        Matrix weight = logLayer.weight;
                        logLayer.setBias(bias.minus(nablaB[i].mtimes(lr)));
                        logLayer.setWeight(weight.minus(nablaW[i].mtimes(lr)));
                    } else {
                        Matrix bias = dALayers[i].biases[0];
                        Matrix weight = dALayers[i].weights[0];
                        dALayers[i].setBias(bias.minus(nablaB[i]).mtimes(lr));
                        dALayers[i].setWeight(weight.minus(nablaW[i].mtimes(lr)));
                    }
                }
            }
            lr *= 0.95;
        }
    }

    protected Matrix[][] backprod(Matrix x, Matrix y) {
        // Init bias update matrix
        Matrix[] nablaB = new Matrix[nLayers - 1];
        for (int i = 0; i < nLayers - 1; i++) {
            nablaB[i] = Matrix.factory.zeros(nLayerSize[i + 1], 1);
        }

        // Init weight update matrix
        Matrix[] nablaW = new Matrix[nLayers - 1];
        for (int i = 0; i < nLayers - 1; i++) {
            nablaW[i] = Matrix.factory.zeros(nLayerSize[i + 1], nLayerSize[i]);
        }

        // Activation of each layer
        Matrix activation = x;
        // Collection of activation values of each layer including input
        Matrix[] activations = new Matrix[nLayers];
        // Set input activation
        activations[0] = x;
        // Row values before activating
        Matrix zs[] = new Matrix[nLayers - 1];
        for (int i = 1; i < nLayers; i++) {
            Matrix weight;
            Matrix bias;
            if (i == nLayers - 1) {
                weight = logLayer.weight;
                bias = logLayer.bias;
            } else {
                weight = dALayers[i - 1].weights[0];
                bias = dALayers[i - 1].biases[0];
            }
            zs[i - 1] = weight.mtimes(activation).plus(bias);
            activation = Activation.sigmoid(zs[i - 1]);
            activations[i] = activation;
        }

        // Calculate output layer error
        Matrix delta = costDerivative(activations[nLayers - 1], y);
        delta = Util.eachMul(delta, Activation.sigmoidPrime(zs[nLayers - 2]));
        nablaB[nLayers - 2] = delta;
        nablaW[nLayers - 2] = delta.mtimes(activations[nLayers - 2].transpose());

        for (int i = nLayers - 2; i > 0; i--) {
            // Back propagation of output layer error to hidden layers
            Matrix weight;
            if (i == nLayers - 2) {
                weight = logLayer.weight;
            } else {
                weight = dALayers[i].weights[0];
            }
            delta = weight.transpose().mtimes(delta);
            delta = Util.eachMul(delta, Activation.sigmoidPrime(zs[i - 1]));
            nablaB[i - 1] = delta;
            nablaW[i - 1] = delta.mtimes(activations[i - 1].transpose());
        }

        Matrix[][] ret = {nablaB, nablaW};
        return ret;
    }

    public double[] predict(double[] x) {
        double[] layerInput = x;
        double[] prevLayerInput;

        for (int l = 0; l <= hiddenLayerSize.length; l++) {
            if (l == 0) {
                layerInput = x;
            } else {
                prevLayerInput = layerInput;
                sigmoidLayers[l - 1].setWeight(dALayers[l - 1].weights[0]);
                sigmoidLayers[l - 1].setBias(dALayers[l - 1].biases[0]);
                layerInput = sigmoidLayers[l - 1].output(prevLayerInput);
            }
        }

        double[] ret = logLayer.predict(layerInput);
        return Activation.softmax(ret);
    }

    public void debug() {
        for (int i = 0; i < hiddenLayerSize.length; i++) {
            System.out.println(dALayers[i].weights[0]);
        }
        for (int i = 0; i < hiddenLayerSize.length; i++) {
            System.out.println(sigmoidLayers[i].weight);
        }
        System.out.println(logLayer.weight);
    }

    protected Matrix costDerivative(Matrix outputActivation, Matrix y) {
        return outputActivation.minus(y);
    }
}
