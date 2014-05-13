package com.lewuathe.magi;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class StackedDenoisedAutoencoder {
    private int nIns;
    private int[] hiddenLayerSize;
    private int nOuts;
    private int nLayers;
    private HiddenLayer[] sigmoidLayers;
    private DenoisedAutoencoder[] dALayers;
    private LogisticsRegression logLayer;

    public StackedDenoisedAutoencoder(int nIns, int[] hiddenLayerSize, int nOuts) {
        this.nIns = nIns;
        this.hiddenLayerSize = hiddenLayerSize;
        this.nOuts = nOuts;

        this.nLayers = 1 + hiddenLayerSize.length + 1;

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
        double[] layerInput = new double[0];
        double[] prevLayerInput = new double[0];
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int n = 0; n < xs.length; n++) {
                for (int l = 0; l < hiddenLayerSize.length; l++) {
                    if (l == 0) {
                        layerInput = xs[n];
                    } else {
                        prevLayerInput = layerInput;
                        sigmoidLayers[l - 1].setWeight(dALayers[l - 1].weights[0]);
                        sigmoidLayers[l - 1].setBias(dALayers[l - 1].biases[0]);
                        layerInput = sigmoidLayers[l - 1].output(prevLayerInput);
                    }
                }
                logLayer.train(layerInput, ys[n], lr, epochs);
            }
            lr *= 0.95;
        }
    }

    public double[] predict(double[] x) {
        double[] layerInput = new double[0];
        double[] prevLayerInput;

        for (int l = 0; l < hiddenLayerSize.length; l++) {
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
}
