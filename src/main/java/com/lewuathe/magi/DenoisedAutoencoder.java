package com.lewuathe.magi;

import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 5/8/14.
 */
public class DenoisedAutoencoder extends NeuralNetwork {
    public double corruptionLevel;

    public DenoisedAutoencoder(int[] numLayers) {
        super(numLayers);
    }

    public DenoisedAutoencoder(int[] numLayers, Matrix w, Matrix b) {
        super(numLayers);
        this.weights[0] = w;
        this.biases[0] = b;
        this.corruptionLevel = 0.2;
    }

    public void setCorruptionLevel(double corruptionLevel) {
        this.corruptionLevel = corruptionLevel;
    }

    public void setWeight(Matrix w) {
        this.weights[0] = w;
    }

    public void setBias(Matrix b) {
        this.biases[0] = b;
    }

    /**
     * update
     *
     * @param x
     * @param y
     * @param lr
     */
    @Override
    public void update(double[][] x, double[][] y, double lr) {
        Matrix[] nablaB = new Matrix[2];
        nablaB[0] = Matrix.factory.zeros(numLayers[1], 1);
        nablaB[1] = Matrix.factory.zeros(numLayers[2], 1);
        Matrix[] nablaW = new Matrix[2];
        nablaW[0] = Matrix.factory.zeros(numLayers[1], numLayers[0]);
        nablaW[1] = Matrix.factory.zeros(numLayers[2], numLayers[1]);

        assert x.length == y.length;
        for (int i = 0; i < x.length; i++) {
            Matrix xMat = Matrix.factory.zeros(numLayers[0], 1);
            Matrix yMat = Matrix.factory.zeros(numLayers[2], 1);
            for (int j = 0; j < numLayers[0]; j++) {
                xMat.setAsDouble(corrupt(x[i][j], corruptionLevel), j, 0);
            }
            for (int j = 0; j < numLayers[2]; j++) {
                yMat.setAsDouble(y[i][j], j, 0);
            }
            Matrix[][] delta = this.backprod(xMat, yMat);
            // delta[0]: nablaB
            // delta[1]: nablaW
            nablaB[0] = nablaB[0].plus(delta[0][0]);
            nablaB[1] = nablaB[1].plus(delta[0][1]);
            nablaW[0] = nablaW[0].plus(delta[1][0]);
            nablaW[1] = nablaW[1].plus(delta[1][1]);
        }

        // Update biases and weights with gradient descent
        biases[0] = biases[0].minus(nablaB[0].mtimes(lr));
        biases[1] = biases[1].minus(nablaB[1].mtimes(lr));
        weights[0] = weights[0].minus(nablaW[0].mtimes(lr));
        weights[0] = weights[0].minus(nablaW[1].transpose().mtimes(lr));
    }

    @Override
    protected Matrix[][] backprod(Matrix x, Matrix y) {
        Matrix[] nablaB = new Matrix[2];
        nablaB[0] = Matrix.factory.zeros(numLayers[1], 1);
        nablaB[1] = Matrix.factory.zeros(numLayers[2], 1);
        Matrix[] nablaW = new Matrix[2];
        nablaW[0] = Matrix.factory.zeros(numLayers[1], numLayers[0]);
        nablaW[1] = Matrix.factory.zeros(numLayers[2], numLayers[1]);

        // In case of denoised autoencoder, no use of 1st weight layer
        weights[1] = weights[0].transpose();

        // Activation of each layer
        Matrix activation = x;
        // Collection of activation values of each layer including input
        Matrix[] activations = new Matrix[3];
        // Set input activation
        activations[0] = x;
        // Row values before activating
        Matrix zs[] = new Matrix[2];
//        Matrix mask = Util.makeMask(numLayers[1], 0.5);
        for (int i = 0; i < 2; i++) {
            zs[i] = weights[i].mtimes(activation).plus(biases[i]);
            activation = Activation.sigmoid(zs[i]);
//            if (i == 0) {
//                activation = Util.eachMul(activation, mask);
//            }
            activations[i + 1] = activation;
        }

        // Calculate output layer error
        Matrix delta = costDerivative(activations[2], y);
        delta = Util.eachMul(delta, Activation.sigmoidPrime(zs[1]));
        nablaB[1] = delta;
        nablaW[1] = delta.mtimes(activations[1].transpose());

        for (int i = 1; i > 0; i--) {
            // Back propagation of output layer error to hidden layers
            delta = weights[i].transpose().mtimes(delta);
            delta = Util.eachMul(delta, Activation.sigmoidPrime(zs[i - 1]));
//            delta = Util.eachMul(delta, mask);
            nablaB[i - 1] = delta;
            nablaW[i - 1] = delta.mtimes(activations[i - 1].transpose());
        }

        Matrix[][] ret = {nablaB, nablaW};
        return ret;
    }

    private double corrupt(double input, double level) {
        double noise = level * (2.0 * Math.random() - 1.0);
        return input * (1.0 + noise);
    }

    private double corrupt(double input, int n, double p) {
        if (input == 0.0) {
            return 0.0;
        } else {
            return Util.binomial(n, p);
        }
    }

}
