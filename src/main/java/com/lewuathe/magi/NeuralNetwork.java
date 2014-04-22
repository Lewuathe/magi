package com.lewuathe.magi;

/**
 * Created by sasakiumi on 4/22/14.
 */

/**
 * Created by sasakiumi on 4/21/14.
 */

import org.ujmp.core.Matrix;

import java.util.*;

/**
 * Created by sasakiumi on 4/16/14.
 */
public class NeuralNetwork {
    private int[] numLayers;
    private int sizes;
    private Matrix[] biases = new Matrix[2];
    private Matrix[] weights = new Matrix[2];

    public NeuralNetwork(int[] numLayers) {
        this.numLayers = numLayers;
        this.sizes = numLayers.length;
        this.biases[0] = Matrix.factory.rand(numLayers[1], 1);
        this.biases[1] = Matrix.factory.rand(numLayers[2], 1);
        this.weights[0] = Matrix.factory.rand(numLayers[1], numLayers[0]);
        this.weights[1] = Matrix.factory.rand(numLayers[2], numLayers[1]);
    }

    public Matrix feedforward(double[] input) {
        Matrix x = Matrix.factory.zeros(input.length, 1);

        for (int i = 0; i < input.length; i++) {
            x.setAsDouble(input[i], i, 0);
        }

        for (int i = 0; i < 2; i++) {
            x = weights[i].mtimes(x).plus(biases[i]);
            for (int j = 0; j < this.numLayers[i + 1]; j++) {
                x.setAsDouble(Activation.sigmoid(x.getAsDouble(j, 0)), j, 0);
            }
        }
        return x;
    }

    public List<double[][]> sampling(double[][] xs, double[][] ys, int size) {
        int xrow = xs.length;
        int xdim = xs[0].length;
        int yrow = ys.length;
        int ydim = ys[0].length;

        double[][] retx = new double[size][xdim];
        double[][] rety = new double[size][ydim];
        for (int i = 0; i < size; i++) {
            int rand = (int) (Math.random() * size);
            for (int j = 0; j < xdim; j++) {
                retx[i][j] = xs[rand][j];
            }
            for (int j = 0; j < ydim; j++) {
                rety[i][j] = xs[rand][j];
            }
        }

        List<double[][]> ret = new ArrayList<double[][]>();
        ret.add(retx);
        ret.add(rety);
        return ret;
    }

    public void train(double[][] x, double[][] y, int epochs, double lr) {
        int n = x.length;
        for (int i = 0; i < epochs; i++) {
            this.update(x, y, lr);
        }
    }

    public void update(double[][] x, double[][] y, double lr) {
        Matrix[] nablaB = new Matrix[2];
        nablaB[0] = Matrix.factory.zeros(numLayers[1], 1);
        nablaB[1] = Matrix.factory.zeros(numLayers[2], 1);
        Matrix[] nablaW = new Matrix[2];
        nablaW[0] = Matrix.factory.zeros(numLayers[1], numLayers[0]);
        nablaW[1] = Matrix.factory.zeros(numLayers[2], numLayers[1]);


        for (int i = 0; i < x.length; i++) {
            Matrix xMat = Matrix.factory.zeros(numLayers[0], 1);
            Matrix yMat = Matrix.factory.zeros(numLayers[2], 1);
            for (int j = 0; j < numLayers[0]; j++) {
                xMat.setAsDouble(x[i][j], j, 0);
            }
            for (int j = 0; j < numLayers[2]; j++) {
                yMat.setAsDouble(y[i][j], j, 0);
            }
            Matrix[][] delta = this.backprod(xMat, yMat);
            nablaB[0] = nablaB[0].plus(delta[0][0]);
            nablaB[1] = nablaB[1].plus(delta[0][1]);
            nablaW[0] = nablaW[0].plus(delta[1][0]);
            nablaW[1] = nablaW[1].plus(delta[1][1]);
        }

        // Update biases and weights with gradient descent
        biases[0] = biases[0].minus(nablaB[0].mtimes(lr));
        biases[1] = biases[1].minus(nablaB[1].mtimes(lr));
        weights[0] = weights[0].minus(nablaW[0].mtimes(lr));
        weights[1] = weights[1].minus(nablaW[1].mtimes(lr));
    }

    public Matrix[][] backprod(Matrix x, Matrix y) {
        Matrix[] nablaB = new Matrix[2];
        nablaB[0] = Matrix.factory.zeros(numLayers[1], 1);
        nablaB[1] = Matrix.factory.zeros(numLayers[2], 1);
        Matrix[] nablaW = new Matrix[2];
        nablaW[0] = Matrix.factory.zeros(numLayers[1], numLayers[0]);
        nablaW[1] = Matrix.factory.zeros(numLayers[2], numLayers[1]);

        // Activation of each layer
        Matrix activation = x;
        // Collection of activation values of each layer including input
        Matrix[] activations = new Matrix[3];
        // Set input activation
        activations[0] = x;
        // Row values before activating
        Matrix zs[] = new Matrix[2];
        for (int i = 0; i < 2; i++) {
            zs[i] = weights[i].mtimes(activation).plus(biases[i]);
            activation = Activation.sigmoid(zs[i]);
            activations[i + 1] = activation;
        }

        // Calculate output layer error
        Matrix delta = costDerivative(activations[2], y);
        nablaB[1] = delta;
        nablaW[1] = delta.mtimes(activations[1].transpose());

        for (int i = 1; i > 0; i--) {
            // Back propagation of output layer error to hidden layers
            delta = weights[i].transpose().mtimes(delta);
            for (int j = 0; j < delta.getRowCount(); j++) {
                // Multiply differential of sigmoid function
                delta.setAsDouble(delta.getAsDouble(j, 0) * Activation.sigmoidPrime(zs[i - 1]).getAsDouble(j, 0), j, 0);
            }
            nablaB[i - 1] = delta;
            nablaW[i - 1] = delta.mtimes(activations[i - 1].transpose());
        }

        Matrix[][] ret = {nablaB, nablaW};
        return ret;
    }

    private Matrix costDerivative(Matrix outputActivation, Matrix y) {
        return outputActivation.minus(y);
    }
}