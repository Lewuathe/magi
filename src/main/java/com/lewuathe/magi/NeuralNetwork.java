package com.lewuathe.magi;

/**
 * Created by Kai Sasaki on 4/22/14.
 */

import org.ujmp.core.Matrix;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;


/**
 * NeuralNetwork
 *
 * @author Kai Sasaki
 * @since 0.0.1
 */
public class NeuralNetwork {
    private int[] numLayers;
    private int sizes;
    private static final double etaPlus = 1.2;
    private static final double etaMinus = 0.5;
    private Matrix[] biases = new Matrix[2];
    private Matrix[] weights = new Matrix[2];
    private Matrix[] updateValueB = new Matrix[2];
    private Matrix[] updateValueW = new Matrix[2];
    private Matrix[] preNablaB = new Matrix[2];
    private Matrix[] preNablaW = new Matrix[2];

    public NeuralNetwork(int[] numLayers) {
        this.numLayers = numLayers;
        this.sizes = numLayers.length;
        // Network weights
        this.biases[0] = Matrix.factory.randn(numLayers[1], 1);
        this.biases[1] = Matrix.factory.randn(numLayers[2], 1);
        this.weights[0] = Matrix.factory.randn(numLayers[1], numLayers[0]);
        this.weights[1] = Matrix.factory.randn(numLayers[2], numLayers[1]);

        // Update factors for resilient propagation
        this.updateValueB[0] = Matrix.factory.randn(numLayers[1], 1);
        this.updateValueB[1] = Matrix.factory.randn(numLayers[2], 1);
        this.updateValueW[0] = Matrix.factory.randn(numLayers[1], numLayers[0]);
        this.updateValueW[1] = Matrix.factory.randn(numLayers[2], numLayers[1]);

        // Gradient descent calculated previously for resilient propagation
        this.preNablaB[0] = Matrix.factory.randn(numLayers[1], 1);
        this.preNablaB[1] = Matrix.factory.randn(numLayers[2], 1);
        this.preNablaW[0] = Matrix.factory.randn(numLayers[1], numLayers[0]);
        this.preNablaW[1] = Matrix.factory.randn(numLayers[2], numLayers[1]);
    }

    /**
     * feedforward
     *
     * @param input
     * @return double[]
     */
    public double[] feedforward(double[] input) {
        Matrix x = Matrix.factory.zeros(input.length, 1);

        for (int i = 0; i < input.length; i++) {
            x.setAsDouble(input[i], i, 0);
        }

        // Activation of each layer
        Matrix activation = x;
        // Set input activation
        for (int i = 0; i < 2; i++) {
            activation = Activation.sigmoid(weights[i].mtimes(activation).plus(biases[i]));
//            activation = Activation.hyperbolicTangent(weights[i].mtimes(activation).plus(biases[i]));
        }

        double[] ret = new double[this.numLayers[2]];
        for (int i = 0; i < activation.getRowCount(); i++) {
            ret[i] = activation.getAsDouble(i, 0);
        }
        return ret;
    }

    /**
     * train
     *
     * @param x
     * @param y
     * @param epochs
     * @param lr
     */
    public void train(double[][] x, double[][] y, int epochs, double lr, int minibatchSize, double[][] testxs, double[][] testys) {
        train(x, y, epochs, lr, minibatchSize, testxs, testys, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                ;
            }
        });
    }

    public void train(double[][] x, double[][] y, int epoch, double lr, int minibatchSize,
                      double[][] testxs, double[][] testys, BiConsumer<double[][], double[][]> evaluator) {
        assert x.length == y.length;
        for (int i = 0; i < epoch; i++) {
            int loop = x.length / minibatchSize;
            for (int j = 0; j < loop; j++) {
                double[][] batchX = new double[minibatchSize][];
                double[][] batchY = new double[minibatchSize][];
                for (int k = 0; k < minibatchSize; k++) {
                    batchX[k] = x[j * minibatchSize + k];
                    batchY[k] = y[j * minibatchSize + k];
                }
                this.update(batchX, batchY, lr);
            }
            this.evaluate(testxs, testys, evaluator);
        }
    }

    /**
     * update
     *
     * @param x
     * @param y
     * @param lr
     */
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
                xMat.setAsDouble(x[i][j], j, 0);
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
        weights[1] = weights[1].minus(nablaW[1].mtimes(lr));

        // Update biases and weight with resilient propagation
//        biases[0] = biases[0].plus(manhattanUpdate(nablaB[0], updateValueB[0]));
//        biases[1] = biases[1].plus(manhattanUpdate(nablaB[1], updateValueB[1]));
//        weights[0] = weights[0].plus(manhattanUpdate(nablaW[0], updateValueW[0]));
//        weights[1] = weights[1].plus(manhattanUpdate(nablaW[1], updateValueW[1]));
//        sdAdaption(nablaB, nablaW);
    }

    private void sdAdaption(Matrix[] nablaB, Matrix[] nablaW) {
        // Update bias of each layer
        for (int l = 0; l < 2; l++) {
            Matrix ret = Util.eachMul(nablaB[l], preNablaB[l]);
            for (int i = 0; i < ret.getRowCount(); i++) {
                for (int j = 0; j < ret.getColumnCount(); j++) {
                    double pre = updateValueB[l].getAsDouble(i, j);
                    if (ret.getAsDouble(i, j) > 0.0) {
                        updateValueB[l].setAsDouble(etaPlus * pre, i, j);
                    } else if (ret.getAsDouble(i, j) < 0.0) {
                        updateValueB[l].setAsDouble(etaMinus * pre, i, j);
                    } else {
                        updateValueB[l].setAsDouble(pre, i, j);
                    }
                }
            }
            preNablaB[l] = nablaB[l];
        }

        // Update weight of each layer
        for (int l = 0; l < 2; l++) {
            Matrix ret = Util.eachMul(nablaW[l], preNablaW[l]);
            for (int i = 0; i < ret.getRowCount(); i++) {
                for (int j = 0; j < ret.getColumnCount(); j++) {
                    double pre = updateValueW[l].getAsDouble(i, j);
                    if (ret.getAsDouble(i, j) > 0.0) {
                        updateValueW[l].setAsDouble(etaPlus * pre, i, j);
                    } else if (ret.getAsDouble(i, j) < 0.0) {
                        updateValueW[l].setAsDouble(etaMinus * pre, i, j);
                    } else {
                        updateValueW[l].setAsDouble(pre, i, j);
                    }
                }
            }
            preNablaW[l] = nablaW[l];
        }
    }

    private Matrix manhattanUpdate(Matrix delta, Matrix update) {
        Matrix ret = Matrix.factory.zeros(delta.getRowCount(), delta.getColumnCount());
        for (int i = 0; i < delta.getRowCount(); i++) {
            for (int j = 0; j < delta.getColumnCount(); j++) {
                if (delta.getAsDouble(i, j) > 0.0) {
                    ret.setAsDouble(-update.getAsDouble(i, j), i, j);
                } else if (delta.getAsDouble(i, j) < 0.0) {
                    ret.setAsDouble(update.getAsDouble(i, j), i, j);
                } else {
                    ret.setAsDouble(0.0, i, j);
                }
            }
        }
        return ret;
    }


    private Matrix[][] backprod(Matrix x, Matrix y) {
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
//            activation = Activation.hyperbolicTangent(zs[i]);
            activations[i + 1] = activation;
        }

        // Calculate output layer error
        Matrix delta = costDerivative(activations[2], y);
        delta = Util.eachMul(delta, Activation.sigmoidPrime(zs[1]));
//        delta = Util.eachMul(delta, Activation.hyperbolicTangentPrime(zs[1]));
        nablaB[1] = delta;
        nablaW[1] = delta.mtimes(activations[1].transpose());

        for (int i = 1; i > 0; i--) {
            // Back propagation of output layer error to hidden layers
            delta = weights[i].transpose().mtimes(delta);
            delta = Util.eachMul(delta, Activation.sigmoidPrime(zs[i - 1]));
//            delta = Util.eachMul(delta, Activation.hyperbolicTangentPrime(zs[i - 1]));
            nablaB[i - 1] = delta;
            nablaW[i - 1] = delta.mtimes(activations[i - 1].transpose());
        }

        Matrix[][] ret = {nablaB, nablaW};
        return ret;
    }

    private Matrix costDerivative(Matrix outputActivation, Matrix y) {
        return outputActivation.minus(y);
    }

    private void evaluate(double[][] xs, double[][] ys, BiConsumer<double[][], double[][]> evaluator) {
        // Verification
        assert xs.length == ys.length;
        int accurate = 0;
        int TEST_NUM = xs.length;
        double[][] answers = new double[xs.length][];
        for (int i = 0; i < TEST_NUM; i++) {
            answers[i] = this.feedforward(xs[i]);

//            if (Util.maxIndex(ans) == Util.maxIndex(ys[i])) {
//                accurate++;
//            }
//            if (Math.abs(ans[0] - ys[i][0]) < 0.1) {
//                accurate++;
//            }
        }
        evaluator.accept(answers, ys);

//        System.out.printf("Accuracy: %d / %d\n", accurate, TEST_NUM);

    }
}