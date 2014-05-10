package com.lewuathe.magi;

import org.ujmp.core.Matrix;

import java.util.function.BiConsumer;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class LogisticsRegression {
    private int nIns;
    private int nOuts;
    private Matrix weight;
    private Matrix bias;

    public LogisticsRegression(int nIns, int nOuts) {
        this.nIns = nIns;
        this.nOuts = nOuts;

        this.weight = Matrix.factory.randn(nOuts, nIns);
        this.bias = Matrix.factory.randn(nOuts, 1);
    }

    public void train(double[][] xs, double[][] ys, double lr, int epochs,
                      double[][] testxs, double[][] testys, BiConsumer<double[][], double[][]> evaluator) {
        for (int trainLoop = 0; trainLoop < epochs; trainLoop++) {
            Matrix nablaB = Matrix.factory.zeros(nOuts, 1);
            Matrix nablaW = Matrix.factory.zeros(nOuts, nIns);
            for (int i = 0; i < xs.length; i++) {
                Matrix xMat = Matrix.factory.zeros(nIns, 1);
                Matrix yMat = Matrix.factory.zeros(nOuts, 1);
                for (int j = 0; j < nIns; j++) {
                    xMat.setAsDouble(xs[i][j], j, 0);
                }
                for (int j = 0; j < nOuts; j++) {
                    yMat.setAsDouble(ys[i][j], j, 0);
                }
                Matrix[] delta = this.update(xMat, yMat, lr, xs.length);
                // delta[0]: nablaB
                // delta[1]: nablaW
                nablaB = nablaB.plus(delta[0]);
                nablaW = nablaW.plus(delta[1]);
            }

            bias = bias.minus(nablaB.mtimes(lr));
            weight = weight.minus(nablaW.mtimes(lr));
            if (evaluator != null) {
                this.evaluate(testxs, testys, evaluator);
            }
        }
    }

    public void train(double[][] xs, double[][] ys, double lr, int epochs) {
        assert xs.length == ys.length;
        train(xs, ys, lr, epochs, null, null, null);
    }

    public Matrix[] update(Matrix x, Matrix y, double lr, int n) {
        Matrix nablaB = Matrix.factory.zeros(nOuts, 1);
        Matrix nablaW = Matrix.factory.zeros(nOuts, nIns);

        Matrix z = weight.mtimes(x).plus(bias);
        Matrix a = Activation.softmax(z);

        Matrix delta = costDerivative(a, y);
        for (int i = 0; i < nablaW.getRowCount(); i++) {
            for (int j = 0; j < nablaW.getColumnCount(); j++) {
                double preW = nablaW.getAsDouble(i, j);
                double d = delta.getAsDouble(i, 0);
                double xi = x.getAsDouble(j, 0);
                nablaW.setAsDouble(preW + d * xi / n, i, j);
            }
            double preB = nablaB.getAsDouble(i, 0);
            nablaB.setAsDouble(preB + delta.getAsDouble(i, 0) / n, i, 0);
        }

        Matrix[] ret = {nablaB, nablaW};
        return ret;
    }

    public double[] predict(double[] input) {
        Matrix x = Matrix.factory.zeros(input.length, 1);

        for (int i = 0; i < input.length; i++) {
            x.setAsDouble(input[i], i, 0);
        }

        Matrix retMat = weight.mtimes(x).plus(bias);
        double[] ret = new double[(int) retMat.getRowCount()];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = retMat.getAsDouble(i, 0);
        }
        return ret;
    }

    private void evaluate(double[][] xs, double[][] ys, BiConsumer<double[][], double[][]> evaluator) {
        // Verification
        assert xs.length == ys.length;
        int accurate = 0;
        int TEST_NUM = xs.length;
        double[][] answers = new double[xs.length][];
        for (int i = 0; i < TEST_NUM; i++) {
            answers[i] = this.predict(xs[i]);
        }
        evaluator.accept(answers, ys);
    }

    protected Matrix costDerivative(Matrix outputActivation, Matrix y) {
        return outputActivation.minus(y);
    }
}
