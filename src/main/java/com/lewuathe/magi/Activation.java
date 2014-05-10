package com.lewuathe.magi;

import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 4/15/14.
 */
public class Activation {
    public static double sigmoid(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    public static Matrix sigmoid(Matrix input) {
        Matrix ret = Matrix.factory.zeros(input.getRowCount(), input.getColumnCount());
        for (int i = 0; i < input.getRowCount(); i++) {
            for (int j = 0; j < input.getColumnCount(); j++) {
                ret.setAsDouble(sigmoid(input.getAsDouble(i, j)), i, j);
            }
        }
        return ret;
    }

    public static double sigmoidPrime(double input) {
        return sigmoid(input) * (1.0 - sigmoid(input));
    }

    public static Matrix sigmoidPrime(Matrix input) {
        Matrix ret = Matrix.factory.zeros(input.getRowCount(), input.getColumnCount());
        for (int i = 0; i < input.getRowCount(); i++) {
            for (int j = 0; j < input.getColumnCount(); j++) {
                ret.setAsDouble(sigmoidPrime(input.getAsDouble(i, j)), i, j);
            }
        }
        return ret;
    }

    public static double hyperbolicTangent(double input) {
        return Math.tanh(input);
    }

    public static Matrix hyperbolicTangent(Matrix input) {
        Matrix ret = Matrix.factory.zeros(input.getRowCount(), input.getColumnCount());
        for (int i = 0; i < input.getRowCount(); i++) {
            for (int j = 0; j < input.getColumnCount(); j++) {
                ret.setAsDouble(hyperbolicTangent(input.getAsDouble(i, j)), i, j);
            }
        }
        return ret;
    }

    public static double hyperbolicTangentPrime(double input) {
        return 1.0 - Math.pow(hyperbolicTangent(input), 2.0);
    }

    public static Matrix hyperbolicTangentPrime(Matrix input) {
        Matrix ret = Matrix.factory.zeros(input.getRowCount(), input.getColumnCount());
        for (int i = 0; i < input.getRowCount(); i++) {
            for (int j = 0; j < input.getColumnCount(); j++) {
                ret.setAsDouble(hyperbolicTangentPrime(input.getAsDouble(i, j)), i, j);
            }
        }
        return ret;
    }

    public static double[] softmax(double[] x) {
        double max = 0.0;
        double sum = 0.0;
        double[] ret = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            if (max < x[i]) {
                max = x[i];
            }
        }

        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i] - max);
            sum += x[i];
        }

        for (int i = 0; i < x.length; i++) {
            ret[i] = x[i] / sum;
        }

        return ret;
    }

    public static Matrix softmax(Matrix input) {
        assert input.getColumnCount() == 1;
        double[] x = new double[(int) input.getRowCount()];
        for (int i = 0; i < x.length; i++) {
            x[i] = input.getAsDouble(i, 0);
        }
        double[] ret = softmax(x);
        Matrix retMat = Matrix.factory.zeros(input.getRowCount(), 1);
        for (int i = 0; i < retMat.getRowCount(); i++) {
            retMat.setAsDouble(ret[i], i, 0);
        }
        return retMat;
    }
}