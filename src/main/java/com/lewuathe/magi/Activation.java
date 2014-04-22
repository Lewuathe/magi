package com.lewuathe.magi;

import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 4/15/14.
 */
public class Activation {
    public static double sigmoid(double input) {
        return 1 / (1 + Math.exp(-input));
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
        return sigmoid(input) * (1 - sigmoid(input));
    }

    public static Matrix sigmoidPrime(Matrix input) {
        Matrix ret = Matrix.factory.zeros(input.getRowCount(), input.getColumnCount());
        for (int i = 0; i < input.getRowCount() ; i++) {
            for (int j = 0; j < input.getColumnCount(); j++) {
                ret.setAsDouble(sigmoid(input.getAsDouble(i, j)), i, j);
            }
        }
        return ret;
    }
}