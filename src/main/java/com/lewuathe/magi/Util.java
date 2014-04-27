package com.lewuathe.magi;

import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by sasakiumi on 4/24/14.
 */

/**
 * Util
 * @since 0.0.1
 */
public class Util {

    /**
     * sampling
     * @param xs
     * @param ys
     * @param size
     * @return Subset of input data
     */
    public static List<double[][]> sampling(double[][] xs, double[][] ys, int size) {
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

    public static Matrix eachMul(Matrix x, Matrix y) {
        assert x.getRowCount() == y.getRowCount();
        assert x.getColumnCount() == y.getColumnCount();
        Matrix ret = Matrix.factory.zeros(x.getRowCount(), x.getColumnCount());
        for (int i = 0; i < x.getRowCount(); i++) {
            for (int j = 0; j < x.getColumnCount(); j++) {
                double mul = x.getAsDouble(i, j) * y.getAsDouble(i, j);
                ret.setAsDouble(mul, i, j);
            }
        }
        return ret;
    }

    public static double[] standardize(double[] input) {
        double mean = 0.0;
        double std = 0.0;
        double[] ret = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            mean += input[i];
        }
        mean /= (double)input.length;
        for (int i = 0; i < input.length; i++) {
            std += Math.pow(input[i] - mean, 2);
        }
        std = Math.sqrt(std) / (double)input.length;
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (input[i] - mean) / std;
        }
        return ret;
    }

    public static double[] nomalize(double[] input) {
        double max = -Double.MAX_VALUE;
        double min = Double.MAX_VALUE;
        double[] ret = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            max = Math.max(max, input[i]);
            min = Math.min(min, input[i]);
        }
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (input[i] - min) / (max - min);
        }
        return ret;
    }

    public static int maxIndex(double[] ds) {
        int maxIndex = 0;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < ds.length; i++) {
            if (max < ds[i]) {
                maxIndex = i;
                max = ds[i];
            }
        }
        return maxIndex;
    }
}
