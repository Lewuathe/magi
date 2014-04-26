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

}
