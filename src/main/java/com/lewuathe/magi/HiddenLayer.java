package com.lewuathe.magi;

import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class HiddenLayer {
    private int nIns;
    private int nOuts;
    public Matrix weight;
    public Matrix bias;

    public HiddenLayer(int nIns, int nOuts) {
        this.nIns = nIns;
        this.nOuts = nOuts;
        this.weight = Matrix.factory.randn(nOuts, nIns);
        this.bias = Matrix.factory.randn(nOuts, 1);
    }

    public double[] output(double[] input) {
        Matrix x = Matrix.factory.zeros(this.nIns, 1);
        for (int i = 0; i < x.getRowCount(); i++) {
            x.setAsDouble(input[i], i, 0);
        }

        Matrix hidden = weight.mtimes(x).plus(bias);
        hidden = Activation.sigmoid(hidden);
        double[] ret = new double[nOuts];
        for (int i = 0; i < nOuts; i++) {
            ret[i] = hidden.getAsDouble(i, 0);
        }
        return ret;
    }
}
