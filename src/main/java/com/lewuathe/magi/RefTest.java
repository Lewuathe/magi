package com.lewuathe.magi;

import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class RefTest {
    public static void main(String[] args) {
        Matrix m = Matrix.factory.zeros(3, 3);
        update(m);
        System.out.println(m);
    }

    public static void update(Matrix mat) {
        mat.setAsDouble(100.0, 0, 0);
    }
}
