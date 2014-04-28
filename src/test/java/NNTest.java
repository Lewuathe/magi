import com.lewuathe.magi.Activation;
import com.lewuathe.magi.NeuralNetwork;
import com.lewuathe.magi.Util;
import junit.framework.TestCase;
import org.junit.Test;
import org.ujmp.core.Matrix;

import java.util.List;
import java.util.function.BiConsumer;

/**
 * Created by sasakiumi on 4/22/14.
 */

public class NNTest extends TestCase {
    NeuralNetwork nn;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }

    @Test
    public void testSmoke() {
        int[] numLayers = {2, 2, 1};
        nn = new NeuralNetwork(numLayers);
        double[][] xs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        double[][] ys = {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };
        double[][] testxs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        double[][] testys = {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };
        //nn.train(xs, ys, 10010, 0.3, testxs, testys);
    }

    @Test
    public void testOR() {
        int[] numLayers = {2, 2, 1};
        nn = new NeuralNetwork(numLayers);
        double[][] xs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        double[][] ys = {
                {0.0},
                {1.0},
                {1.0},
                {1.0}
        };
        nn.train(xs, ys, 150, 0.3, 4, xs, ys, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                assert doubles.length == doubles2.length;
                for (int i = 0; i < doubles.length; i++) {
                    if (Math.abs(doubles[i][0] - doubles2[i][0]) < 0.1) {
                        System.out.printf("%f <-> %f\n", doubles[i][0], doubles2[i][0]);
                    }
                }
            }
        });
    }

    @Test
    public void testAND() {
        int[] numLayers = {2, 2, 1};
        nn = new NeuralNetwork(numLayers);
        double[][] xs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        double[][] ys = {
                {0.0},
                {0.0},
                {0.0},
                {1.0}
        };
        nn.train(xs, ys, 150, 0.3, 4, xs, ys, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                assert doubles.length == doubles2.length;
                for (int i = 0; i < doubles.length; i++) {
                    if (Math.abs(doubles[i][0] - doubles2[i][0]) < 0.1) {
                        System.out.printf("%f <-> %f\n", doubles[i][0], doubles2[i][0]);
                    }
                }
            }
        });
    }

    @Test
    public void testNOT() {
        int[] numLayers = {1, 2, 1};
        nn = new NeuralNetwork(numLayers);
        double[][] xs = {
                {0.0},
                {0.0},
                {1.0},
                {1.0}
        };
        double[][] ys = {
                {1.0},
                {1.0},
                {0.0},
                {0.0}
        };
        nn.train(xs, ys, 150, 0.3, 4, xs, ys, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                assert doubles.length == doubles2.length;
                for (int i = 0; i < doubles.length; i++) {
                    if (Math.abs(doubles[i][0] - doubles2[i][0]) < 0.1) {
                        System.out.printf("%f <-> %f\n", doubles[i][0], doubles2[i][0]);
                    }
                }
            }
        });
    }

    @Test
    public void testStandarization() {
        double[] x = {0.1, 0.2, 0.3};
        double[] ret = Util.standardize(x);
        assertTrue(ret[0] < ret[1]);
        assertTrue(ret[1] < ret[2]);
    }

    @Test
    public void testNormalization() {
        double[] x = {0.1, 0.2, 0.7};
        double[] ret = Util.nomalize(x);
        assertTrue(ret[0] < ret[1]);
        assertTrue(ret[1] < ret[2]);
    }

    @Test
    public void testTanh() {
        double[] x = {1.0, 2.0, 3.0};
        x = Util.standardize(x);
        Matrix xMat = Matrix.factory.zeros(x.length, 1);
        for (int i = 0; i < xMat.getRowCount(); i++) {
            xMat.setAsDouble(x[i], i, 0);
        }
        Matrix ret = Activation.hyperbolicTangent(xMat);
        assertTrue(ret.getAsDouble(0, 0) < ret.getAsDouble(1, 0));
        assertTrue(ret.getAsDouble(1, 0) < ret.getAsDouble(2, 0));
    }

    @Test
    public void testRandom() {
        double[][] xs = {
                {0.1, 0.2, 0.3},
                {0.2, 0.3, 0.4},
                {0.3, 0.4, 0.5},
                {0.4, 0.5, 0.6},
                {0.5, 0.6, 0.7},
                {0.6, 0.7, 0.8},
                {0.7, 0.8, 0.9}
        };
        List<double[][]> ret = Util.sampling(xs, xs, 3);
        double[][] newXs = ret.get(0);
        double[][] newYs = ret.get(1);
        for (int i = 0; i < newXs.length; i++) {
            for (int j = 0; j < newXs[i].length; j++) {
                assertEquals(newXs[i][j], newYs[i][j]);
            }
        }

    }
}
