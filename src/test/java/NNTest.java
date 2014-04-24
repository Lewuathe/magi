import com.lewuathe.magi.NeuralNetwork;
import com.lewuathe.magi.Util;
import junit.framework.TestCase;
import org.junit.Test;

import java.util.List;

/**
 * Created by sasakiumi on 4/22/14.
 */

public class NNTest extends TestCase {
    NeuralNetwork nn;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        int[] numLayers = {3, 2, 3};
        nn = new NeuralNetwork(numLayers);
        double[][] xs = {
                {0.9, 0.1, 0.0},
                {0.8, 0.1, 0.1},
                {0.6, 0.2, 0.2},
                {0.5, 0.3, 0.2},
                {0.2, 0.5, 0.3},
                {0.3, 0.6, 0.1},
                {0.1, 0.3, 0.6},
                {0.0, 0.4, 0.6},
                {0.0, 0.2, 0.8},
                {0.1, 0.8, 0.1},
                {0.0, 0.9, 0.1}
        };
        nn.train(xs, xs, 10000, 0.01);
    }

    @Test
    public void testSmoke() {
        double[] test = {0.9, 0.0, 0.1};
        double[] ret = nn.feedforward(test);
        for (int i = 0; i < ret.length; i++) {
            System.out.printf("%f ", ret[i]);
        }
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
