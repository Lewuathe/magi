import com.lewuathe.magi.NeuralNetwork;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by sasakiumi on 4/22/14.
 */

public class NNTest extends TestCase {

    @Test
    public void testSmoke() {
        int[] numLayers = {3, 2, 3};
        NeuralNetwork nn = new NeuralNetwork(numLayers);
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
        double[] test = {0.9, 0.0, 0.1};
        System.out.println(nn.feedforward(test));
    }
}
