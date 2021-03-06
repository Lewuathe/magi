import com.lewuathe.magi.DenoisedAutoencoder;
import junit.framework.TestCase;
import org.junit.Test;

import java.util.function.BiConsumer;

/**
 * Created by sasakiumi on 5/9/14.
 */
public class DATest extends TestCase {
    DenoisedAutoencoder dA;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }

    @Test
    public void testSmoke() {
        int[] numLayers = {5, 4, 5};
        dA = new DenoisedAutoencoder(numLayers);
        dA.setCorruptionLevel(0.2);
        double[][] xs = {
                {0.2, 0.2, 0.0, 0.9, 0.1},
                {0.0, 0.2, 0.0, 0.1, 0.0},
                {1.0, 0.1, 1.0, 0.1, 0.5},
                {1.0, 0.1, 0.1, 0.0, 0.4},
                {1.0, 0.2, 0.4, 0.3, 0.3},
                {1.0, 0.0, 0.5, 0.2, 0.0},
                {0.0, 0.3, 1.0, 0.0, 0.4},
                {0.0, 0.1, 0.6, 0.0, 0.3},
                {0.0, 0.0, 0.8, 0.0, 0.2},
                {0.0, 0.4, 0.9, 0.0, 0.1},
                {0.0, 0.5, 0.0, 0.2, 0.9},
                {0.0, 0.0, 1.0, 0.4, 0.8},
                {1.0, 0.1, 0.0, 0.5, 0.7},
                {0.0, 0.0, 0.0, 0.5, 0.6},
                {1.0, 0.0, 0.0, 0.0, 0.5},
                {0.1, 0.2, 0.1, 0.6, 0.4},
                {0.2, 0.0, 0.0, 0.7, 0.3},
                {0.3, 0.0, 0.0, 0.6, 0.2},
                {0.4, 0.0, 0.2, 0.1, 0.0},
                {0.5, 0.0, 0.0, 1.0, 0.0},
                {0.6, 0.1, 0.1, 0.2, 0.0},
                {0.7, 0.0, 0.0, 0.1, 0.1}
        };

        double[][] testxs = {
                {0.0, 0.1, 0.0, 0.0, 0.0},
                {0.0, 0.2, 0.9, 0.1, 0.1},
                {0.1, 0.3, 0.8, 0.2, 0.4},
                {0.2, 0.1, 0.8, 0.3, 0.3},
                {0.3, 1.0, 0.7, 0.5, 0.1},
                {0.4, 0.0, 0.6, 0.1, 0.2},
                {0.5, 0.4, 0.5, 0.9, 0.1}
        };

        dA.train(xs, xs, 100, 0.1, 4, xs, xs, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                assert doubles.length == doubles2.length;
                int accuracy = 0;
                for (int i = 0; i < doubles.length; i++) {
                    boolean isOk = true;
                    for (int j = 0; j < doubles[i].length; j++) {
                        if (Math.abs(doubles[i][j] - doubles2[i][j]) > 0.1) {
                            isOk = false;
                            break;
                        }
                    }
                    if (isOk) {
                        accuracy++;
                    }
                }
                System.out.printf("dA Accuracy: %d / %d\n", accuracy, doubles.length);
            }
        });
    }
}
