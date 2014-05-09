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
        int[] numLayers = {3, 2, 3};
        dA = new DenoisedAutoencoder(numLayers);
        double[][] xs = {
                {0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {1.0, 0.0, 0.5},
                {1.0, 1.0, 0.6},
                {0.4, 0.3, 0.3},
                {0.8, 0.1, 0.1},
                {0.2, 0.1, 0.7},
                {0.0, 0.0, 1.0},
                {0.0, 0.1, 0.9},
                {0.0, 0.5, 0.5},
                {0.1, 0.8, 0.1},
                {0.2, 0.2, 0.6},
                {0.8, 0.1, 0.1}
        };

        dA.train(xs, xs, 150, 0.1, 4, xs, xs, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                assert doubles.length == doubles2.length;
                for (int i = 0; i < doubles.length; i++) {
                    for (int j = 0; j < doubles[i].length; j++) {
                        System.out.printf("%f ", doubles[i][j]);
                    }
                    System.out.printf(" <--> ");
                    for (int j = 0; j < doubles2[i].length; j++) {
                        System.out.printf("%f ", doubles2[i][j]);
                    }
                    System.out.println("");
                }
            }
        });
    }
}
