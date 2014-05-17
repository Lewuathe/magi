import com.lewuathe.magi.StackedDenoisedAutoencoder;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by sasakiumi on 5/12/14.
 */
public class SdATest extends TestCase {
    StackedDenoisedAutoencoder sda;
    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }

    @Test
    public void testSmoke() {
        int[] hiddenLayerSize = {4, 3};
        sda = new StackedDenoisedAutoencoder(5, hiddenLayerSize, 2);
        double[][] xs = {
                {1.0, 0.0, 0.0, 0.0, 0.0},
                {0.9, 0.1, 0.0, 0.0, 0.0},
                {0.8, 0.1, 0.1, 0.0, 0.0},
                {0.8, 0.2, 0.0, 0.0, 0.0},
                {0.85, 0.15, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.1, 0.1, 0.8},
                {0.0, 0.0, 0.0, 0.1, 0.9},
                {0.0, 0.0, 0.0, 0.15, 0.85},
                {0.0, 0.0, 0.1, 0.1, 0.8},
                {0.0, 0.0, 0.0, 0.2, 0.8}
        };
        double[][] ys = {
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0}
        };
//        sda.debug();
//        System.out.println("------------------");
        sda.pretrain(xs, 0.1, 0.2, 1000);
        sda.finetune(xs, ys, 0.3, 30);

        for (int i = 0; i < xs.length; i++) {
            double[] ret = sda.predict(xs[i]);
            for (int j = 0; j < ret.length; j++) {
                assertTrue(Math.abs(ret[j] - ys[i][j]) < 0.1);
            }
        }
//        sda.debug();
    }
}
