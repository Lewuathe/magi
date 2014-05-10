import com.lewuathe.magi.HiddenLayer;
import com.lewuathe.magi.StackedDenoisedAutoencoder;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class HiddenLayerTest extends TestCase {

    HiddenLayer hiddenLayer;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }

    @Test
    public void testSmoke() {
        hiddenLayer = new HiddenLayer(3, 2);
        double[] input = {0.2, 0.3, 0.1};
        double[] ret = hiddenLayer.output(input);
        for (int i = 0; i < 2; i++) {
            System.out.printf("%f ", ret[i]);
        }
        System.out.println("");
    }
}
