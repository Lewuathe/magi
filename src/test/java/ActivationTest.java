import com.lewuathe.magi.Activation;
import junit.framework.TestCase;
import org.junit.Test;
import org.ujmp.core.Matrix;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class ActivationTest extends TestCase {
    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }

    @Test
    public void testSoftmax() {
        double[] input = {0.4, 0.5, 0.8};
        double[] ret = Activation.softmax(input);
        for (int i = 0; i < input.length; i++) {
            System.out.printf("%f ", ret[i]);
        }
        System.out.println("");
    }

    @Test
    public void testMatrixSoftmax() {
        Matrix m = Matrix.factory.zeros(3, 1);
        m.setAsDouble(0.4, 0, 0);
        m.setAsDouble(0.5, 1, 0);
        m.setAsDouble(0.8, 2, 0);
        Matrix ret = Activation.softmax(m);
        System.out.println(ret);
    }
}
