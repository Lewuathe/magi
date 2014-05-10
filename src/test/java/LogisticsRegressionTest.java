import com.lewuathe.magi.LogisticsRegression;
import com.lewuathe.magi.Util;
import junit.framework.TestCase;
import org.junit.Test;

import java.util.function.BiConsumer;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class LogisticsRegressionTest extends TestCase {
    LogisticsRegression logRegression;
    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }

    @Test
    public void testSmoke() {
        logRegression = new LogisticsRegression(4, 2);
        double[][] xs = {
                {1.0, 0.0, 0.0, 0.0},
                {0.9, 0.1, 0.0, 0.0},
                {0.99, 0.01, 0.0, 0.0},
                {0.8, 0.2, 0.0, 0.0},
                {0.85, 0.15, 0.0, 0.0},
                {0.0, 0.0, 0.1, 0.9},
                {0.0, 0.0, 0.2, 0.8},
                {0.0, 0.0, 0.05, 0.95},
                {0.0, 0.0, 0.2, 0.8},
                {0.0, 0.0, 0.12, 0.88}
        };

        double[][] ys = {
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0}
        };

        logRegression.train(xs, ys, 0.1, 100, xs, ys, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] answers, double[][] ys) {
                int accuracy = 0;
                for (int i = 0; i < answers.length; i++) {
                    if (Util.maxIndex(answers[i]) == Util.maxIndex(ys[i])) {
                        accuracy++;
                    }
                }
                System.out.printf("LogisticsRegression: %d / %d\n", accuracy, answers.length);
            }
        });

    }
}
