package com.lewuathe.magi;

/**
 * Created by sasakiumi on 5/10/14.
 */
public class StackedDenoisedAutoencoder {
    private int nIns;
    private int[] hiddenLayerSize;
    private int nOuts;
    private int nLayers;
    private HiddenLayer[] sigmoidLayers;
    private DenoisedAutoencoder[] dALayers;

    public StackedDenoisedAutoencoder(int nIns, int[] hiddenLayerSize, int nOuts) {
        this.nIns = nIns;
        this.hiddenLayerSize = hiddenLayerSize;
        this.nOuts = nOuts;

        this.nLayers = 1 + hiddenLayerSize.length + 1;

        sigmoidLayers = new HiddenLayer[nLayers];
        dALayers = new DenoisedAutoencoder[nLayers];

        int inputSize;
        for (int i = 0; i < nLayers; i++) {
            if (i == 0) {
                inputSize = nIns;
            } else {
                inputSize = hiddenLayerSize[i - 1];
            }

            this.sigmoidLayers[i] = new HiddenLayer(inputSize, hiddenLayerSize[i]);

            int[] numLayers = {inputSize, hiddenLayerSize[i], inputSize};
            this.dALayers[i] = new DenoisedAutoencoder(numLayers, sigmoidLayers[i].weight, sigmoidLayers[i].bias);
        }


    }
}
