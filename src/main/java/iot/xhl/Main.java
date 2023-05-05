package iot.xhl;

import org.apache.commons.collections4.map.HashedMap;
import org.deeplearning4j.nn.graph
        .ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import oshi.jna.platform.unix.solaris.LibKstat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;

public class Main {

    static int[] indices = {
            1, 107, 231, 25, 109, 15, 233, 0, 2, 3, 235, 4, 29, 234, 232, 199, 239, 5, 230, 16, 180, 98,
            236, 6, 31, 238, 237, 30, 24, 9, 11, 126, 13, 7, 198, 10, 43, 56, 50, 28, 608, 202, 318, 201,
            117, 8, 23, 34, 111, 12, 48, 33, 51, 32, 27, 45, 14, 46, 54, 74, 119, 40, 228, 125, 37, 224,
            35, 44, 61, 248, 118, 70, 121, 60, 49, 52, 96, 206, 17, 39, 36, 18, 38, 123, 41, 241, 57, 240,
            66, 222, 20, 122, 62, 134, 204, 42, 69, 59, 192, 229, 203, 226, 120, 65, 55, 129, 26, 130, 19,
            127, 227, 21, 99, 139, 58, 64, 68, 213, 113, 140, 135, 141, 53, 67, 200, 22, 72, 63
    };

    public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<Integer, String> labelMap = new HashedMap<>(8);
        labelMap.put(0, "benign");
        labelMap.put(1, "ddos");
        labelMap.put(2, "dos");
        labelMap.put(3, "ftp-patator");
        labelMap.put(4, "infiltration");
        labelMap.put(5, "port-scan");
        labelMap.put(6, "ssh-patator");
        labelMap.put(7, "web-attack");

        // load the model
        String simpleMlp = new ClassPathResource("m2_128.h5").getFile().getPath();
        MultiLayerNetwork computationGraph = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);
//        ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(simpleMlp);
        int[][] data = new int[2570][128];

        String filename = new ClassPathResource("bitstring.csv").getFile().getPath();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null && row < 2570) {
                String[] values = line.split(",");
                int[] result = Arrays.stream(indices)
                        .map(i -> Integer.parseInt(values[i + 1]))
                        .toArray();
                data[row] = result;
                row++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        INDArray tensor = Nd4j.create(data);
        System.out.println("tensor.shape() = " + Arrays.toString(tensor.shape()));

        INDArray output = computationGraph.output(tensor);
        System.out.println("output.shape = " + Arrays.toString(output.shape()));

        LongStream stream = Arrays.stream(output.toLongVector());
        stream.forEach(System.out::println);
//        collect.forEach(System.out::println);
    }
}
