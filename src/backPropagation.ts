import type { Neuron } from "./neuron";
import type { Example } from "./example";
import { BinaryCrossEntropy } from "./bce";

const bce = new BinaryCrossEntropy();

export function backPropagation(neuron: Neuron, example: Example, learningRate: number = 0.1) {
    const inputValues = Array.isArray(example.inputs) ? example.inputs : Array.from(example.inputs.data);
    const targetValue = typeof example.target === "number" ? example.target : example.target.data[0];

    const prediction = neuron.forward(inputValues);
    const error = prediction - targetValue;
    const loss = bce.forward(prediction, targetValue);

    const dl_dpred = (prediction - targetValue) / (prediction * (1 - prediction)) + 1e-15;
    const dpred_dz = prediction * (1 - prediction);
    const dl_dz = dl_dpred * dpred_dz;

    return {
        prediction,
        error,
        loss,
        weightsGrads: inputValues.map((input) => dl_dz * input),
        biasGrad: dl_dz
    };
}

export default backPropagation;

