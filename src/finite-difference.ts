import type { Neuron } from "./neuron";
import { BinaryCrossEntropy } from "./bce";

type Output ={
    weightsGrads :number [];
    biasGrad : number;
}
const epsilon = 1e-7; // Valor pequeno para evitar divisao por zero
const bce = new BinaryCrossEntropy();

export function numericalGradientsForNeuron(neuron: Neuron, inputs: number[], target: number) : Output{
    const lossFn = (n: Neuron) => {
        const prediction = n.forward(inputs);
        return bce.forward(prediction, target);
    }
// Gradiente em relacao aos pesos
const weightsGrads: number[] = [];
for (let i = 0; i < neuron.weigths.length; i++) {
    const originalWeight = neuron.weigths[i];
    neuron.weigths[i] = originalWeight + epsilon;
    const lossPlus = lossFn(neuron);
    neuron.weigths[i] = originalWeight - epsilon;
    const lossMinus = lossFn(neuron);
    neuron.weigths[i] = originalWeight;
    weightsGrads.push((lossPlus - lossMinus) / (2 * epsilon));
}

 // Gradiente em relacao ao bias
const originalBias = neuron.bias;
neuron.bias = originalBias + epsilon;
const lossPlus = lossFn(neuron);
neuron.bias = originalBias - epsilon;
const lossMinus = lossFn(neuron);
neuron.bias = originalBias;
const biasGrad = (lossPlus - lossMinus) / (2 * epsilon);



return {
    weightsGrads,
    biasGrad
}
}