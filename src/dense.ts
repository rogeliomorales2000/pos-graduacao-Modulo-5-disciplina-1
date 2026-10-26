import Tensor from "./tensor";

export class Dense {

    private lastInput?: Tensor;
    gradWeights?: Tensor;
    gradBias?: Tensor;
    constructor(
        public weights: Tensor,
        public bias: Tensor
    ) {}

    get weigths(): Tensor {
        return this.weights;
    }

    set weigths(value: Tensor) {
        this.weights = value;
    }

    forward(inputs: Tensor): Tensor {
        this.lastInput = inputs;
        const output = new Float64Array(this.bias.data.length);
        const inSize = inputs.data.length;

        for (let i = 0; i < output.length; i++) {
            let sum = this.bias.data[i];

            for (let j = 0; j < inSize; j++) {
                const weightIndex = j * output.length + i;
                sum += inputs.data[j] * this.weights.data[weightIndex];
            }

            output[i] = sum;
        }

        return new Tensor(output, [output.length]);
    }

    backward(gradOutput: Tensor): Tensor {
        const input = this.lastInput!;
        const inFeatures = input.data.length;
        const outFeatures = gradOutput.data.length;

        this.gradWeights = Tensor.zeros(this.weights.shape);
        this.gradBias = Tensor.zeros(this.bias.shape);
        const gradInput = new Float64Array(inFeatures);

        for (let i = 0; i < outFeatures; i++) {
            this.gradBias.data[i] = gradOutput.data[i];
            for (let j = 0; j < inFeatures; j++) {
                const wIdx = j * outFeatures + i;
                this.gradWeights.data[wIdx] = input.data[j] * gradOutput.data[i];
                gradInput[j] += this.weights.data[wIdx] * gradOutput.data[i];
            }
        }

        return new Tensor(gradInput, [inFeatures]);
    }
}
