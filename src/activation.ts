import Tensor from "./tensor";

export class RELU {
    private lastinput?: Tensor;

    forward(input: Tensor): Tensor {
        this.lastinput = input;

        const output = new Float64Array(input.data.length);
        for (let i = 0; i < input.data.length; i++) {
            output[i] = Math.max(0, input.data[i]);
        }

        return new Tensor(output, [...input.shape]);
    }

    backward(grad: Tensor): Tensor {
        const output = new Float64Array(grad.data.length);
        for (let i = 0; i < grad.data.length; i++) {
            output[i] = this.lastinput && this.lastinput.data[i] > 0 ? grad.data[i] : 0;
        }

        return new Tensor(output, [...grad.shape]);
    }
}

export class Sigmoid {
    private lastoutput?: Tensor;

    forward(input: Tensor): Tensor {
        const output = input.map((x) => 1 / (1 + Math.exp(-x)));
        this.lastoutput = output;
        return output;
    }

    backward(grad: Tensor): Tensor {
        const output = new Float64Array(grad.data.length);
        for (let i = 0; i < grad.data.length; i++) {
            const y = this.lastoutput ? this.lastoutput.data[i] : 0;
            output[i] = grad.data[i] * y * (1 - y);
        }
        return new Tensor(output, [...grad.shape]);
    }
}