import Tensor from "./tensor";

export interface Example {
    inputs: Tensor | number[]; // pistas feture vector

    target: Tensor | number; // reposta esperada ground truth value
}
