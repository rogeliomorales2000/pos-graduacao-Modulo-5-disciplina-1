import Tensor from "./tensor";

export class BinaryCrossEntropy {
    private lastPrediction = 0;
    private lastTarget = 0;

    forward(predictions: Tensor | number, targets: Tensor | number): number {
        const prediction = typeof predictions === "number" ? predictions : predictions.data[0] ?? 0;
        const target = typeof targets === "number" ? targets : targets.data[0] ?? 0;

        const epsilon = Math.max(1e-15, Math.min(1 - 1e-15, prediction));
        this.lastPrediction = prediction;
        this.lastTarget = target;

        return -(target * Math.log(epsilon) + (1 - target) * Math.log(1 - epsilon));
    }

    backward(): Tensor {
        const p = Math.max(1e-15, Math.min(1 - 1e-15, this.lastPrediction));
        const grad = p - this.lastTarget;
        return new Tensor(new Float64Array([grad]), [1]);
    }
}