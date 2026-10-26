import Tensor from "./tensor";

export function randomMatrix(rows: number, cols: number, seed: number): Tensor {
    let state = seed;
    const data: number[] = [];

    for (let i = 0; i < rows * cols; i++) {
        state = (state * 1664525 + 1013904223) % 4294967296;
        const u = state / 4294967296;
        data.push((u - 0.5) * 0.5);
    }

    return new Tensor(new Float64Array(data), [rows, cols]);
}
