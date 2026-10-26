import Tensor from "./tensor";
import { TreinableMLP } from "./treinable-mlp";

const net = new TreinableMLP(2, 4, 1, 0.1, 42);

const dataset = [
    { inputs: Tensor.fromMatrix([[0, 0]]), target: Tensor.fromMatrix([[0]]) },
    { inputs: Tensor.fromMatrix([[0, 1]]), target: Tensor.fromMatrix([[1]]) },
    { inputs: Tensor.fromMatrix([[1, 0]]), target: Tensor.fromMatrix([[1]]) },
    { inputs: Tensor.fromMatrix([[1, 1]]), target: Tensor.fromMatrix([[0]]) }
];

for (let epoch = 0; epoch < 200; epoch++) {
    let totalLoss = 0;

    for (const example of dataset) {
        totalLoss += net.trainStep(example);
    }

    const averageLoss = totalLoss / dataset.length;

    if (epoch % 20 === 0 || epoch === 199) {
        console.log(`Epoch ${epoch} | loss = ${averageLoss.toFixed(6)}`);
    }
}

const pred0 = net.forward(Tensor.fromMatrix([[0, 0]])).toArray();
const pred1 = net.forward(Tensor.fromMatrix([[0, 1]])).toArray();
const pred2 = net.forward(Tensor.fromMatrix([[1, 0]])).toArray();
const pred3 = net.forward(Tensor.fromMatrix([[1, 1]])).toArray();

console.log("Predições finais:");
console.log("[0,0] =>", pred0);
console.log("[0,1] =>", pred1);
console.log("[1,0] =>", pred2);
console.log("[1,1] =>", pred3);