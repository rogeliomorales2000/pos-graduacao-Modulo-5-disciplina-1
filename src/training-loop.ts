import { Adam } from "./adam";
import { ImageDataset, Sample } from "./dataset";
import { TinyCNN } from "./tiny-cnn";

export interface ValidationEpoch {
epoch: number;
trainLoss: number;
trainAccuracy: number;
valLoss: number;
valAccuracy: number;
}

export function detectOverfitting(
    history: ValidationEpoch[],
    window = 3,
) {
    if (history.length < window) {
        return false;
    }

    const recent = history.slice(-window);
    const earlier = history[history.length - window - 1];
    const last = recent[recent.length - 1];

    return last.trainLoss < earlier.trainLoss && last.valLoss > earlier.valLoss;
}

/** Avalia sem atualizar pesos — só forward + métricas. */
export async function evaluate(
model: TinyCNN,
dataset: ImageDataset,
samples: Sample[],
): Promise<{ loss: number; accuracy: number }> {
let totalLoss = 0;
let correct = 0;

for (const sample of samples) {
const { image, label } = await dataset.loadSample(sample);
const prediction = model.forward(image);
// Loss forward-only — sem backward, sem optimizer.step
totalLoss += model.lossFn.forward(prediction, label);
const predictedClass = prediction.get(0, 0) >= 0.5 ? 1 : 0;
if (predictedClass === sample.label) correct += 1;
}

return {
loss: totalLoss / Math.max(1, samples.length),
accuracy: correct / Math.max(1, samples.length),
};
}

export async function trainCnn(
model: TinyCNN,
dataset: ImageDataset,
trainSamples: Sample[],
valSamples: Sample[],
options: {
epochs: number;
learningRate?: number;
seed?: number;
},
): Promise<ValidationEpoch[]> {
const optimizer = new Adam(options.learningRate ?? 0.001);
const history: ValidationEpoch[] = [];

for (let epoch = 1; epoch <= options.epochs; epoch++) {
const order = ImageDataset.shuffled(
trainSamples,
(options.seed ?? 0) + epoch,
);

// --- Fase TREINO: pesos ATUALIZAM
for (const sample of order) {
const { image, label } = await dataset.loadSample(sample);
const prediction = model.forward(image);
model.backward(prediction, label);
optimizer.step(model.parameters());
}

// --- Fase AVALIAÇÃO: forward ONLY — sem backward, sem step
const train = await evaluate(model, dataset, trainSamples);
const val = await evaluate(model, dataset, valSamples);

const row: ValidationEpoch = {
epoch,
trainLoss: train.loss,
trainAccuracy: train.accuracy,
valLoss: val.loss,
valAccuracy: val.accuracy,
};
history.push(row);

console.log(`Epoch ${epoch}/${options.epochs}`);
console.log(
` train loss=${row.trainLoss.toFixed(4)} acc=${(row.trainAccuracy * 100).toFixed(1)}%`,
);
console.log(
` val loss=${row.valLoss.toFixed(4)} acc=${(row.valAccuracy * 100).toFixed(1)}%`,
);

if (detectOverfitting(history)) {
console.log("Model has overfitted!")
}
if(detectOverfitting(history)){
    console.log(`Model has overfited`)
}
}

return history;
}

const trainingLoop = { detectOverfitting, evaluate, trainCnn };
export default trainingLoop;