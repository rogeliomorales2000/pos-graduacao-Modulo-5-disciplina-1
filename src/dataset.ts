import fs from "node:fs";
import path from "node:path";
import { isSupportedImageFile, loadImageAsTensor } from "./imagem-loader";
import { Tensor } from "./tensor";

export type SplitName = "train" | "validation" | "test";
export type ClassName = "cats" | "dogs";

export interface Sample {
path: string;
label: number; // 0 = cat, 1 = dog
className: ClassName;
}

export interface DatasetSplit {
name: SplitName;
samples: Sample[];
}

export class ImageDataset {
root: string;
imageSize: number;

constructor(root: string, imageSize = 32) {
this.root = root;
this.imageSize = imageSize;
}

/** Lista amostras de um split sem carregar pixels (só metadados). */
list(split: SplitName): DatasetSplit {
const samples: Sample[] = [];
for (const className of ["cats", "dogs"] as ClassName[]) {
const dir = path.join(this.root, split, className);
if (!fs.existsSync(dir)) continue;
const label = className === "cats" ? 0 : 1;
for (const file of fs.readdirSync(dir).sort()) {
if (!isSupportedImageFile(file)) continue;
samples.push({
path: path.join(dir, file),
label,
className,
});
}
}
return { name: split, samples };
}

/** Carrega uma amostra em tensor + label — só quando for treinar/avaliar. */
async loadSample(sample: Sample): Promise<{ image: Tensor; label: Tensor }> {
const image = await loadImageAsTensor(sample.path, this.imageSize);
const label = Tensor.fromMatrix([[sample.label]]);
return { image, label };
}

/** Embaralha cópia do array (Fisher–Yates) — ordem diferente a cada época. */
static shuffled(samples: Sample[], seed = 123): Sample[] {
const arr = [...samples];
let state = seed;
for (let i = arr.length - 1; i > 0; i--) {
state = (state * 1664525 + 1013904223) % 4294967296;
const j = state % (i + 1);
[arr[i], arr[j]] = [arr[j]!, arr[i]!];
}
return arr;
}

describe(): string {
const lines = ["Dataset splits", "--------------"];
for (const split of ["train", "validation", "test"] as SplitName[]) {
const s = this.list(split);
const cats = s.samples.filter((x) => x.className === "cats").length;
const dogs = s.samples.filter((x) => x.className === "dogs").length;
lines.push(
`${split.padEnd(12)} cats=${cats} dogs=${dogs} total=${s.samples.length}`,
);
}
lines.push("");
lines.push("Remember: train updates weights; validation watches; test is final.");
return lines.join("\n");
}
}