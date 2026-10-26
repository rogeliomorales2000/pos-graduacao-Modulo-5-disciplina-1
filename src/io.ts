import { defaultCnnConfig, TinyCNN } from "./tiny-cnn";
import fs from "node:fs"
import path from "node:path"
type TreinedModelArtifact = {
    config : ReturnType<typeof defaultCnnConfig>,
    parameters : Record<string,number[]>
}

export function saveModel(
    model : TinyCNN,
    file: string
){
    const artifact: TreinedModelArtifact = {
        config : {...model.config},
        parameters: {}
    }

    for (const p of model.parameters()) {
        if (typeof p.data === "number") {
            artifact.parameters[p.name] = [p.data];
            continue;
        }

        artifact.parameters[p.name] = Array.from(p.data.data);
    }

    //void file;
    fs.mkdirSync(path.dirname(file),{recursive : true})
    fs.writeFileSync(file,JSON.stringify(artifact,null,2))

}
export function loadModel(file:string): TinyCNN {
    const artifact = JSON.parse(
        fs.readFileSync(file, "utf-8")
    ) as {
        config: ReturnType<typeof defaultCnnConfig>;
        parameters: Record<string, number[]>;
    };

    const model = new TinyCNN(artifact.config);

    for (const p of model.parameters()) {
        const values = artifact.parameters[p.name];
        if (!values) {
            throw new Error(`Erro artifact model missing ${p.name} in ${file}`);
        }

        if (typeof p.data === "number") {
            p.data = values[0] ?? 0;
            continue;
        }

        for (let i = 0; i < values.length; i++) {
            p.data.data[i] = values[i]!;
        }
    }

    return model;
}