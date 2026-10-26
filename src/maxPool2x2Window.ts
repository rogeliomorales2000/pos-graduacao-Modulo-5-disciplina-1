import { Nullable } from "./types";

export function masPool2x2Window(

    window : number [] [] 
): number { 
    let m = -Infinity;

    for (
        const row of window
    ){
        for(
            const v of row
        ){
            if (
                v > m
            ){
                m = v ;
            }
        }


    }
    return m 

}


export class MaxPool2D {
    private lastArgMax: Nullable<{
        c:number,
        i: number,
        j: number
    }[]>| null = null
    private lastInputShape: Nullable<number[]>= null
    constructor(
        private poolSize = 2,
        private stride ?: number
    ) {}

}
