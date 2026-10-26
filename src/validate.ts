import { ValidationEpoch } from "./training-loop"
export function detectOverfitting(
    history: ValidationEpoch[],
    window = 3 
){
    if (history.length < window){
        return false

    }
    const recent = history.slice(-window)
    const earlier = history[history.length - window - 1]
    const last = recent[recent.length -1]

}