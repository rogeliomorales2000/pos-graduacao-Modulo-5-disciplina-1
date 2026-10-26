import fs from "node:fs";
import path from "node:path";
import { PNG } from "pngjs";
import jpeg from "jpeg-js";
import decodeHeic from "heic-decode";
import { Tensor } from "./tensor";

export interface RgbImage {
width: number;
height: number;
/** RGB interleaved (sem alpha), valores 0–255 — length = width × height × 3 */
rgb: Uint8Array;
}

export const SUPPORTED_IMAGE_EXTENSIONS = [
".png", ".jpg", ".jpeg", ".heic", ".heif",
] as const;

export function isSupportedImageFile(fileName: string): boolean {
const ext = path.extname(fileName).toLowerCase();
return (SUPPORTED_IMAGE_EXTENSIONS as readonly string[]).includes(ext);
}

/** Lê PNG, JPEG ou HEIC/HEIF do disco → buffer RGB. */
export async function loadRgbImage(filePath: string): Promise<RgbImage> {
const ext = path.extname(filePath).toLowerCase();
const bytes = fs.readFileSync(filePath);

if (ext === ".png") return decodePng(bytes);
if (ext === ".jpg" || ext === ".jpeg") return decodeJpeg(bytes);
if (ext === ".heic" || ext === ".heif") return decodeHeicToRgb(bytes);

throw new Error(
`Unsupported image type: ${ext}. Supported: ${SUPPORTED_IMAGE_EXTENSIONS.join(", ")}`,
);
}

function decodePng(bytes: Buffer): RgbImage {
const png = PNG.sync.read(bytes);
return rgbaToRgb(png.width, png.height, png.data);
}

function decodeJpeg(bytes: Buffer): RgbImage {
const decoded = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: false });
const { width, height, data } = decoded;
if (data.length === width * height * 3) {
return { width, height, rgb: data as Uint8Array };
}
return rgbaToRgb(width, height, data);
}

async function decodeHeicToRgb(bytes: Buffer): Promise<RgbImage> {
const { width, height, data } = await decodeHeic({ buffer: bytes });
return rgbaToRgb(width, height, data);
}

/** Remove canal alpha — todos os formatos viram o mesmo layout RGB. */
function rgbaToRgb(
width: number,
height: number,
rgba: ArrayLike<number>,
): RgbImage {
const rgb = new Uint8Array(width * height * 3);
for (let i = 0, j = 0; i < width * height * 4; i += 4, j += 3) {
rgb[j] = rgba[i]!;
rgb[j + 1] = rgba[i + 1]!;
rgb[j + 2] = rgba[i + 2]!;
}
return { width, height, rgb };
}

/** Redimensiona com nearest neighbor — cada pixel de saída copia o vizinho mais próximo. */
export function resizeNearest(
image: RgbImage,
targetW: number,
targetH: number,
): RgbImage {
const rgb = new Uint8Array(targetW * targetH * 3);
for (let y = 0; y < targetH; y++) {
for (let x = 0; x < targetW; x++) {
const srcX = Math.min(
image.width - 1,
Math.floor((x / targetW) * image.width),
);
const srcY = Math.min(
image.height - 1,
Math.floor((y / targetH) * image.height),
);
const src = (srcY * image.width + srcX) * 3;
const dst = (y * targetW + x) * 3;
rgb[dst] = image.rgb[src]!;
rgb[dst + 1] = image.rgb[src + 1]!;
rgb[dst + 2] = image.rgb[src + 2]!;
}
}
return { width: targetW, height: targetH, rgb };
}

/** Converte bytes RGB (0–255) → tensor channel-first [3, H, W] normalizado (0–1). */
export function imageToTensor(image: RgbImage): Tensor {
const { width: w, height: h, rgb } = image;
const data = new Float64Array(3 * h * w);
for (let y = 0; y < h; y++) {
for (let x = 0; x < w; x++) {
const src = (y * w + x) * 3;
// Channel-first: bloco R, depois G, depois B
data[0 * h * w + y * w + x] = rgb[src]! / 255;
data[1 * h * w + y * w + x] = rgb[src + 1]! / 255;
data[2 * h * w + y * w + x] = rgb[src + 2]! / 255;
}
}
return new Tensor(data, [3, h, w]);
}

/** Pipeline completo: PNG / JPEG / HEIC → tensor [3, H, W] com valores 0–1. */
export async function loadImageAsTensor(
filePath: string,
size = 32,
): Promise<Tensor> {
const raw = await loadRgbImage(filePath);
const resized = resizeNearest(raw, size, size);
return imageToTensor(resized);
}