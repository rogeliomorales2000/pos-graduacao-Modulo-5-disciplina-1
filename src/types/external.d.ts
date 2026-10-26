declare module "pngjs" {
  export const PNG: {
    sync: {
      read: (buffer: Buffer) => {
        width: number;
        height: number;
        data: Buffer;
      };
    };
  };
}

declare module "jpeg-js" {
  const jpeg: {
    decode: (buffer: Buffer, options?: any) => {
      width: number;
      height: number;
      data: Uint8Array | Buffer | number[];
    };
  };

  export default jpeg;
}

declare module "heic-decode" {
  function decode(options: { buffer: Buffer }): Promise<{
    width: number;
    height: number;
    data: Uint8Array | Buffer | number[];
  }>;

  export default decode;
}
