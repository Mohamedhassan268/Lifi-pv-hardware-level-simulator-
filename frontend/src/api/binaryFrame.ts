/**
 * Binary WebSocket frame decoder for probe data.
 *
 * Wire format (matches backend/transport/binary_frame.py):
 *   [ 4 bytes  big-endian uint32 = header length ]
 *   [ N bytes  UTF-8 JSON header ]
 *   [ M bytes  raw little-endian sample data ]
 *
 * Zero-copy decode: the returned TypedArray views the original ArrayBuffer
 * directly when alignment permits, otherwise it copies into an aligned buffer.
 */

export interface ProbeFrameHeader {
  type: "probe_data";
  id: string;
  dtype: "float64" | "float32" | "uint8" | "int32";
  shape: number[];
  units: string;
  config_hash: string;
  request_id: number;
  stage: string;
  suggest_decimate: boolean;
  [extra: string]: unknown;
}

export type ProbePayload =
  | Float64Array
  | Float32Array
  | Uint8Array
  | Int32Array;

export interface DecodedProbeFrame {
  header: ProbeFrameHeader;
  payload: ProbePayload;
}

const DECODER = new TextDecoder("utf-8");

export function decodeBinaryFrame(buf: ArrayBuffer): DecodedProbeFrame {
  if (buf.byteLength < 4) {
    throw new Error("probe frame too short");
  }
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, /* littleEndian */ false);
  if (headerLen <= 0 || 4 + headerLen > buf.byteLength) {
    throw new Error(`invalid probe-frame header length ${headerLen}`);
  }

  const headerBytes = new Uint8Array(buf, 4, headerLen);
  const header = JSON.parse(DECODER.decode(headerBytes)) as ProbeFrameHeader;

  const sampleOffset = 4 + headerLen;
  const sampleBytes = buf.byteLength - sampleOffset;

  let payload: ProbePayload;
  switch (header.dtype) {
    case "float64": {
      // Float64Array requires 8-byte alignment. The 4-byte length prefix +
      // variable header length will usually break alignment, so we copy
      // into a fresh aligned buffer.
      const out = new Float64Array(sampleBytes / 8);
      new Uint8Array(out.buffer).set(new Uint8Array(buf, sampleOffset, sampleBytes));
      payload = out;
      break;
    }
    case "float32": {
      const out = new Float32Array(sampleBytes / 4);
      new Uint8Array(out.buffer).set(new Uint8Array(buf, sampleOffset, sampleBytes));
      payload = out;
      break;
    }
    case "int32": {
      const out = new Int32Array(sampleBytes / 4);
      new Uint8Array(out.buffer).set(new Uint8Array(buf, sampleOffset, sampleBytes));
      payload = out;
      break;
    }
    case "uint8":
      payload = new Uint8Array(buf.slice(sampleOffset));
      break;
    default:
      throw new Error(`unsupported probe dtype: ${(header as ProbeFrameHeader).dtype}`);
  }

  return { header, payload };
}
