/**
 * Training Web Worker: WebGPU runs here so the UI thread never stalls.
 * All logic lives in `handler.ts` (shared with the main-thread fallback).
 */
import { createTrainHandler } from "./handler";
import type { TrainEvent, TrainRequest } from "./protocol";

const scope = self as unknown as { postMessage(msg: TrainEvent, transfer?: Transferable[]): void; onmessage: ((ev: MessageEvent<TrainRequest>) => void) | null };

const handler = createTrainHandler((ev, transfer) => (transfer ? scope.postMessage(ev, transfer) : scope.postMessage(ev)), true);

scope.onmessage = (ev: MessageEvent<TrainRequest>) => {
  handler.handle(ev.data);
};
