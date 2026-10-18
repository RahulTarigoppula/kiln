/**
 * TrainSession: drives the training handler in a Web Worker, falling back to the main thread when the
 * worker has no WebGPU but the page does (or when workers are unavailable).
 */
import { createTrainHandler, type TrainHandler } from "../worker/handler";
import type { TrainEvent, TrainRequest } from "../worker/protocol";
import { gpuState } from "./gpu";

export type SessionListener = (ev: TrainEvent) => void;

export class TrainSession {
  private worker: Worker | null = null;
  private local: TrainHandler | null = null;
  private listeners = new Set<SessionListener>();
  private disposed = false;
  /** "worker" | "main" once ready. */
  mode: "worker" | "main" | "starting" = "starting";

  constructor(private readonly preferGpu: boolean) {
    this.boot();
  }

  private boot(): void {
    const pageHasGpu = gpuState.peek().backend !== null;
    if (typeof Worker === "undefined") {
      this.useMainThread();
      return;
    }
    try {
      this.worker = new Worker(new URL("../worker/train.worker.ts", import.meta.url), { type: "module", name: "kiln-train" });
    } catch (err: unknown) {
      console.warn("[kiln] worker unavailable, training on the main thread:", err);
      this.useMainThread();
      return;
    }
    this.worker.onmessage = (ev: MessageEvent<TrainEvent>) => {
      const msg = ev.data;
      if (msg.type === "ready" && this.mode === "starting") {
        if (msg.device === "cpu" && pageHasGpu && this.preferGpu) {
          // WebGPU missing inside workers (non-Chromium) → train on the main thread instead.
          this.worker?.terminate();
          this.worker = null;
          this.useMainThread();
          return;
        }
        this.mode = "worker";
      }
      this.emit(msg);
    };
    this.worker.onerror = (e) => {
      this.emit({ type: "error", message: `Worker error: ${e.message}` });
    };
    this.worker.postMessage({ type: "init", preferGpu: this.preferGpu } satisfies TrainRequest);
  }

  private useMainThread(): void {
    this.mode = "main";
    this.local = createTrainHandler((ev) => this.emit(ev), false);
    this.local.handle({ type: "init", preferGpu: this.preferGpu });
  }

  private emit(ev: TrainEvent): void {
    if (this.disposed) return;
    for (const l of this.listeners) l(ev);
  }

  on(listener: SessionListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  send(req: TrainRequest): void {
    if (this.disposed) return;
    if (this.worker) this.worker.postMessage(req);
    else this.local?.handle(req);
  }

  dispose(): void {
    this.disposed = true;
    this.local?.dispose();
    this.worker?.terminate();
    this.worker = null;
    this.listeners.clear();
  }
}
