/**
 * Command batching for the WebGPU backend.
 *
 * {@link CommandBatch} accumulates compute dispatches (in a single compute pass where possible) and
 * buffer copies into one `GPUCommandEncoder`. The batch is submitted on readback, on `sync()`, when
 * the uniform ring is exhausted, or once {@link CommandBatch.threshold} dispatches are queued.
 *
 * Buffers freed while referenced by unsubmitted or in-flight work are released back to the pool only
 * after `queue.onSubmittedWorkDone()` for the submission that last touched them resolves.
 */

import type { UniformRing } from "./bufferpool";

/** Callback invoked when a deferred buffer may be reused. */
export type ReleaseFn = (buffer: GPUBuffer) => void;

/** Groups GPU work into a command encoder and tracks completion of each submission. */
export class CommandBatch {
  /** Dispatch count that triggers an automatic submit. */
  threshold = 64;

  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private dispatches = 0;
  private touched = new Set<GPUBuffer>();
  private deferred: GPUBuffer[] = [];
  private inFlight = new Set<Promise<void>>();
  private inFlightBuffers = new Map<GPUBuffer, number>();
  private submitCount = 0;
  private readonly releaseFn: ReleaseFn;

  constructor(private readonly device: GPUDevice, private readonly uniforms: UniformRing, releaseFn: ReleaseFn) {
    this.releaseFn = releaseFn;
  }

  /** Number of dispatches recorded since the last submit. */
  get pending(): number {
    return this.dispatches;
  }

  /** Total submits so far (diagnostics). */
  get submits(): number {
    return this.submitCount;
  }

  /** True if the current unsubmitted batch or any in-flight submission references `buf`. */
  isBusy(buf: GPUBuffer): boolean {
    return this.touched.has(buf) || this.inFlightBuffers.has(buf);
  }

  private ensureEncoder(): GPUCommandEncoder {
    if (!this.encoder) this.encoder = this.device.createCommandEncoder({ label: "kiln-batch" });
    return this.encoder;
  }

  private ensurePass(): GPUComputePassEncoder {
    const enc = this.ensureEncoder();
    if (!this.pass) this.pass = enc.beginComputePass({ label: "kiln-pass" });
    return this.pass;
  }

  private endPass(): void {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }

  /**
   * Reserve a uniform slice for a dispatch. Submits first if the ring is exhausted so that the slice
   * cannot alias one referenced by this batch.
   */
  uniformSlice(data: ArrayBuffer, byteLength: number): { buffer: GPUBuffer; offset: number; size: number } {
    if (this.uniforms.full) this.submit();
    return this.uniforms.write(data, byteLength);
  }

  /** Record a compute dispatch. `buffers` are the storage buffers it touches (for lifetime tracking). */
  dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, x: number, y = 1, z = 1, buffers: readonly GPUBuffer[] = []): void {
    const pass = this.ensurePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
    for (const b of buffers) this.touched.add(b);
    this.dispatches++;
    if (this.dispatches >= this.threshold) this.submit();
  }

  /** Record a buffer-to-buffer copy (ends the current compute pass). Sizes/offsets must be 4-byte aligned. */
  copyBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, byteLength: number): void {
    if (byteLength <= 0) return;
    this.endPass();
    const enc = this.ensureEncoder();
    enc.copyBufferToBuffer(src, srcOffset, dst, dstOffset, byteLength);
    this.touched.add(src);
    this.touched.add(dst);
    this.dispatches++;
  }

  /** Zero `byteLength` bytes of `buf` (ends the current compute pass). */
  clearBuffer(buf: GPUBuffer, offset: number, byteLength: number): void {
    if (byteLength <= 0) return;
    this.endPass();
    const enc = this.ensureEncoder();
    enc.clearBuffer(buf, offset, byteLength);
    this.touched.add(buf);
    this.dispatches++;
  }

  /**
   * Release `buf` to the pool once no recorded or in-flight work references it. Returns immediately when
   * the buffer is idle.
   */
  deferRelease(buf: GPUBuffer): void {
    if (!this.isBusy(buf)) {
      this.releaseFn(buf);
      return;
    }
    this.deferred.push(buf);
  }

  /**
   * Submit everything recorded so far. Returns a promise that resolves when the GPU has finished this
   * submission (and all earlier ones). Safe to call with nothing recorded.
   */
  submit(): Promise<void> {
    this.endPass();
    const deferred = this.deferred;
    this.deferred = [];
    if (!this.encoder) {
      // Nothing recorded; deferred buffers may still be tied to in-flight work.
      if (deferred.length) this.attachDeferred(this.lastSubmission(), deferred);
      return this.lastSubmission();
    }
    const cmd = this.encoder.finish();
    this.encoder = null;
    this.device.queue.submit([cmd]);
    this.submitCount++;
    this.dispatches = 0;
    this.uniforms.reset();

    const touched = this.touched;
    this.touched = new Set();
    for (const b of touched) this.inFlightBuffers.set(b, (this.inFlightBuffers.get(b) ?? 0) + 1);

    const done: Promise<void> = this.device.queue.onSubmittedWorkDone().then(
      () => {
        this.inFlight.delete(done);
        for (const b of touched) {
          const c = (this.inFlightBuffers.get(b) ?? 1) - 1;
          if (c <= 0) this.inFlightBuffers.delete(b);
          else this.inFlightBuffers.set(b, c);
        }
      },
      () => {
        this.inFlight.delete(done);
      },
    );
    this.inFlight.add(done);
    this.attachDeferred(done, deferred);
    return done;
  }

  private lastSubmission(): Promise<void> {
    const all = [...this.inFlight];
    return all.length ? Promise.all(all).then(() => undefined) : Promise.resolve();
  }

  private attachDeferred(done: Promise<void>, deferred: GPUBuffer[]): void {
    if (!deferred.length) return;
    void done.then(() => {
      for (const b of deferred) {
        if (this.isBusy(b)) this.deferred.push(b);
        else this.releaseFn(b);
      }
    });
  }

  /** Submit and wait for all in-flight GPU work. */
  async sync(): Promise<void> {
    await this.submit();
    await this.lastSubmission();
  }
}
