/**
 * GPU buffer pooling for the WebGPU backend.
 *
 * - {@link BufferPool}: size-class pool of STORAGE|COPY_SRC|COPY_DST buffers. Sizes are rounded to
 *   256-byte multiples (powers of two below 4 MiB, 1 MiB multiples above) so freed buffers are reusable.
 * - {@link StagingRing}: MAP_READ|COPY_DST buffers for readback via `mapAsync`.
 * - {@link UniformRing}: one large UNIFORM|COPY_DST buffer carved into 256-byte-aligned slices that
 *   receive per-dispatch parameters via `queue.writeBuffer`.
 */

/** Round `bytes` up to the pool's size class (always a multiple of 256, at least 256). */
export function sizeClass(bytes: number): number {
  const b = Math.max(256, Math.ceil(bytes / 256) * 256);
  const MiB = 1 << 20;
  if (b <= 4 * MiB) {
    let p = 256;
    while (p < b) p *= 2;
    return p;
  }
  return Math.ceil(b / MiB) * MiB;
}

/** Statistics exposed for the Bench/Studio memory readouts. */
export interface PoolStats {
  /** Bytes currently handed out to callers. */
  liveBytes: number;
  /** Number of buffers currently handed out to callers. */
  liveBuffers: number;
  /** Bytes sitting idle in the pool. */
  pooledBytes: number;
  /** Total `createBuffer` calls. */
  created: number;
}

/** Size-class pool of storage buffers. */
export class BufferPool {
  private readonly free = new Map<number, GPUBuffer[]>();
  private liveBytes = 0;
  private liveBuffers = 0;
  private pooledBytes = 0;
  private created = 0;
  /** Idle bytes allowed before extra freed buffers are destroyed instead of pooled. */
  maxPooledBytes = 512 * 1024 * 1024;

  constructor(private readonly device: GPUDevice) {}

  /** Get a buffer whose size is `sizeClass(byteLength)`. */
  alloc(byteLength: number): GPUBuffer {
    const size = sizeClass(byteLength);
    const list = this.free.get(size);
    let buf = list?.pop();
    if (buf) {
      this.pooledBytes -= size;
    } else {
      buf = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: `kiln-storage-${size}`,
      });
      this.created++;
    }
    this.liveBytes += size;
    this.liveBuffers++;
    return buf;
  }

  /** Return a buffer to the pool. The caller guarantees no in-flight GPU work references it. */
  release(buf: GPUBuffer): void {
    const size = buf.size;
    this.liveBytes -= size;
    this.liveBuffers--;
    if (this.pooledBytes + size > this.maxPooledBytes) {
      buf.destroy();
      return;
    }
    let list = this.free.get(size);
    if (!list) {
      list = [];
      this.free.set(size, list);
    }
    list.push(buf);
    this.pooledBytes += size;
  }

  /** Destroy every pooled (idle) buffer. */
  drain(): void {
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
    this.pooledBytes = 0;
  }

  stats(): PoolStats {
    return { liveBytes: this.liveBytes, liveBuffers: this.liveBuffers, pooledBytes: this.pooledBytes, created: this.created };
  }
}

/** Pool of mappable staging buffers for readback. */
export class StagingRing {
  private readonly free = new Map<number, GPUBuffer[]>();
  private idle = 0;
  /** Maximum idle staging buffers kept alive. */
  maxIdle = 8;

  constructor(private readonly device: GPUDevice) {}

  /** Acquire a MAP_READ|COPY_DST buffer with size ≥ byteLength (multiple of 256). */
  acquire(byteLength: number): GPUBuffer {
    const size = sizeClass(byteLength);
    const list = this.free.get(size);
    const buf = list?.pop();
    if (buf) {
      this.idle--;
      return buf;
    }
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: `kiln-staging-${size}`,
    });
  }

  /** Give back an unmapped staging buffer. */
  release(buf: GPUBuffer): void {
    if (this.idle >= this.maxIdle) {
      buf.destroy();
      return;
    }
    let list = this.free.get(buf.size);
    if (!list) {
      list = [];
      this.free.set(buf.size, list);
    }
    list.push(buf);
    this.idle++;
  }

  drain(): void {
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
    this.idle = 0;
  }
}

/** A 256-byte-aligned slice of the uniform ring. */
export interface UniformSlice {
  buffer: GPUBuffer;
  offset: number;
  size: number;
}

/**
 * Ring of uniform slices. `write()` copies parameter bytes into the next free slice via `queue.writeBuffer`.
 * Because queue operations are ordered, slices used by already *submitted* work may be reused immediately;
 * the ring only has to avoid clobbering slices referenced by the current, unsubmitted command batch, so the
 * owner must call {@link UniformRing.reset} right after each submit, and submit when {@link UniformRing.full}.
 */
export class UniformRing {
  readonly buffer: GPUBuffer;
  readonly sliceSize: number;
  private readonly slots: number;
  private next = 0;

  constructor(private readonly device: GPUDevice, slots = 512, sliceSize = 256) {
    const align = Math.max(256, device.limits.minUniformBufferOffsetAlignment || 256);
    this.sliceSize = Math.ceil(sliceSize / align) * align;
    this.slots = slots;
    this.buffer = device.createBuffer({
      size: this.sliceSize * slots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "kiln-uniform-ring",
    });
  }

  /** True when no slice is left before a reset. */
  get full(): boolean {
    return this.next >= this.slots;
  }

  /** Upload `data` (≤ sliceSize bytes) into the next slice. Caller must check {@link full} first. */
  write(data: ArrayBuffer, byteLength: number): UniformSlice {
    if (this.next >= this.slots) throw new Error("UniformRing: ring exhausted; submit and reset first.");
    const offset = this.next * this.sliceSize;
    this.next++;
    const len = Math.ceil(byteLength / 16) * 16;
    this.device.queue.writeBuffer(this.buffer, offset, data, 0, Math.min(len, this.sliceSize));
    return { buffer: this.buffer, offset, size: this.sliceSize };
  }

  /** Make every slice available again (call right after `queue.submit`). */
  reset(): void {
    this.next = 0;
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
