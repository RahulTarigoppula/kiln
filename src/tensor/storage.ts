/**
 * Reference-counted ownership of a backend buffer. Views share one Storage; the buffer is returned to the
 * backend when the last Tensor referencing it calls `free()`.
 */
import type { Backend, Buf, Device, DType } from "../backend/types";
import { dtypeSize } from "./dtype";

/** Refcounted wrapper around a backend `Buf`. */
export class Storage {
  readonly backend: Backend;
  readonly buf: Buf;
  readonly dtype: DType;
  /** Number of elements the buffer was allocated for. */
  readonly numel: number;
  private refs = 1;
  private freed = false;

  constructor(backend: Backend, buf: Buf, dtype: DType, numel: number) {
    this.backend = backend;
    this.buf = buf;
    this.dtype = dtype;
    this.numel = numel;
  }

  /** Allocate uninitialised storage for `numel` elements (at least 4 bytes so GPU bindings stay valid). */
  static alloc(backend: Backend, dtype: DType, numel: number): Storage {
    const bytes = Math.max(4, numel * dtypeSize(dtype));
    return new Storage(backend, backend.alloc(bytes), dtype, numel);
  }

  get device(): Device {
    return this.backend.device;
  }

  get byteLength(): number {
    return this.buf.byteLength;
  }

  /** Current reference count (for tests/debugging). */
  get refCount(): number {
    return this.refs;
  }

  /** True after the buffer was handed back to the backend. */
  get isFreed(): boolean {
    return this.freed;
  }

  /** Add a reference (a new view). */
  retain(): this {
    if (this.freed) throw new Error("Storage already freed");
    this.refs++;
    return this;
  }

  /** Drop a reference; frees the backend buffer when it reaches zero. */
  release(): void {
    if (this.freed) return;
    this.refs--;
    if (this.refs <= 0) {
      this.freed = true;
      this.backend.free(this.buf);
    }
  }
}
