/**
 * Shader-module / compute-pipeline cache and bind-group helpers.
 *
 * Every Kiln kernel uses one bind group: binding 0 is the parameter uniform, bindings 1..k are storage
 * buffers whose access modes are declared by the kernel ({@link KernelDef.storage}). Explicit bind-group
 * layouts are built from that declaration so shader access modes and layout entries always agree.
 */

/** Access mode of a storage binding. */
export type Access = "read" | "read_write";

/** A fully specialised kernel ready to be compiled: WGSL source plus binding declaration. */
export interface KernelDef {
  /** Cache key: kernel name + workgroup size + constant overrides + access modes. */
  key: string;
  /** Complete WGSL source with placeholders already substituted. */
  code: string;
  /** Access modes for storage bindings 1..k (binding 0 is always the uniform). */
  storage: readonly Access[];
  /** Entry point name (default `main`). */
  entry?: string;
}

/** Cached pipeline + layout. */
export interface CompiledKernel {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  def: KernelDef;
}

/** Compiles and caches compute pipelines keyed by {@link KernelDef.key}. */
export class PipelineCache {
  private readonly pipelines = new Map<string, CompiledKernel>();
  private readonly layouts = new Map<string, GPUBindGroupLayout>();

  constructor(private readonly device: GPUDevice) {}

  /** Layout for a given storage access signature (shared across kernels with the same signature). */
  layoutFor(storage: readonly Access[]): GPUBindGroupLayout {
    const key = storage.join(",");
    let layout = this.layouts.get(key);
    if (layout) return layout;
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ];
    storage.forEach((acc, i) => {
      entries.push({
        binding: i + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: acc === "read" ? "read-only-storage" : "storage" },
      });
    });
    layout = this.device.createBindGroupLayout({ entries, label: `kiln-layout-${key}` });
    this.layouts.set(key, layout);
    return layout;
  }

  /** Get (compiling on first use) the pipeline for `def`. */
  get(def: KernelDef): CompiledKernel {
    const hit = this.pipelines.get(def.key);
    if (hit) return hit;
    const layout = this.layoutFor(def.storage);
    const module = this.device.createShaderModule({ code: def.code, label: def.key });
    const pipeline = this.device.createComputePipeline({
      label: def.key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: def.entry ?? "main" },
    });
    const compiled: CompiledKernel = { pipeline, layout, def };
    this.pipelines.set(def.key, compiled);
    return compiled;
  }

  /** Look up by precomputed key; build (and compile) via `make` only on a miss. */
  getOrBuild(key: string, make: () => KernelDef): CompiledKernel {
    const hit = this.pipelines.get(key);
    if (hit) return hit;
    const def = make();
    if (def.key !== key) throw new Error(`PipelineCache: key mismatch (${def.key} vs ${key})`);
    return this.get(def);
  }

  /** Number of compiled pipelines (for diagnostics). */
  get size(): number {
    return this.pipelines.size;
  }

  /**
   * Create a bind group for `kernel`: `uniform` goes to binding 0, `buffers[i]` to binding i+1.
   * Each storage entry may bind a sub-range via `offset`/`size` (both in bytes).
   */
  bindGroup(
    kernel: CompiledKernel,
    uniform: { buffer: GPUBuffer; offset: number; size: number },
    buffers: readonly (GPUBuffer | { buffer: GPUBuffer; offset?: number; size?: number })[],
  ): GPUBindGroup {
    if (buffers.length !== kernel.def.storage.length) {
      throw new Error(`bindGroup(${kernel.def.key}): expected ${kernel.def.storage.length} buffers, got ${buffers.length}`);
    }
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: uniform }];
    buffers.forEach((b, i) => {
      const res: GPUBufferBinding = "buffer" in b && !(b instanceof GPUBuffer)
        ? { buffer: b.buffer, offset: b.offset ?? 0, size: b.size ?? b.buffer.size - (b.offset ?? 0) }
        : { buffer: b as GPUBuffer };
      entries.push({ binding: i + 1, resource: res });
    });
    return this.device.createBindGroup({ layout: kernel.layout, entries, label: kernel.def.key });
  }
}
