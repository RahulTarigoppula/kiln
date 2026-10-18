import { describe, expect, it } from "vitest";
import * as K from "../src/backend/webgpu/kernels/index";
import type { UnaryOp, BinaryOp, ReduceOp } from "../src/backend/types";
import { hashNormal as cpuHashNormal, hashUniform as cpuHashUniform } from "../src/tensor/random";

const UNARY_OPS: UnaryOp[] = [
  "neg", "exp", "log", "sqrt", "rsqrt", "tanh", "sigmoid", "relu", "gelu", "square", "abs", "recip", "sin", "cos",
  "scale", "addScalar", "powScalar", "geluGrad", "tanhGrad", "sigmoidGrad", "reluGrad", "clamp",
];
const BINARY_OPS: BinaryOp[] = ["add", "sub", "mul", "div", "pow", "max", "min", "eq", "lt", "gt", "le", "ge"];
const REDUCE_OPS: ReduceOp[] = ["sum", "max", "min"];

/** Count braces/parens to catch obviously unbalanced WGSL. */
function balanced(code: string, open: string, close: string): boolean {
  let depth = 0;
  for (const ch of code) {
    if (ch === open) depth++;
    else if (ch === close) depth--;
    if (depth < 0) return false;
  }
  return depth === 0;
}

describe("WGSL templates", () => {
  it("every raw template is non-empty and declares a compute entry point", () => {
    const names = Object.keys(K.KERNEL_TEMPLATES);
    expect(names.length).toBeGreaterThanOrEqual(25);
    for (const name of names) {
      const code = K.KERNEL_TEMPLATES[name] as string;
      expect(code.length, name).toBeGreaterThan(50);
      expect(code, name).toContain("@compute");
      expect(code, name).toContain("fn main(");
      expect(code, name).toContain("@group(0) @binding(0) var<uniform>");
      expect(code, name).not.toContain("f16");
    }
  });

  it("substitute() fills placeholders and rejects missing ones", () => {
    expect(K.substitute("a {{X}} b {{Y}}", { X: 1, Y: "two" })).toBe("a 1 b two");
    expect(() => K.substitute("{{MISSING}}", {})).toThrow(/MISSING/);
    expect(K.accessVars(["read", "read_write"])).toEqual({ A0: "read", A1: "read_write" });
  });

  it("every builder specialises without leftover placeholders and with matching access modes", () => {
    const builders = [
      ...UNARY_OPS.map((op) => K.unaryKernel(op, 256)),
      ...BINARY_OPS.map((op) => K.binaryKernel(op, 256)),
      K.copyStridedKernel(256),
      K.fillKernel(256),
      K.whereKernel(256),
      K.maskedFillKernel(256),
      K.matmulKernel(),
      ...REDUCE_OPS.flatMap((op) => [K.reduceRowsKernel(op, 256), K.reduceGeneralKernel(op, 256)]),
      K.argmaxKernel(256),
      K.sumSquaresKernel(256),
      K.softmaxKernel(256),
      K.softmaxBackwardKernel(256),
      K.layerNormKernel(256),
      K.layerNormBackwardXKernel(256),
      K.layerNormGbPartialKernel(256),
      K.layerNormGbFinalKernel(256),
      K.gatherKernel(256),
      K.scatterHistKernel(256),
      K.scanExclusiveKernel(256),
      K.scatterPlaceKernel(256),
      K.scatterRankKernel(256),
      K.scatterSegsumKernel(),
      K.crossEntropyKernel(256, false),
      K.crossEntropyKernel(256, true),
      K.adamwKernel(256),
      K.rngKernel("rand", 256),
      K.rngKernel("randn", 256),
      K.rngKernel("dropoutMask", 256),
    ];
    const keys = new Set<string>();
    for (const b of builders) {
      const def = b.build(b.defaultAccess);
      expect(def.key).toBe(b.key(b.defaultAccess));
      expect(keys.has(def.key), `duplicate key ${def.key}`).toBe(false);
      keys.add(def.key);
      expect(def.code).not.toMatch(/\{\{[A-Z0-9_]+\}\}/);
      expect(def.code).toContain("@compute");
      expect(balanced(def.code, "{", "}"), b.name).toBe(true);
      expect(balanced(def.code, "(", ")"), b.name).toBe(true);
      // storage bindings 1..k exist with the declared access
      def.storage.forEach((acc, i) => {
        const re = new RegExp(`@binding\\(${i + 1}\\) var<storage, ${acc}>`);
        expect(def.code, `${b.name} binding ${i + 1}`).toMatch(re);
      });
      // no extra storage binding beyond the declared count
      expect(def.code).not.toContain(`@binding(${def.storage.length + 1})`);
      // aliasing promotion: all read_write must also build
      const rw = b.build(b.defaultAccess.map(() => "read_write" as const));
      expect(rw.code).not.toContain("var<storage, read>");
      if (b.defaultAccess.includes("read")) expect(rw.key).not.toBe(def.key);
      else expect(rw.key).toBe(def.key);
    }
  });

  it("unary/binary expression tables cover every op", () => {
    for (const op of UNARY_OPS) expect(K.UNARY_EXPR[op], op).toBeTruthy();
    for (const op of BINARY_OPS) expect(K.BINARY_EXPR[op], op).toBeTruthy();
    expect(() => K.unaryKernel("nope" as UnaryOp, 256)).toThrow();
  });

  it("reference hash RNG matches the tensor layer's CPU implementation", () => {
    for (const seed of [0, 1, 1234, 0xdeadbeef]) {
      for (let i = 0; i < 64; i++) {
        expect(K.hashUniform(seed, i)).toBe(cpuHashUniform(seed, i));
        expect(K.hashNormal(seed, i)).toBeCloseTo(cpuHashNormal(seed, i), 12);
      }
    }
    expect(K.splitmix32(0)).toBe(K.splitmix32(0));
    expect(K.hashUniform(7, 3)).toBeGreaterThanOrEqual(0);
    expect(K.hashUniform(7, 3)).toBeLessThan(1);
  });

  it("matmul constants are consistent with the 256-thread layout", () => {
    expect(K.MATMUL_TILE).toBe(64);
    expect((K.MATMUL_TILE * K.MATMUL_BK) % 256).toBe(0);
    expect(K.MATMUL_TEMPLATE).toContain("@workgroup_size(16, 16, 1)");
  });
});
