/**
 * Barrel for the WGSL kernel modules plus a registry of every raw template (used by the WGSL syntax
 * test and the Bench page).
 */

export * from "./common";
export * from "./unary";
export * from "./binary";
export * from "./elementwise";
export * from "./matmul";
export * from "./reduce";
export * from "./softmax";
export * from "./layernorm";
export * from "./indexing";
export * from "./loss_optim";
export * from "./rng";

import { UNARY_TEMPLATE } from "./unary";
import { BINARY_TEMPLATE } from "./binary";
import { COPY_STRIDED_TEMPLATE, FILL_TEMPLATE, MASKED_FILL_TEMPLATE, WHERE_TEMPLATE } from "./elementwise";
import { MATMUL_TEMPLATE } from "./matmul";
import { ARGMAX_TEMPLATE, REDUCE_GENERAL_TEMPLATE, REDUCE_ROWS_TEMPLATE, SUM_SQUARES_TEMPLATE } from "./reduce";
import { SOFTMAX_BACKWARD_TEMPLATE, SOFTMAX_TEMPLATE } from "./softmax";
import {
  LAYERNORM_BACKWARD_X_TEMPLATE,
  LAYERNORM_GB_FINAL_TEMPLATE,
  LAYERNORM_GB_PARTIAL_TEMPLATE,
  LAYERNORM_TEMPLATE,
} from "./layernorm";
import {
  GATHER_TEMPLATE,
  SCAN_EXCLUSIVE_TEMPLATE,
  SCATTER_HIST_TEMPLATE,
  SCATTER_PLACE_TEMPLATE,
  SCATTER_RANK_TEMPLATE,
  SCATTER_SEGSUM_TEMPLATE,
} from "./indexing";
import { ADAMW_TEMPLATE, CROSS_ENTROPY_GRAD_TEMPLATE, CROSS_ENTROPY_TEMPLATE } from "./loss_optim";
import { RNG_TEMPLATE } from "./rng";

/** Every raw WGSL template keyed by kernel name. */
export const KERNEL_TEMPLATES: Readonly<Record<string, string>> = {
  unary: UNARY_TEMPLATE,
  binary_broadcast: BINARY_TEMPLATE,
  copy_strided: COPY_STRIDED_TEMPLATE,
  fill: FILL_TEMPLATE,
  where: WHERE_TEMPLATE,
  masked_fill: MASKED_FILL_TEMPLATE,
  matmul_tiled: MATMUL_TEMPLATE,
  reduce_rows: REDUCE_ROWS_TEMPLATE,
  reduce_general: REDUCE_GENERAL_TEMPLATE,
  argmax_rows: ARGMAX_TEMPLATE,
  sum_squares_partial: SUM_SQUARES_TEMPLATE,
  softmax_rows: SOFTMAX_TEMPLATE,
  softmax_backward: SOFTMAX_BACKWARD_TEMPLATE,
  layernorm_rows: LAYERNORM_TEMPLATE,
  layernorm_backward_x: LAYERNORM_BACKWARD_X_TEMPLATE,
  layernorm_gb_partial: LAYERNORM_GB_PARTIAL_TEMPLATE,
  layernorm_gb_final: LAYERNORM_GB_FINAL_TEMPLATE,
  gather: GATHER_TEMPLATE,
  scatter_hist: SCATTER_HIST_TEMPLATE,
  scan_exclusive: SCAN_EXCLUSIVE_TEMPLATE,
  scatter_place: SCATTER_PLACE_TEMPLATE,
  scatter_rank: SCATTER_RANK_TEMPLATE,
  scatter_segsum: SCATTER_SEGSUM_TEMPLATE,
  cross_entropy_rows: CROSS_ENTROPY_TEMPLATE,
  cross_entropy_rows_grad: CROSS_ENTROPY_GRAD_TEMPLATE,
  adamw: ADAMW_TEMPLATE,
  rng: RNG_TEMPLATE,
};
