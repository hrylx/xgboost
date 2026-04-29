/**
 * Copyright 2017-2025, XGBoost contributors
 */
#pragma once
#include <thrust/iterator/counting_iterator.h>          // for make_counting_iterator
#include <thrust/iterator/transform_output_iterator.h>  // for make_transform_output_iterator

#include <algorithm>        // for max
#include <cstddef>          // for size_t
#include <cstdint>          // for int32_t, uint32_t
#include <utility>          // for swap
#include <cuda/functional>  // for proclaim_return_type
#include <vector>           // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                   // for bst_idx_t
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
namespace cuda_impl {
using RowIndexT = std::uint32_t;
// TODO(Rory): Can be larger. To be tuned alongside other batch operations.
inline constexpr std::int32_t kMaxUpdatePositionBatchSize = 32;
}  // namespace cuda_impl

/**
 * @brief Used to demarcate a contiguous set of row indices associated with some tree
 *        node.
 */
struct Segment {
  cuda_impl::RowIndexT begin{0};
  cuda_impl::RowIndexT end{0};

  Segment() = default;

  Segment(cuda_impl::RowIndexT begin, cuda_impl::RowIndexT end) : begin(begin), end(end) {
    CHECK_GE(end, begin);
  }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t Size() const { return end - begin; }
};

template <typename OpDataT>
struct PerNodeData {
  Segment segment;
  OpDataT data;
};

/**
 * @param global_thread_idx In practice, the row index within the total number of rows for
 *        this node batch.
 * @param batch_idx The nidx within this node batch (not the actual node index in a tree).
 * @param item_idx The resulting global row index (without accounting for base_rowid). This maps the
 *        row index within the node batch back to the global row index.
 */
template <typename T>
XGBOOST_DEV_INLINE void AssignBatch(dh::LDGIterator<T> const& batch_info_iter,
                                    std::size_t global_thread_idx, int* batch_idx,
                                    std::size_t* item_idx) {
  cuda_impl::RowIndexT sum = 0;
  // Search for the nidx in batch and the corresponding global row index, exit once found.
  for (std::int32_t i = 0; i < cuda_impl::kMaxUpdatePositionBatchSize; i++) {
    if (sum + batch_info_iter[i].segment.Size() > global_thread_idx) {
      *batch_idx = i;
      // the beginning of the segment plus the offset into that segment
      *item_idx = (global_thread_idx - sum) + batch_info_iter[i].segment.begin;
      break;
    }
    sum += batch_info_iter[i].segment.Size();
  }
}

// We can scan over this tuple, where the scan gives us information on how to partition inputs
// according to the flag
struct IndexFlagTuple {
  cuda_impl::RowIndexT idx;        // The location of the item we are working on in ridx_
  cuda_impl::RowIndexT flag_scan;  // This gets populated after scanning
  std::int32_t batch_idx;          // Which node in the batch does this item belong to
  bool flag;                       // Result of op (is this item going left?)
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a, const IndexFlagTuple& b) const {
    // Segmented scan - resets if we cross batch boundaries
    if (a.batch_idx == b.batch_idx) {
      // Accumulate the flags, everything else stays the same
      return {b.idx, a.flag_scan + b.flag_scan, b.batch_idx, b.flag};
    } else {
      return b;
    }
  }
};

// Scatter from `ridx_in` to `ridx_out`.
template <typename OpDataT>
struct WriteResultsFunctor {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info;
  cuda_impl::RowIndexT const* ridx_in;
  cuda_impl::RowIndexT* ridx_out;
  cuda_impl::RowIndexT* counts;

  __device__ IndexFlagTuple operator()(IndexFlagTuple const& x) {
    cuda_impl::RowIndexT scatter_address;
    // Get the segment that this row belongs to.
    const Segment& segment = batch_info[x.batch_idx].segment;
    if (x.flag) {
      // Go left.
      cuda_impl::RowIndexT num_previous_flagged = x.flag_scan - 1;  // -1 because inclusive scan
      scatter_address = segment.begin + num_previous_flagged;
    } else {
      cuda_impl::RowIndexT num_previous_unflagged = (x.idx - segment.begin) - x.flag_scan;
      scatter_address = segment.end - num_previous_unflagged - 1;
    }
    ridx_out[scatter_address] = ridx_in[x.idx];

    if (x.idx == (segment.end - 1)) {
      // Write out counts
      counts[x.batch_idx] = x.flag_scan;
    }

    // Discard
    return {};
  }
};

/**
 * @param d_batch_info Node data, with the size of the input number of nodes.
 *
 * Reads from `ridx_in` and scatters the sorted indices into `ridx_out`.
 * The caller is responsible for ensuring that rows outside any segment in
 * `d_batch_info` already hold the correct values in `ridx_out` before the call
 * (typically via the ping-pong buffers in RowPartitioner).
 */
template <typename OpT, typename OpDataT>
void SortPositionBatch(Context const* ctx, common::Span<const PerNodeData<OpDataT>> d_batch_info,
                       common::Span<cuda_impl::RowIndexT const> ridx_in,
                       common::Span<cuda_impl::RowIndexT> ridx_out,
                       common::Span<cuda_impl::RowIndexT> d_counts, bst_idx_t total_rows, OpT op,
                       dh::DeviceUVector<int8_t>* tmp) {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info_itr(d_batch_info.data());
  WriteResultsFunctor<OpDataT> write_results{batch_info_itr, ridx_in.data(), ridx_out.data(),
                                             d_counts.data()};

  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator = dh::MakeTransformIterator<IndexFlagTuple>(
      counting, cuda::proclaim_return_type<IndexFlagTuple>([=] __device__(std::size_t idx) {
        std::int32_t nidx_in_batch;
        std::size_t item_idx;
        AssignBatch(batch_info_itr, idx, &nidx_in_batch, &item_idx);
        auto go_left = op(ridx_in[item_idx], nidx_in_batch, batch_info_itr[nidx_in_batch].data);
        return IndexFlagTuple{static_cast<cuda_impl::RowIndexT>(item_idx), go_left, nidx_in_batch,
                              go_left};
      }));
  // Reach down to the dispatch function to avoid using int as the offset type.
  std::size_t n_bytes = 0;
  if (tmp->empty()) {
    // The size of temporary storage is calculated based on the total number of
    // rows. Since the root node has all the rows, subsequence allocatioin must be smaller
    // than the root node. As a result, we can calculate this once and reuse it throughout
    // the iteration.
    auto ret =
        cub::DispatchScan<decltype(input_iterator), decltype(discard_write_iterator), IndexFlagOp,
                          cub::NullType, std::uint64_t>::Dispatch(nullptr, n_bytes, input_iterator,
                                                                  discard_write_iterator,
                                                                  IndexFlagOp{}, cub::NullType{},
                                                                  static_cast<std::uint64_t>(
                                                                      total_rows),
                                                                  ctx->CUDACtx()->Stream());
    dh::safe_cuda(ret);
    tmp->resize(n_bytes);
  }
  n_bytes = tmp->size();
  auto ret =
      cub::DispatchScan<decltype(input_iterator), decltype(discard_write_iterator), IndexFlagOp,
                        cub::NullType, std::uint64_t>::Dispatch(tmp->data(), n_bytes,
                                                                input_iterator,
                                                                discard_write_iterator,
                                                                IndexFlagOp{}, cub::NullType{},
                                                                static_cast<std::uint64_t>(
                                                                    total_rows),
                                                                ctx->CUDACtx()->Stream());
  dh::safe_cuda(ret);
}

struct NodePositionInfo {
  Segment segment;
  bst_node_t left_child = -1;
  bst_node_t right_child = -1;
  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf() const { return left_child == -1; }
};

struct LeafInfo {
  bst_node_t nidx;
  NodePositionInfo node;
};

// Flat (segment_begin, leaf_nidx) entry built once per FinalisePosition call.
// Leaf segments tile [0, n_samples) disjointly so a sort by `seg_begin` is unique.
struct LeafBoundary {
  cuda_impl::RowIndexT seg_begin;
  bst_node_t nidx;
};

// Locate the leaf whose segment contains position `idx` via upper_bound − 1 over
// `d_leaves[*].seg_begin`. ceil(log2(n_leaves)) iterations vs the previous tree
// walk's `depth` iterations — same count for a perfectly balanced tree, fewer for
// unbalanced — and one read per iteration instead of two.
XGBOOST_DEV_INLINE bst_node_t FindLeafForPosition(std::size_t idx,
                                                  LeafBoundary const* d_leaves,
                                                  std::int32_t n_leaves) {
  std::int32_t lo = 0;
  std::int32_t hi = n_leaves - 1;
  while (lo < hi) {
    std::int32_t mid = lo + ((hi - lo + 1) >> 1);
    if (static_cast<std::size_t>(d_leaves[mid].seg_begin) <= idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return d_leaves[lo].nidx;
}

// Build an inverse-permutation table inv_ridx so that inv_ridx[d_ridx[i]-base_ridx] == i.
// Used by the reverse-iteration FinalisePosition kernel below to turn a scattered read
// of d_gpair into a sequential one.
template <int kBlockSize>
__global__ __launch_bounds__(kBlockSize) void BuildInverseRidxKernel(
    common::Span<const cuda_impl::RowIndexT> d_ridx, bst_idx_t base_ridx,
    common::Span<cuda_impl::RowIndexT> d_inv_ridx) {
  for (auto idx : dh::GridStrideRange<std::size_t>(0, d_ridx.size())) {
    cuda_impl::RowIndexT ridx_offset = d_ridx[idx] - static_cast<cuda_impl::RowIndexT>(base_ridx);
    d_inv_ridx[ridx_offset] = static_cast<cuda_impl::RowIndexT>(idx);
  }
}

// Reverse-iteration variant: iterate by `ridx` (the row index) instead of by `idx`
// (the position in d_ridx). The previous version had `for idx in [0,N): op(d_ridx[idx])`,
// which made every read of d_gpair (80 MB, ridx-indexed) and write to d_out_position
// (20 MB, ridx-indexed) a scattered access. With inv_ridx pre-built, we can iterate ridx
// linearly: the d_gpair lookup inside `op` becomes a sequential 80 MB stream, and the
// d_out_position write a sequential 20 MB stream. The cost is one extra
// BuildInverseRidxKernel pass with a 20 MB scattered write — much cheaper than the 80
// MB scattered read it eliminates.
template <int kBlockSize, typename OpT>
__global__ __launch_bounds__(kBlockSize) void FinalisePositionKernel(
    common::Span<const LeafBoundary> d_leaves,
    common::Span<const cuda_impl::RowIndexT> d_inv_ridx,
    common::Span<bst_node_t> d_out_position, OpT op) {
  auto const n_leaves = static_cast<std::int32_t>(d_leaves.size());
  for (auto ridx : dh::GridStrideRange<std::size_t>(0, d_inv_ridx.size())) {
    cuda_impl::RowIndexT idx = d_inv_ridx[ridx];
    bst_node_t leaf_nidx = FindLeafForPosition(idx, d_leaves.data(), n_leaves);
    bst_node_t encoded = op(static_cast<cuda_impl::RowIndexT>(ridx), leaf_nidx);
    d_out_position[ridx] = encoded;
  }
}

/** \brief Class responsible for tracking subsets of rows as we add splits and
 * partition training rows into different leaf nodes. */
class RowPartitioner {
 public:
  using RowIndexT = cuda_impl::RowIndexT;

 private:
  /**
   * In here if you want to find the rows belong to a node nid, first you need to get the
   * indices segment from ridx_segments[nid], then get the row index that represents
   * position of row in input data X.  `RowPartitioner::GetRows` would be a good starting
   * place to get a sense what are these vector storing.
   *
   * node id -> segment -> indices of rows belonging to node
   */

  /** @brief Range of row index for each node, pointers into ridx below. */
  std::vector<NodePositionInfo> ridx_segments_;
  /**
   * @brief mapping for node id -> rows.
   *
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   *
   * `ridx_` is the active buffer, `ridx_swap_` is the scratch destination for the
   * next partitioning pass. After each `UpdatePositionBatch` we swap them — the
   * scratch becomes active and the old active becomes the next scratch. This
   * eliminates a per-call copy kernel that previously copied the scattered
   * results back into `ridx_`.
   */
  dh::DeviceUVector<RowIndexT> ridx_;
  dh::DeviceUVector<RowIndexT> ridx_swap_;
  // Scratch inverse-permutation buffer used by FinalisePosition. Mutable so that
  // the const FinalisePosition member can resize it lazily; the contents are
  // recomputed every call from `ridx_`.
  mutable dh::DeviceUVector<RowIndexT> inv_ridx_;
  dh::DeviceUVector<int8_t> tmp_;
  dh::PinnedMemory pinned_;
  dh::PinnedMemory pinned2_;
  bst_node_t n_nodes_{0};  // Counter for internal checks.

 public:
  /**
   * @param ctx Context for device ordinal and stream.
   * @param n_samples The number of samples in each batch.
   * @param base_rowid The base row index for the current batch.
   */
  RowPartitioner() = default;
  void Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid);

  ~RowPartitioner();
  RowPartitioner(const RowPartitioner&) = delete;
  RowPartitioner& operator=(const RowPartitioner&) = delete;

  /**
   * \brief Gets the row indices of training instances in a given node.
   */
  common::Span<const RowIndexT> GetRows(bst_node_t nidx);

  /**
   * \brief Gets all training rows in the set.
   */
  common::Span<const RowIndexT> GetRows() const;
  /**
   * @brief Get the number of rows in this partitioner.
   */
  std::size_t Size() const { return this->GetRows().size(); }

  [[nodiscard]] bst_node_t GetNumNodes() const { return n_nodes_; }

  /**
   * @brief Convenience method for testing.
   */
  std::vector<RowIndexT> GetRowsHost(bst_node_t nidx);

  [[nodiscard]] std::vector<LeafInfo> GetLeaves() const {
    std::vector<LeafInfo> leaves;
    bst_node_t nidx = 0;
    for (auto const& node : this->ridx_segments_) {
      if (node.IsLeaf()) {
        leaves.emplace_back(LeafInfo{nidx, node});
      }
      nidx += 1;
    }
    return leaves;
  }

  /**
   * \brief Updates the tree position for set of training instances being split
   * into left and right child nodes. Accepts a user-defined lambda specifying
   * which branch each training instance should go down.
   *
   * \tparam  UpdatePositionOpT
   * \tparam  OpDataT
   * \param nidx        The index of the nodes being split.
   * \param left_nidx   The left child indices.
   * \param right_nidx  The right child indices.
   * \param op_data     User-defined data provided as the second argument to op
   * \param op          Device lambda with the row index as the first argument and op_data as the
   * second. Returns true if this training instance goes on the left partition.
   */
  template <typename UpdatePositionOpT, typename OpDataT>
  void UpdatePositionBatch(Context const* ctx, std::vector<bst_node_t> const& nidx,
                           std::vector<bst_node_t> const& left_nidx,
                           std::vector<bst_node_t> const& right_nidx,
                           std::vector<OpDataT> const& op_data,
                           UpdatePositionOpT op) {
    if (nidx.empty()) {
      return;
    }

    CHECK_EQ(nidx.size(), left_nidx.size());
    CHECK_EQ(nidx.size(), right_nidx.size());
    CHECK_EQ(nidx.size(), op_data.size());
    this->n_nodes_ += (left_nidx.size() + right_nidx.size());
    common::Span<PerNodeData<OpDataT>> h_batch_info =
        pinned2_.GetSpan<PerNodeData<OpDataT>>(nidx.size());
    dh::TemporaryArray<PerNodeData<OpDataT>> d_batch_info(nidx.size());

    bst_idx_t total_touched = 0;
    for (std::size_t i = 0; i < nidx.size(); i++) {
      auto seg = ridx_segments_.at(nidx[i]).segment;
      h_batch_info[i] = {seg, op_data[i]};
      total_touched += seg.Size();
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size_bytes(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));
    // Temporary arrays
    auto h_counts = pinned_.GetSpan<RowIndexT>(nidx.size());
    // Must initialize with 0 as 0 count is not written in the kernel.
    dh::TemporaryArray<RowIndexT> d_counts(nidx.size(), 0);
    CHECK_EQ(ridx_swap_.size(), this->ridx_.size());

    // If this call does not touch every row, the untouched rows in the swap buffer
    // would otherwise hold stale values from a previous pass. Seed the swap buffer
    // with the current ridx so that after the scatter+swap the inactive rows still
    // hold the correct row indices.
    if (total_touched != this->ridx_.size()) {
      dh::safe_cuda(cudaMemcpyAsync(this->ridx_swap_.data(), this->ridx_.data(),
                                    sizeof(RowIndexT) * this->ridx_.size(), cudaMemcpyDefault,
                                    ctx->CUDACtx()->Stream()));
    }

    // Process a sub-batch
    auto sub_batch_impl = [&](common::Span<bst_node_t const> nidx,
                              common::Span<PerNodeData<OpDataT>> d_batch_info,
                              common::Span<RowIndexT> d_counts) {
      std::size_t total_rows = 0;
      for (bst_node_t i : nidx) {
        total_rows += this->ridx_segments_[i].segment.Size();
      }

      // Partition the rows according to the operator: read from ridx_, scatter into ridx_swap_.
      SortPositionBatch<UpdatePositionOpT, OpDataT>(
          ctx, d_batch_info,
          common::Span<RowIndexT const>{this->ridx_.data(), this->ridx_.size()},
          dh::ToSpan(this->ridx_swap_), d_counts, total_rows, op, &this->tmp_);
    };

    // Divide inputs into sub-batches.
    for (std::size_t batch_begin = 0, n = nidx.size(); batch_begin < n;
         batch_begin += cuda_impl::kMaxUpdatePositionBatchSize) {
      auto constexpr kMax = static_cast<decltype(n)>(cuda_impl::kMaxUpdatePositionBatchSize);
      auto batch_size = std::min(kMax, n - batch_begin);
      auto nidx_batch = common::Span{nidx}.subspan(batch_begin, batch_size);
      auto d_info_batch = dh::ToSpan(d_batch_info).subspan(batch_begin, batch_size);
      auto d_counts_batch = dh::ToSpan(d_counts).subspan(batch_begin, batch_size);
      sub_batch_impl(nidx_batch, d_info_batch, d_counts_batch);
    }

    // Ping-pong: the freshly scattered swap buffer becomes the active one. The old
    // active buffer is now scratch for the next call. No copy kernel needed.
    using std::swap;
    swap(this->ridx_, this->ridx_swap_);

    dh::safe_cuda(cudaMemcpyAsync(h_counts.data(), d_counts.data().get(), h_counts.size_bytes(),
                                  cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
    // TODO(Rory): this synchronisation hurts performance a lot
    // Future optimisation should find a way to skip this
    ctx->CUDACtx()->Stream().Sync();

    // Update segments
    for (std::size_t i = 0; i < nidx.size(); i++) {
      auto segment = ridx_segments_.at(nidx[i]).segment;
      auto left_count = h_counts[i];
      CHECK_LE(left_count, segment.Size());
      ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                     std::max(left_nidx[i], right_nidx[i]) + 1));
      ridx_segments_[nidx[i]] = NodePositionInfo{segment, left_nidx[i], right_nidx[i]};
      ridx_segments_[left_nidx[i]] =
          NodePositionInfo{Segment{segment.begin, segment.begin + left_count}};
      ridx_segments_[right_nidx[i]] =
          NodePositionInfo{Segment{segment.begin + left_count, segment.end}};
    }
  }

  /**
   * @brief Finalise the position of all training instances after tree construction is
   * complete. Does not update any other meta information in this data structure, so
   * should only be used at the end of training.
   *
   * @param p_out_position Node index for each row.
   * @param op Device lambda. Should provide the row index and current position as an
   *           argument and return the new position for this training instance.
   */
  template <typename FinalisePositionOpT>
  void FinalisePosition(Context const* ctx, common::Span<bst_node_t> d_out_position,
                        bst_idx_t base_ridx, FinalisePositionOpT op) const {
    // 1) Build a flat sorted (segment.begin, leaf_nidx) array on the host. Leaf
    //    segments tile [0, ridx_.size()) disjointly so a sort by `seg_begin` is
    //    unique. The kernel uses binary search on this array to find each row's
    //    leaf, replacing the previous root-to-leaf tree walk.
    std::vector<LeafBoundary> h_leaves;
    h_leaves.reserve(ridx_segments_.size());
    for (bst_node_t i = 0; i < static_cast<bst_node_t>(ridx_segments_.size()); ++i) {
      auto const& info = ridx_segments_[i];
      if (info.IsLeaf()) {
        h_leaves.push_back(LeafBoundary{info.segment.begin, i});
      }
    }
    std::sort(h_leaves.begin(), h_leaves.end(),
              [](LeafBoundary const& a, LeafBoundary const& b) {
                return a.seg_begin < b.seg_begin;
              });

    dh::TemporaryArray<LeafBoundary> d_leaves(h_leaves.size());
    dh::safe_cuda(cudaMemcpyAsync(d_leaves.data().get(), h_leaves.data(),
                                  sizeof(LeafBoundary) * h_leaves.size(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));

    // 2) Build inv_ridx so that inv_ridx[ridx] == idx (where d_ridx[idx] == ridx + base_ridx).
    //    The build pass has one scattered write of 20 MB (5M × 4 B) — much cheaper than
    //    the 80 MB scattered read of d_gpair the main kernel saves below.
    inv_ridx_.resize(ridx_.size());

    constexpr std::uint32_t kBlockSize = 512;
    const int kItemsThread = 8;
    const std::uint32_t grid_size =
        xgboost::common::DivRoundUp(ridx_.size(), kBlockSize * kItemsThread);
    common::Span<RowIndexT const> d_ridx{ridx_.data(), ridx_.size()};
    dh::LaunchKernel{grid_size, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
        BuildInverseRidxKernel<kBlockSize>, d_ridx, base_ridx, dh::ToSpan(inv_ridx_));

    // 3) Reverse-iteration finalise pass: iterate by ridx (sequential reads of d_gpair
    //    inside `op` and sequential writes to d_out_position). The single random read is
    //    inv_ridx[ridx], which is itself a sequential stream now.
    common::Span<RowIndexT const> d_inv_ridx{inv_ridx_.data(), inv_ridx_.size()};
    dh::LaunchKernel{grid_size, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
        FinalisePositionKernel<kBlockSize, FinalisePositionOpT>, dh::ToSpan(d_leaves), d_inv_ridx,
        d_out_position, op);
  }
};

// Partitioner for all batches, used for external memory training.
class RowPartitionerBatches {
 private:
  // Partitioners for each batch. Each partitioner owns its own ping-pong scratch
  // buffer; no shared scratch is needed at this level.
  std::vector<std::unique_ptr<RowPartitioner>> partitioners_;

 public:
  void Reset(Context const* ctx, std::vector<bst_idx_t> const& batch_ptr) {
    CHECK_GE(batch_ptr.size(), 2);
    std::size_t n_batches = batch_ptr.size() - 1;
    if (partitioners_.size() != n_batches) {
      partitioners_.clear();
    }

    for (std::size_t k = 0; k < n_batches; ++k) {
      if (partitioners_.size() != n_batches) {
        // First run.
        partitioners_.emplace_back(std::make_unique<RowPartitioner>());
      }
      auto base_ridx = batch_ptr[k];
      auto n_samples = batch_ptr.at(k + 1) - base_ridx;
      partitioners_[k]->Reset(ctx, n_samples, base_ridx);
      CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
    }
  }

  // Accessors
  [[nodiscard]] decltype(auto) operator[](std::size_t i) { return partitioners_[i]; }
  decltype(auto) At(std::size_t i) { return partitioners_.at(i); }
  [[nodiscard]] std::size_t Size() const { return this->partitioners_.size(); }
  decltype(auto) cbegin() const { return this->partitioners_.cbegin(); }  // NOLINT
  decltype(auto) cend() const { return this->partitioners_.cend(); }      // NOLINT
  decltype(auto) begin() const { return this->partitioners_.cbegin(); }   // NOLINT
  decltype(auto) end() const { return this->partitioners_.cend(); }       // NOLINT

  [[nodiscard]] decltype(auto) Front() { return this->partitioners_.front(); }
  [[nodiscard]] bool Empty() const { return this->partitioners_.empty(); }

  template <typename UpdatePositionOpT, typename OpDataT>
  void UpdatePositionBatch(Context const* ctx, std::int32_t batch_idx,
                           std::vector<bst_node_t> const& nidx,
                           std::vector<bst_node_t> const& left_nidx,
                           std::vector<bst_node_t> const& right_nidx,
                           std::vector<OpDataT> const& op_data, UpdatePositionOpT op) {
    auto& part = this->At(batch_idx);
    part->UpdatePositionBatch(ctx, nidx, left_nidx, right_nidx, op_data, op);
  }
};
};  // namespace xgboost::tree
