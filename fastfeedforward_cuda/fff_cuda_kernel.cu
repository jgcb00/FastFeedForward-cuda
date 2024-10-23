#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu_squared(scalar_t z) {
  return z > 0 ? z * z : static_cast<scalar_t>(0);
}


// torch::PackedTensorAccessor is a helper class that provides a more
// convenient way to access the data in a torch::Tensor.
// scalar_t is the type of the data in the tensor.
// 2 is the number of dimensions in the tensor.
// torch::RestrictPtrTraits is a helper class that provides a more
// convenient way to access the data in a torch::Tensor.
// size_t is the type of the index used to access the data in the tensor.

template <typename scalar_t>
__global__ void fff_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> in_weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in_bias,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> out_weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> load_balancing_bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> activated_nodes,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> activated_nodes_values,
    const unsigned int width,
    const unsigned int depth,
    const unsigned int parallel_size,
    const unsigned int master_node_width,
    const unsigned int n_nodes
  ) {
  // compute which row of inputs we're dealing with
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;

  // zero the output
  for (int i = 0; i < width; ++i) {
    output[row_index][i] = 0;
  }

  if (row_index < x.size(0)) {
    int current_node = 0;
    int current_tree_offset = 0;
    int current_node_w_offset = 0;
    for (int current_tree = 0; current_tree < parallel_size; current_tree++) {
      current_tree_offset = current_tree * n_nodes;
      current_node = 0;
      for (int current_depth = 0; current_depth < depth; current_depth++) {
        scalar_t acc = 0;
        current_node_w_offset = current_node + current_tree_offset;
        activated_nodes[row_index][current_tree*depth + current_depth] = current_node_w_offset;
        for (int i = 0; i < width;++i) {
            acc += x[row_index][i] * in_weight[current_node_w_offset][i];
        }
        acc += in_bias[current_node_w_offset];
        activated_nodes_values[row_index][current_tree*depth + current_depth] = acc;

        // compute the activation
        scalar_t activation = relu_squared(acc);
        // compute the output contribution due to the current node
        for (int i = 0; i < width; ++i) {
            output[row_index][i] += activation * out_weight[current_node_w_offset][i];
        }

        // decide where to move to
        current_node = (current_node<<1) + 1 + (acc > load_balancing_bias[current_node_w_offset] ? 1 : 0);
      }
      // Apply master node leaf weights
      for (int i = 1; i <= master_node_width; i++) {
        scalar_t acc = 0;
        current_node_w_offset = current_tree_offset + n_nodes - i;
        activated_nodes[row_index][current_tree*depth + depth + i - 1] = current_node_w_offset;
        for (int i = 0; i < width;++i) {
            acc += x[row_index][i] * in_weight[current_node_w_offset][i];
        }
        acc += in_bias[current_node_w_offset];
        activated_nodes_values[row_index][current_tree*depth + depth + i - 1] = acc;
        acc = relu_squared(acc);
        for (int i = 0; i < width; ++i) {
            output[row_index][i] += acc * out_weight[current_node_w_offset][i];
        }
      }

    }
  }
}
} // namespace

void fff_cuda_forward(
	torch::Tensor x,
	torch::Tensor in_weight,
	torch::Tensor in_bias,
	torch::Tensor out_weight,
  torch::Tensor load_balancing_bias,
  torch::Tensor output,
  torch::Tensor activated_nodes,
  torch::Tensor activated_nodes_values,
	const unsigned int width,
	const unsigned int depth,
	const unsigned int parallel_size,
  const unsigned int master_node_width,
	const unsigned int n_nodes
) {

  const int batch_size = x.size(0);

  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(torch::kBFloat16, in_weight.scalar_type(), "fff_forward_cuda", ([&] {
    fff_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        out_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        load_balancing_bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        activated_nodes.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        activated_nodes_values.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        width,
        depth,
        parallel_size,
        master_node_width,
        n_nodes
    );
  }));

  cudaError_t err;
  err = cudaGetLastError();
  if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  cudaError_t cudaStatus;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }

}