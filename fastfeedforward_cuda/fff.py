import torch
import fff_cuda

@torch.compile
def fffn_backward(grad, in_weight, out_weight, input_values, intermediate_values):
    out_weight_grad = torch.sparse.mm(torch.relu(intermediate_values).pow(2).T, grad)
    grad_intermediate = torch.sparse.mm(out_weight.T, 2 * torch.relu(intermediate_values))
    grad_intermediate = torch.sparse.mm(grad, grad_intermediate)
    in_bias_grad = grad_intermediate.sum(dim=0)
    in_weight_grad = torch.sparse.mm(input_values.T, grad_intermediate)
    grad_input = torch.sparse.mm(grad_intermediate, in_weight.T)
    return grad_input, in_weight_grad, in_bias_grad, out_weight_grad


class FFFN_function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        x : torch.Tensor, 
        in_weight : torch.Tensor, 
        in_bias : torch.Tensor, 
        out_weight : torch.Tensor, 
        load_balancing_bias : torch.Tensor, 
        model_dim : int, 
        depth : int, 
        number_of_tree : int, 
        master_node_width : int, 
        n_nodes_per_tree : int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        init_shape = x.shape
        x= x.view(-1, model_dim)
        batch_size = x.shape[0]
        output = torch.zeros((batch_size, model_dim), dtype=x.dtype, device=x.device)
        activated_nodes = torch.zeros((batch_size, (depth + master_node_width) * number_of_tree), dtype=torch.int32, device=x.device, requires_grad=False)
        activated_nodes_values = torch.zeros((batch_size, (depth + master_node_width) * number_of_tree), dtype=x.dtype, device=x.device, requires_grad=False)
        fff_cuda.forward(
            x.view(-1, model_dim),
            in_weight,
            in_bias,
            out_weight,
            load_balancing_bias,
            output,
            activated_nodes,
            activated_nodes_values,
            int(model_dim),
            int(depth),
            int(number_of_tree),
            int(master_node_width),
            int(n_nodes_per_tree),
        )
        row_indices = torch.repeat_interleave(torch.arange(activated_nodes.shape[0]), 
                            torch.tensor([activated_nodes.shape[1]]))
        activated_nodes = activated_nodes.view(-1)
        col_indices = torch.stack((row_indices, activated_nodes), dim=0)
        intermediate_values = intermediate_values.view(-1)
        sparse_inter = torch.sparse_coo_tensor(col_indices, intermediate_values, (batch_size, n_nodes_per_tree * number_of_tree))
        ctx.save_for_backward(x, sparse_inter)
        return output.view(init_shape), activated_nodes

    @staticmethod
    def backward(ctx, grad_output, in_weight, out_weight, model_dim):
        init_shape = grad_output.shape
        grad_output = grad_output.view(-1, model_dim)
        x, sparse_intermediate = ctx.saved_tensors
        grad_output, in_weight_grad, in_bias_grad, out_weight_grad = fffn_backward(grad_output, in_weight, out_weight, x, sparse_intermediate)
        return grad_output.view(*init_shape), in_weight_grad, in_bias_grad, out_weight_grad
                
    # This is required to make Sphinx happy :-(
    @classmethod
    def apply(cls, *args, **kwargs):
        return super().apply(*args, **kwargs)


fffn_function = FFFN_function.apply