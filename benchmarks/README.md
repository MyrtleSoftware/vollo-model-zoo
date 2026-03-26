# Vollo Model Zoo Benchmarks (version 26.2.0)

Compute latency for an approximately 1-million parameter model.

## Configuration: V80

| Model | Latency/us | Latency/us (contiguous) | Metadata |
| ----- | ---------- | ----------------------- | -------- |
| ffn-swiglu | 1.84 | 1.85 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.57 | 4.82 | layers=4, channels=64, kernel_size=64 |

## Performance over time

![cnn_IA_420f](../plots/cnn_IA_420f.svg)

![cnn_IA_840f](../plots/cnn_IA_840f.svg)

![cnn_NT400D11](../plots/cnn_NT400D11.svg)

![cnn_V80](../plots/cnn_V80.svg)

![cnn_V80LL](../plots/cnn_V80LL.svg)

![ffn_swiglu_IA_420f](../plots/ffn_swiglu_IA_420f.svg)

![ffn_swiglu_IA_840f](../plots/ffn_swiglu_IA_840f.svg)

![ffn_swiglu_NT400D11](../plots/ffn_swiglu_NT400D11.svg)

![ffn_swiglu_V80](../plots/ffn_swiglu_V80.svg)

![ffn_swiglu_V80LL](../plots/ffn_swiglu_V80LL.svg)

![gru_IA_420f](../plots/gru_IA_420f.svg)

![gru_IA_840f](../plots/gru_IA_840f.svg)

![gru_NT400D11](../plots/gru_NT400D11.svg)

![gru_V80](../plots/gru_V80.svg)

![gru_V80LL](../plots/gru_V80LL.svg)

![lstm_IA_420f](../plots/lstm_IA_420f.svg)

![lstm_IA_840f](../plots/lstm_IA_840f.svg)

![lstm_NT400D11](../plots/lstm_NT400D11.svg)

![lstm_V80](../plots/lstm_V80.svg)

![lstm_V80LL](../plots/lstm_V80LL.svg)

![mamba1_IA_420f](../plots/mamba1_IA_420f.svg)

![mamba1_IA_840f](../plots/mamba1_IA_840f.svg)

![mamba1_NT400D11](../plots/mamba1_NT400D11.svg)

![mamba1_V80](../plots/mamba1_V80.svg)

![mamba1_V80LL](../plots/mamba1_V80LL.svg)

![mamba2_IA_420f](../plots/mamba2_IA_420f.svg)

![mamba2_IA_840f](../plots/mamba2_IA_840f.svg)

![mamba2_NT400D11](../plots/mamba2_NT400D11.svg)

![mamba2_V80](../plots/mamba2_V80.svg)

![mamba2_V80LL](../plots/mamba2_V80LL.svg)

![mlp_IA_420f](../plots/mlp_IA_420f.svg)

![mlp_IA_840f](../plots/mlp_IA_840f.svg)

![mlp_NT400D11](../plots/mlp_NT400D11.svg)

![mlp_V80](../plots/mlp_V80.svg)

![mlp_V80LL](../plots/mlp_V80LL.svg)

![mlp_res_rms_IA_420f](../plots/mlp_res_rms_IA_420f.svg)

![mlp_res_rms_IA_840f](../plots/mlp_res_rms_IA_840f.svg)

![mlp_res_rms_NT400D11](../plots/mlp_res_rms_NT400D11.svg)

![mlp_res_rms_V80](../plots/mlp_res_rms_V80.svg)

![mlp_res_rms_V80LL](../plots/mlp_res_rms_V80LL.svg)

![mobilenet_IA_420f](../plots/mobilenet_IA_420f.svg)

![mobilenet_IA_840f](../plots/mobilenet_IA_840f.svg)

![mobilenet_NT400D11](../plots/mobilenet_NT400D11.svg)

![mobilenet_V80](../plots/mobilenet_V80.svg)

![mobilenet_V80LL](../plots/mobilenet_V80LL.svg)

![moe_IA_420f](../plots/moe_IA_420f.svg)

![moe_IA_840f](../plots/moe_IA_840f.svg)

![moe_NT400D11](../plots/moe_NT400D11.svg)

![moe_V80](../plots/moe_V80.svg)

![moe_V80LL](../plots/moe_V80LL.svg)

![resmlp_IA_420f](../plots/resmlp_IA_420f.svg)

![resmlp_IA_840f](../plots/resmlp_IA_840f.svg)

![resmlp_NT400D11](../plots/resmlp_NT400D11.svg)

![resmlp_V80](../plots/resmlp_V80.svg)

![resmlp_V80LL](../plots/resmlp_V80LL.svg)

![slp_IA_420f](../plots/slp_IA_420f.svg)

![slp_IA_840f](../plots/slp_IA_840f.svg)

![slp_NT400D11](../plots/slp_NT400D11.svg)

![slp_V80](../plots/slp_V80.svg)

![slp_V80LL](../plots/slp_V80LL.svg)

![ssm_IA_420f](../plots/ssm_IA_420f.svg)

![ssm_IA_840f](../plots/ssm_IA_840f.svg)

![ssm_NT400D11](../plots/ssm_NT400D11.svg)

![ssm_V80](../plots/ssm_V80.svg)

![ssm_V80LL](../plots/ssm_V80LL.svg)

![tcn_IA_420f](../plots/tcn_IA_420f.svg)

![tcn_IA_840f](../plots/tcn_IA_840f.svg)

![tcn_NT400D11](../plots/tcn_NT400D11.svg)

![tcn_V80](../plots/tcn_V80.svg)

![tcn_V80LL](../plots/tcn_V80LL.svg)

![wavenet_IA_420f](../plots/wavenet_IA_420f.svg)

![wavenet_IA_840f](../plots/wavenet_IA_840f.svg)

![wavenet_NT400D11](../plots/wavenet_NT400D11.svg)

![wavenet_V80](../plots/wavenet_V80.svg)

![wavenet_V80LL](../plots/wavenet_V80LL.svg)
