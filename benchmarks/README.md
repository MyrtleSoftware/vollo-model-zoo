# Vollo Model Zoo Benchmarks (version 26.2.0)

Compute latency for an approximately 1-million parameter model.

## Configuration: V80LL

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 3.12 | 3.23 | layers=4, blocks=1, hidden=198 |
| tcn | 1.11 | 1.50 | inputs=1, kernel=3, channels=[256, 256, 256] |
| ssm | 0.80 | 0.88 | dim=576, hidden=448 |
| slp | 0.86 | 0.86 | input=1024, output=1024, activation=ReLU |
| resmlp | 6.84 | 6.85 | dim=160, patches=9, layers=5, activation=ReLU |
| moe | 0.85 | 0.85 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 9.22 | 9.30 | width_mult=0.42 |
| mlp-res-rms | 1.61 | 1.61 | dim=320, hidden=768, activation=relu |
| mlp | 1.30 | 1.30 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 2.75 | 4.65 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 2.82 | 3.21 | dim=384, state=12, layers=1 |
| lstm | 0.75 | 0.97 | layers=2, hidden_size=250 |
| gru | 0.85 | 1.18 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.68 | 1.68 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.43 | 4.38 | layers=4, channels=64, kernel_size=64 |

## Configuration: V80

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 3.44 | 3.55 | layers=4, blocks=1, hidden=198 |
| tcn | 1.22 | 1.65 | inputs=1, kernel=3, channels=[256, 256, 256] |
| ssm | 0.88 | 0.97 | dim=576, hidden=448 |
| slp | 0.94 | 0.95 | input=1024, output=1024, activation=ReLU |
| resmlp | 7.53 | 7.53 | dim=160, patches=9, layers=5, activation=ReLU |
| moe | 0.94 | 0.94 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 10.15 | 10.23 | width_mult=0.42 |
| mlp-res-rms | 1.77 | 1.77 | dim=320, hidden=768, activation=relu |
| mlp | 1.43 | 1.43 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 3.03 | 5.12 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 3.10 | 3.53 | dim=384, state=12, layers=1 |
| lstm | 0.82 | 1.07 | layers=2, hidden_size=250 |
| gru | 0.94 | 1.30 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.84 | 1.85 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.57 | 4.82 | layers=4, channels=64, kernel_size=64 |

## Configuration: NT400D11

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 3.68 | 3.79 | layers=4, blocks=1, hidden=198 |
| tcn | 1.27 | 1.68 | inputs=1, kernel=3, channels=[256, 256, 256] |
| ssm | 0.86 | 0.96 | dim=576, hidden=448 |
| slp | 0.90 | 0.90 | input=1024, output=1024, activation=ReLU |
| resmlp | 15.50 | 15.50 | dim=160, patches=9, layers=5, activation=ReLU |
| moe | 1.14 | 1.15 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 10.86 | 10.86 | width_mult=0.42 |
| mlp-res-rms | 1.80 | 1.80 | dim=320, hidden=768, activation=relu |
| mlp | 1.45 | 1.45 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 3.02 | 5.12 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 3.10 | 3.51 | dim=384, state=12, layers=1 |
| lstm | 0.80 | 1.02 | layers=2, hidden_size=250 |
| gru | 0.91 | 1.27 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.84 | 1.84 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.63 | 3.06 | layers=4, channels=64, kernel_size=64 |

## Configuration: IA-840f

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 3.31 | 3.33 | layers=4, blocks=1, hidden=198 |
| tcn | 1.04 | 1.46 | inputs=1, kernel=3, channels=[256, 256, 256] |
| ssm | 0.63 | 0.68 | dim=576, hidden=448 |
| slp | 0.56 | 0.56 | input=1024, output=1024, activation=ReLU |
| resmlp | 4.99 | 4.99 | dim=160, patches=9, layers=5, activation=ReLU |
| moe | 0.69 | 0.69 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 10.12 | 10.23 | width_mult=0.42 |
| mlp-res-rms | 1.51 | 1.51 | dim=320, hidden=768, activation=relu |
| mlp | 1.21 | 1.22 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 3.63 | 6.83 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 2.35 | 2.74 | dim=384, state=12, layers=1 |
| lstm | 0.58 | 0.60 | layers=2, hidden_size=250 |
| gru | 0.80 | 1.08 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.42 | 1.43 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.61 | 3.07 | layers=4, channels=64, kernel_size=64 |

## Configuration: IA-420f

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 3.68 | 3.79 | layers=4, blocks=1, hidden=198 |
| tcn | 1.27 | 1.68 | inputs=1, kernel=3, channels=[256, 256, 256] |
| ssm | 0.86 | 0.96 | dim=576, hidden=448 |
| slp | 0.90 | 0.90 | input=1024, output=1024, activation=ReLU |
| resmlp | 15.50 | 15.50 | dim=160, patches=9, layers=5, activation=ReLU |
| moe | 1.14 | 1.15 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 10.86 | 10.86 | width_mult=0.42 |
| mlp-res-rms | 1.80 | 1.80 | dim=320, hidden=768, activation=relu |
| mlp | 1.45 | 1.45 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 3.02 | 5.12 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 3.10 | 3.51 | dim=384, state=12, layers=1 |
| lstm | 0.80 | 1.02 | layers=2, hidden_size=250 |
| gru | 0.91 | 1.27 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.84 | 1.84 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.63 | 3.06 | layers=4, channels=64, kernel_size=64 |

## Performance over time

Click to expand each plot:

<details>
<summary>cnn_IA_420f</summary>

![cnn_IA_420f](../plots/cnn_IA_420f.svg)

</details>

<details>
<summary>cnn_IA_840f</summary>

![cnn_IA_840f](../plots/cnn_IA_840f.svg)

</details>

<details>
<summary>cnn_NT400D11</summary>

![cnn_NT400D11](../plots/cnn_NT400D11.svg)

</details>

<details>
<summary>cnn_V80</summary>

![cnn_V80](../plots/cnn_V80.svg)

</details>

<details>
<summary>cnn_V80LL</summary>

![cnn_V80LL](../plots/cnn_V80LL.svg)

</details>

<details>
<summary>ffn_swiglu_IA_420f</summary>

![ffn_swiglu_IA_420f](../plots/ffn_swiglu_IA_420f.svg)

</details>

<details>
<summary>ffn_swiglu_IA_840f</summary>

![ffn_swiglu_IA_840f](../plots/ffn_swiglu_IA_840f.svg)

</details>

<details>
<summary>ffn_swiglu_NT400D11</summary>

![ffn_swiglu_NT400D11](../plots/ffn_swiglu_NT400D11.svg)

</details>

<details>
<summary>ffn_swiglu_V80</summary>

![ffn_swiglu_V80](../plots/ffn_swiglu_V80.svg)

</details>

<details>
<summary>ffn_swiglu_V80LL</summary>

![ffn_swiglu_V80LL](../plots/ffn_swiglu_V80LL.svg)

</details>

<details>
<summary>gru_IA_420f</summary>

![gru_IA_420f](../plots/gru_IA_420f.svg)

</details>

<details>
<summary>gru_IA_840f</summary>

![gru_IA_840f](../plots/gru_IA_840f.svg)

</details>

<details>
<summary>gru_NT400D11</summary>

![gru_NT400D11](../plots/gru_NT400D11.svg)

</details>

<details>
<summary>gru_V80</summary>

![gru_V80](../plots/gru_V80.svg)

</details>

<details>
<summary>gru_V80LL</summary>

![gru_V80LL](../plots/gru_V80LL.svg)

</details>

<details>
<summary>lstm_IA_420f</summary>

![lstm_IA_420f](../plots/lstm_IA_420f.svg)

</details>

<details>
<summary>lstm_IA_840f</summary>

![lstm_IA_840f](../plots/lstm_IA_840f.svg)

</details>

<details>
<summary>lstm_NT400D11</summary>

![lstm_NT400D11](../plots/lstm_NT400D11.svg)

</details>

<details>
<summary>lstm_V80</summary>

![lstm_V80](../plots/lstm_V80.svg)

</details>

<details>
<summary>lstm_V80LL</summary>

![lstm_V80LL](../plots/lstm_V80LL.svg)

</details>

<details>
<summary>mamba1_IA_420f</summary>

![mamba1_IA_420f](../plots/mamba1_IA_420f.svg)

</details>

<details>
<summary>mamba1_IA_840f</summary>

![mamba1_IA_840f](../plots/mamba1_IA_840f.svg)

</details>

<details>
<summary>mamba1_NT400D11</summary>

![mamba1_NT400D11](../plots/mamba1_NT400D11.svg)

</details>

<details>
<summary>mamba1_V80</summary>

![mamba1_V80](../plots/mamba1_V80.svg)

</details>

<details>
<summary>mamba1_V80LL</summary>

![mamba1_V80LL](../plots/mamba1_V80LL.svg)

</details>

<details>
<summary>mamba2_IA_420f</summary>

![mamba2_IA_420f](../plots/mamba2_IA_420f.svg)

</details>

<details>
<summary>mamba2_IA_840f</summary>

![mamba2_IA_840f](../plots/mamba2_IA_840f.svg)

</details>

<details>
<summary>mamba2_NT400D11</summary>

![mamba2_NT400D11](../plots/mamba2_NT400D11.svg)

</details>

<details>
<summary>mamba2_V80</summary>

![mamba2_V80](../plots/mamba2_V80.svg)

</details>

<details>
<summary>mamba2_V80LL</summary>

![mamba2_V80LL](../plots/mamba2_V80LL.svg)

</details>

<details>
<summary>mlp_IA_420f</summary>

![mlp_IA_420f](../plots/mlp_IA_420f.svg)

</details>

<details>
<summary>mlp_IA_840f</summary>

![mlp_IA_840f](../plots/mlp_IA_840f.svg)

</details>

<details>
<summary>mlp_NT400D11</summary>

![mlp_NT400D11](../plots/mlp_NT400D11.svg)

</details>

<details>
<summary>mlp_res_rms_IA_420f</summary>

![mlp_res_rms_IA_420f](../plots/mlp_res_rms_IA_420f.svg)

</details>

<details>
<summary>mlp_res_rms_IA_840f</summary>

![mlp_res_rms_IA_840f](../plots/mlp_res_rms_IA_840f.svg)

</details>

<details>
<summary>mlp_res_rms_NT400D11</summary>

![mlp_res_rms_NT400D11](../plots/mlp_res_rms_NT400D11.svg)

</details>

<details>
<summary>mlp_res_rms_V80</summary>

![mlp_res_rms_V80](../plots/mlp_res_rms_V80.svg)

</details>

<details>
<summary>mlp_res_rms_V80LL</summary>

![mlp_res_rms_V80LL](../plots/mlp_res_rms_V80LL.svg)

</details>

<details>
<summary>mlp_V80</summary>

![mlp_V80](../plots/mlp_V80.svg)

</details>

<details>
<summary>mlp_V80LL</summary>

![mlp_V80LL](../plots/mlp_V80LL.svg)

</details>

<details>
<summary>mobilenet_IA_420f</summary>

![mobilenet_IA_420f](../plots/mobilenet_IA_420f.svg)

</details>

<details>
<summary>mobilenet_IA_840f</summary>

![mobilenet_IA_840f](../plots/mobilenet_IA_840f.svg)

</details>

<details>
<summary>mobilenet_NT400D11</summary>

![mobilenet_NT400D11](../plots/mobilenet_NT400D11.svg)

</details>

<details>
<summary>mobilenet_V80</summary>

![mobilenet_V80](../plots/mobilenet_V80.svg)

</details>

<details>
<summary>mobilenet_V80LL</summary>

![mobilenet_V80LL](../plots/mobilenet_V80LL.svg)

</details>

<details>
<summary>moe_IA_420f</summary>

![moe_IA_420f](../plots/moe_IA_420f.svg)

</details>

<details>
<summary>moe_IA_840f</summary>

![moe_IA_840f](../plots/moe_IA_840f.svg)

</details>

<details>
<summary>moe_NT400D11</summary>

![moe_NT400D11](../plots/moe_NT400D11.svg)

</details>

<details>
<summary>moe_V80</summary>

![moe_V80](../plots/moe_V80.svg)

</details>

<details>
<summary>moe_V80LL</summary>

![moe_V80LL](../plots/moe_V80LL.svg)

</details>

<details>
<summary>resmlp_IA_420f</summary>

![resmlp_IA_420f](../plots/resmlp_IA_420f.svg)

</details>

<details>
<summary>resmlp_IA_840f</summary>

![resmlp_IA_840f](../plots/resmlp_IA_840f.svg)

</details>

<details>
<summary>resmlp_NT400D11</summary>

![resmlp_NT400D11](../plots/resmlp_NT400D11.svg)

</details>

<details>
<summary>resmlp_V80</summary>

![resmlp_V80](../plots/resmlp_V80.svg)

</details>

<details>
<summary>resmlp_V80LL</summary>

![resmlp_V80LL](../plots/resmlp_V80LL.svg)

</details>

<details>
<summary>slp_IA_420f</summary>

![slp_IA_420f](../plots/slp_IA_420f.svg)

</details>

<details>
<summary>slp_IA_840f</summary>

![slp_IA_840f](../plots/slp_IA_840f.svg)

</details>

<details>
<summary>slp_NT400D11</summary>

![slp_NT400D11](../plots/slp_NT400D11.svg)

</details>

<details>
<summary>slp_V80</summary>

![slp_V80](../plots/slp_V80.svg)

</details>

<details>
<summary>slp_V80LL</summary>

![slp_V80LL](../plots/slp_V80LL.svg)

</details>

<details>
<summary>ssm_IA_420f</summary>

![ssm_IA_420f](../plots/ssm_IA_420f.svg)

</details>

<details>
<summary>ssm_IA_840f</summary>

![ssm_IA_840f](../plots/ssm_IA_840f.svg)

</details>

<details>
<summary>ssm_NT400D11</summary>

![ssm_NT400D11](../plots/ssm_NT400D11.svg)

</details>

<details>
<summary>ssm_V80</summary>

![ssm_V80](../plots/ssm_V80.svg)

</details>

<details>
<summary>ssm_V80LL</summary>

![ssm_V80LL](../plots/ssm_V80LL.svg)

</details>

<details>
<summary>tcn_IA_420f</summary>

![tcn_IA_420f](../plots/tcn_IA_420f.svg)

</details>

<details>
<summary>tcn_IA_840f</summary>

![tcn_IA_840f](../plots/tcn_IA_840f.svg)

</details>

<details>
<summary>tcn_NT400D11</summary>

![tcn_NT400D11](../plots/tcn_NT400D11.svg)

</details>

<details>
<summary>tcn_V80</summary>

![tcn_V80](../plots/tcn_V80.svg)

</details>

<details>
<summary>tcn_V80LL</summary>

![tcn_V80LL](../plots/tcn_V80LL.svg)

</details>

<details>
<summary>wavenet_IA_420f</summary>

![wavenet_IA_420f](../plots/wavenet_IA_420f.svg)

</details>

<details>
<summary>wavenet_IA_840f</summary>

![wavenet_IA_840f](../plots/wavenet_IA_840f.svg)

</details>

<details>
<summary>wavenet_NT400D11</summary>

![wavenet_NT400D11](../plots/wavenet_NT400D11.svg)

</details>

<details>
<summary>wavenet_V80</summary>

![wavenet_V80](../plots/wavenet_V80.svg)

</details>

<details>
<summary>wavenet_V80LL</summary>

![wavenet_V80LL](../plots/wavenet_V80LL.svg)

</details>
