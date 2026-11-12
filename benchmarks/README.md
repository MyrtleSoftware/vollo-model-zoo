# Vollo Model Zoo Benchmarks (Vollo SDK 30.1.0, model zoo 0.6.0)

Compute latency for an approximately 1-million parameter model.

Note: These latencies are from a (near cycle-accurate) software model but without IO (non-negligible for some models)

## Configuration: V80LL

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 2.42 | 2.57 | layers=4, blocks=1, hidden=198 |
| tcn | 1.05 | 1.45 | inputs=1, kernel=3, channels=[256, 256, 256] |
| swa | 3.33 | 3.76 | masked=True, partitions=6, dim=224, window=32, layers=2 |
| stateless-rnnt | 1.04 | 1.04 | prog=predictor, hidden=192, joint_hidden=288 |
| ssm | 0.72 | 0.72 | dim=576, hidden=448 |
| slp | 0.88 | 0.89 | input=1024, output=1024, activation=ReLU |
| rnnt | 0.68 | 0.68 | prog=predictor, hidden=192, joint_hidden=288 |
| resmlp | 6.63 | 6.63 | dim=160, patches=9, layers=5, activation=ReLU |
| qec-lstm | 0.95 | 1.03 | distance=11, hidden_size=256, soft=False |
| nano-llm | 2.78 | 4.80 | d_model=256, d_head=32, expand=1.5, n_layers=1, head_partitions=6, ffn_fp8=False |
| moe | 0.88 | 0.88 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 8.27 | 8.27 | width_mult=0.42 |
| mlp-res-rms | 1.53 | 1.53 | dim=320, hidden=768, activation=relu |
| mlp | 1.27 | 1.27 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 1.56 | 3.44 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 2.83 | 3.30 | dim=384, state=12, layers=1 |
| lstm | 0.77 | 1.00 | layers=2, hidden_size=250 |
| gru | 0.77 | 1.03 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.25 | 1.26 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.25 | 4.23 | layers=4, channels=64, kernel_size=64 |

## Configuration: V80

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 2.66 | 2.83 | layers=4, blocks=1, hidden=198 |
| tcn | 1.16 | 1.60 | inputs=1, kernel=3, channels=[256, 256, 256] |
| swa | 3.66 | 4.13 | masked=True, partitions=6, dim=224, window=32, layers=2 |
| stateless-rnnt | 1.14 | 1.14 | prog=predictor, hidden=192, joint_hidden=288 |
| ssm | 0.79 | 0.79 | dim=576, hidden=448 |
| slp | 0.97 | 0.98 | input=1024, output=1024, activation=ReLU |
| rnnt | 0.75 | 0.75 | prog=predictor, hidden=192, joint_hidden=288 |
| resmlp | 7.29 | 7.30 | dim=160, patches=9, layers=5, activation=ReLU |
| qec-lstm | 1.04 | 1.14 | distance=11, hidden_size=256, soft=False |
| nano-llm | 3.06 | 5.28 | d_model=256, d_head=32, expand=1.5, n_layers=1, head_partitions=6, ffn_fp8=False |
| moe | 0.97 | 0.97 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 9.09 | 9.10 | width_mult=0.42 |
| mlp-res-rms | 1.68 | 1.68 | dim=320, hidden=768, activation=relu |
| mlp | 1.39 | 1.40 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 1.72 | 3.79 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 3.12 | 3.63 | dim=384, state=12, layers=1 |
| lstm | 0.85 | 1.10 | layers=2, hidden_size=250 |
| gru | 0.85 | 1.14 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.38 | 1.38 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.38 | 4.66 | layers=4, channels=64, kernel_size=64 |

## Configuration: NT400D11

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 2.93 | 3.07 | layers=4, blocks=1, hidden=198 |
| tcn | 1.22 | 1.65 | inputs=1, kernel=3, channels=[256, 256, 256] |
| swa | 4.08 | 4.44 | masked=True, partitions=6, dim=224, window=32, layers=2 |
| stateless-rnnt | 1.17 | 1.18 | prog=predictor, hidden=192, joint_hidden=288 |
| ssm | 0.75 | 0.76 | dim=576, hidden=448 |
| slp | 0.94 | 0.94 | input=1024, output=1024, activation=ReLU |
| rnnt | 0.75 | 0.76 | prog=predictor, hidden=192, joint_hidden=288 |
| resmlp | 15.93 | 15.93 | dim=160, patches=9, layers=5, activation=ReLU |
| qec-lstm | 1.11 | 1.20 | distance=11, hidden_size=256, soft=False |
| nano-llm | 3.32 | 5.40 | d_model=256, d_head=32, expand=1.5, n_layers=1, head_partitions=6, ffn_fp8=False |
| moe | 1.17 | 1.18 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 10.25 | 10.30 | width_mult=0.42 |
| mlp-res-rms | 1.79 | 1.79 | dim=320, hidden=768, activation=relu |
| mlp | 1.48 | 1.48 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 2.05 | 3.77 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 3.12 | 3.54 | dim=384, state=12, layers=1 |
| lstm | 0.85 | 1.09 | layers=2, hidden_size=250 |
| gru | 0.87 | 1.16 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.43 | 1.43 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.52 | 3.00 | layers=4, channels=64, kernel_size=64 |

## Configuration: IA-840f

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 2.48 | 2.54 | layers=4, blocks=1, hidden=198 |
| tcn | 1.02 | 1.62 | inputs=1, kernel=3, channels=[256, 256, 256] |
| swa | 3.99 | 4.73 | masked=True, partitions=3, dim=224, window=32, layers=2 |
| stateless-rnnt | 0.96 | 0.96 | prog=predictor, hidden=192, joint_hidden=288 |
| ssm | 0.57 | 0.57 | dim=576, hidden=448 |
| slp | 0.63 | 0.63 | input=1024, output=1024, activation=ReLU |
| rnnt | 0.69 | 0.69 | prog=predictor, hidden=192, joint_hidden=288 |
| resmlp | 5.07 | 5.08 | dim=160, patches=9, layers=5, activation=ReLU |
| qec-lstm | 0.93 | 0.93 | distance=11, hidden_size=256, soft=False |
| nano-llm | 3.14 | 5.27 | d_model=256, d_head=32, expand=1.5, n_layers=1, head_partitions=3, ffn_fp8=False |
| moe | 0.69 | 0.70 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 9.68 | 9.77 | width_mult=0.42 |
| mlp-res-rms | 1.54 | 1.54 | dim=320, hidden=768, activation=relu |
| mlp | 1.35 | 1.35 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 1.59 | 3.70 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 2.36 | 2.79 | dim=384, state=12, layers=1 |
| lstm | 0.62 | 0.69 | layers=2, hidden_size=250 |
| gru | 0.66 | 0.87 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.18 | 1.18 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.53 | 3.02 | layers=4, channels=64, kernel_size=64 |

## Configuration: IA-420f

| Model | Latency (us) | Latency contiguous (us)  | Metadata |
| ----- | ------------ | ------------------------ | -------- |
| wavenet | 2.93 | 3.07 | layers=4, blocks=1, hidden=198 |
| tcn | 1.22 | 1.65 | inputs=1, kernel=3, channels=[256, 256, 256] |
| swa | 4.08 | 4.44 | masked=True, partitions=6, dim=224, window=32, layers=2 |
| stateless-rnnt | 1.17 | 1.18 | prog=predictor, hidden=192, joint_hidden=288 |
| ssm | 0.75 | 0.76 | dim=576, hidden=448 |
| slp | 0.94 | 0.94 | input=1024, output=1024, activation=ReLU |
| rnnt | 0.75 | 0.76 | prog=predictor, hidden=192, joint_hidden=288 |
| resmlp | 15.93 | 15.93 | dim=160, patches=9, layers=5, activation=ReLU |
| qec-lstm | 1.11 | 1.20 | distance=11, hidden_size=256, soft=False |
| nano-llm | 3.32 | 5.40 | d_model=256, d_head=32, expand=1.5, n_layers=1, head_partitions=6, ffn_fp8=False |
| moe | 1.17 | 1.18 | dim=192, hidden=640, n_experts=4 |
| mobilenet | 10.25 | 10.30 | width_mult=0.42 |
| mlp-res-rms | 1.79 | 1.79 | dim=320, hidden=768, activation=relu |
| mlp | 1.48 | 1.48 | layers=7, n_features=384, activation=ReLU |
| mamba2 | 2.05 | 3.77 | fp32=False, dim=400, state=32, layers=1 |
| mamba1 | 3.12 | 3.54 | dim=384, state=12, layers=1 |
| lstm | 0.85 | 1.09 | layers=2, hidden_size=250 |
| gru | 0.87 | 1.16 | fp32=False, input=512, hidden=384, layers=1 |
| ffn-swiglu | 1.43 | 1.43 | dim=288, hidden=1152, activation=SwiGLU, fused=True |
| cnn | 1.52 | 3.00 | layers=4, channels=64, kernel_size=64 |

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
<summary>cnn_V80LL</summary>

![cnn_V80LL](../plots/cnn_V80LL.svg)

</details>

<details>
<summary>cnn_V80</summary>

![cnn_V80](../plots/cnn_V80.svg)

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
<summary>ffn_swiglu_V80LL</summary>

![ffn_swiglu_V80LL](../plots/ffn_swiglu_V80LL.svg)

</details>

<details>
<summary>ffn_swiglu_V80</summary>

![ffn_swiglu_V80](../plots/ffn_swiglu_V80.svg)

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
<summary>gru_V80LL</summary>

![gru_V80LL](../plots/gru_V80LL.svg)

</details>

<details>
<summary>gru_V80</summary>

![gru_V80](../plots/gru_V80.svg)

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
<summary>lstm_V80LL</summary>

![lstm_V80LL](../plots/lstm_V80LL.svg)

</details>

<details>
<summary>lstm_V80</summary>

![lstm_V80](../plots/lstm_V80.svg)

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
<summary>mamba1_V80LL</summary>

![mamba1_V80LL](../plots/mamba1_V80LL.svg)

</details>

<details>
<summary>mamba1_V80</summary>

![mamba1_V80](../plots/mamba1_V80.svg)

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
<summary>mamba2_V80LL</summary>

![mamba2_V80LL](../plots/mamba2_V80LL.svg)

</details>

<details>
<summary>mamba2_V80</summary>

![mamba2_V80](../plots/mamba2_V80.svg)

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
<summary>mlp_res_rms_V80LL</summary>

![mlp_res_rms_V80LL](../plots/mlp_res_rms_V80LL.svg)

</details>

<details>
<summary>mlp_res_rms_V80</summary>

![mlp_res_rms_V80](../plots/mlp_res_rms_V80.svg)

</details>

<details>
<summary>mlp_V80LL</summary>

![mlp_V80LL](../plots/mlp_V80LL.svg)

</details>

<details>
<summary>mlp_V80</summary>

![mlp_V80](../plots/mlp_V80.svg)

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
<summary>mobilenet_V80LL</summary>

![mobilenet_V80LL](../plots/mobilenet_V80LL.svg)

</details>

<details>
<summary>mobilenet_V80</summary>

![mobilenet_V80](../plots/mobilenet_V80.svg)

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
<summary>moe_V80LL</summary>

![moe_V80LL](../plots/moe_V80LL.svg)

</details>

<details>
<summary>moe_V80</summary>

![moe_V80](../plots/moe_V80.svg)

</details>

<details>
<summary>nano_llm_IA_420f</summary>

![nano_llm_IA_420f](../plots/nano_llm_IA_420f.svg)

</details>

<details>
<summary>nano_llm_IA_840f</summary>

![nano_llm_IA_840f](../plots/nano_llm_IA_840f.svg)

</details>

<details>
<summary>nano_llm_NT400D11</summary>

![nano_llm_NT400D11](../plots/nano_llm_NT400D11.svg)

</details>

<details>
<summary>nano_llm_V80LL</summary>

![nano_llm_V80LL](../plots/nano_llm_V80LL.svg)

</details>

<details>
<summary>nano_llm_V80</summary>

![nano_llm_V80](../plots/nano_llm_V80.svg)

</details>

<details>
<summary>qec_lstm_IA_420f</summary>

![qec_lstm_IA_420f](../plots/qec_lstm_IA_420f.svg)

</details>

<details>
<summary>qec_lstm_IA_840f</summary>

![qec_lstm_IA_840f](../plots/qec_lstm_IA_840f.svg)

</details>

<details>
<summary>qec_lstm_NT400D11</summary>

![qec_lstm_NT400D11](../plots/qec_lstm_NT400D11.svg)

</details>

<details>
<summary>qec_lstm_V80LL</summary>

![qec_lstm_V80LL](../plots/qec_lstm_V80LL.svg)

</details>

<details>
<summary>qec_lstm_V80</summary>

![qec_lstm_V80](../plots/qec_lstm_V80.svg)

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
<summary>resmlp_V80LL</summary>

![resmlp_V80LL](../plots/resmlp_V80LL.svg)

</details>

<details>
<summary>resmlp_V80</summary>

![resmlp_V80](../plots/resmlp_V80.svg)

</details>

<details>
<summary>rnnt_IA_420f</summary>

![rnnt_IA_420f](../plots/rnnt_IA_420f.svg)

</details>

<details>
<summary>rnnt_IA_840f</summary>

![rnnt_IA_840f](../plots/rnnt_IA_840f.svg)

</details>

<details>
<summary>rnnt_NT400D11</summary>

![rnnt_NT400D11](../plots/rnnt_NT400D11.svg)

</details>

<details>
<summary>rnnt_V80LL</summary>

![rnnt_V80LL](../plots/rnnt_V80LL.svg)

</details>

<details>
<summary>rnnt_V80</summary>

![rnnt_V80](../plots/rnnt_V80.svg)

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
<summary>slp_V80LL</summary>

![slp_V80LL](../plots/slp_V80LL.svg)

</details>

<details>
<summary>slp_V80</summary>

![slp_V80](../plots/slp_V80.svg)

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
<summary>ssm_V80LL</summary>

![ssm_V80LL](../plots/ssm_V80LL.svg)

</details>

<details>
<summary>ssm_V80</summary>

![ssm_V80](../plots/ssm_V80.svg)

</details>

<details>
<summary>stateless_rnnt_IA_420f</summary>

![stateless_rnnt_IA_420f](../plots/stateless_rnnt_IA_420f.svg)

</details>

<details>
<summary>stateless_rnnt_IA_840f</summary>

![stateless_rnnt_IA_840f](../plots/stateless_rnnt_IA_840f.svg)

</details>

<details>
<summary>stateless_rnnt_NT400D11</summary>

![stateless_rnnt_NT400D11](../plots/stateless_rnnt_NT400D11.svg)

</details>

<details>
<summary>stateless_rnnt_V80LL</summary>

![stateless_rnnt_V80LL](../plots/stateless_rnnt_V80LL.svg)

</details>

<details>
<summary>stateless_rnnt_V80</summary>

![stateless_rnnt_V80](../plots/stateless_rnnt_V80.svg)

</details>

<details>
<summary>swa_IA_420f</summary>

![swa_IA_420f](../plots/swa_IA_420f.svg)

</details>

<details>
<summary>swa_IA_840f</summary>

![swa_IA_840f](../plots/swa_IA_840f.svg)

</details>

<details>
<summary>swa_NT400D11</summary>

![swa_NT400D11](../plots/swa_NT400D11.svg)

</details>

<details>
<summary>swa_V80LL</summary>

![swa_V80LL](../plots/swa_V80LL.svg)

</details>

<details>
<summary>swa_V80</summary>

![swa_V80](../plots/swa_V80.svg)

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
<summary>tcn_V80LL</summary>

![tcn_V80LL](../plots/tcn_V80LL.svg)

</details>

<details>
<summary>tcn_V80</summary>

![tcn_V80](../plots/tcn_V80.svg)

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
<summary>wavenet_V80LL</summary>

![wavenet_V80LL](../plots/wavenet_V80LL.svg)

</details>

<details>
<summary>wavenet_V80</summary>

![wavenet_V80](../plots/wavenet_V80.svg)

</details>
