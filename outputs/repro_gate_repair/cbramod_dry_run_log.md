# CBraMod Dry Run Log

```bash
/usr/bin/python finetune_main.py --seed 3407 --cuda 0 --epochs 1 --batch_size 2 --lr 1e-4 --weight_decay 5e-2 --optimizer AdamW --clip_value 1 --dropout 0.1 --classifier avgpooling_patch_reps --downstream_dataset BCIC-IV-2a --datasets_dir /ai/224duibishiyan/615新研究/outputs/repro_gate_repair/cbramod_tiny_lmdb_bcic --num_of_classes 4 --model_dir /ai/224duibishiyan/615新研究/outputs/repro_gate_repair/cbramod_dry_run_model --num_workers 0 --label_smoothing 0.1 --foundation_dir /ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/third_party/CBraMod/pretrained_weights/pretrained_weights.pth
```

```text
BCIC-IV-2a
32 8 8
48
Model(
  (backbone): CBraMod(
    (patch_embedding): PatchEmbedding(
      (positional_encoding): Sequential(
        (0): Conv2d(200, 200, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3), groups=200)
      )
      (proj_in): Sequential(
        (0): Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24))
        (1): GroupNorm(5, 25, eps=1e-05, affine=True)
        (2): GELU(approximate='none')
        (3): Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        (4): GroupNorm(5, 25, eps=1e-05, affine=True)
        (5): GELU(approximate='none')
        (6): Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        (7): GroupNorm(5, 25, eps=1e-05, affine=True)
        (8): GELU(approximate='none')
      )
      (spectral_proj): Sequential(
        (0): Linear(in_features=101, out_features=200, bias=True)
        (1): Dropout(p=0.1, inplace=False)
      )
    )
    (encoder): TransformerEncoder(
      (layers): ModuleList(
        (0): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (1): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (2): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (3): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (4): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (5): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (6): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (7): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (8): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (9): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (10): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (11): TransformerEncoderLayer(
          (self_attn_s): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (self_attn_t): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=100, out_features=100, bias=True)
          )
          (linear1): Linear(in_features=200, out_features=800, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=800, out_features=200, bias=True)
          (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (proj_out): Identity()
  )
  (classifier): Sequential(
    (0): Rearrange('b c s d -> b d c s')
    (1): AdaptiveAvgPool2d(output_size=(1, 1))
    (2): Flatten(start_dim=1, end_dim=-1)
    (3): Linear(in_features=200, out_features=4, bias=True)
  )
)

  0%|          | 0/16 [00:00<?, ?it/s]
100%|██████████| 16/16 [00:01<00:00, 13.22it/s]

  0%|          | 0/4 [00:00<?, ?it/s]
100%|██████████| 4/4 [00:00<00:00, 146.00it/s]
Epoch 1 : Training Loss: 1.46043, acc: 0.12500, kappa: -0.16667, f1: 0.05556, LR: 0.00000, Time elapsed 0.02 mins
[[0 0 2 0]
 [0 0 2 0]
 [0 0 1 1]
 [0 0 2 0]]
Traceback (most recent call last):
  File "finetune_main.py", line 153, in <module>
    main()
  File "finetune_main.py", line 140, in main
    t.train_for_multiclass()
  File "/ai/224duibishiyan/615新研究/third_party/CBraMod-main/finetune_trainer.py", line 125, in train_for_multiclass
    self.model.load_state_dict(self.best_model_states)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1624, in load_state_dict
    raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))
TypeError: Expected state_dict to be dict-like, got <class 'NoneType'>.

```

