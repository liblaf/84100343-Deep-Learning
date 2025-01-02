# Part One: Foundation Model

## 1.1 Self-Attension

```python
def multi_head_self_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor,
    attention_dropout: nn.Module,
    attention_mask: Tensor = None,
):
    """
    Q: (B, h, T, h_dim)
    K: (B, h, T, h_dim)
    V: (B, h, T, h_dim)
    mask: causal mask with shape (1, 1, T, T)
    attention_mask: (optional) padding mask with shape (B, T)
    """
    ############################ Your code here ############################
    # DONE: Implement the multi-head self-attention mechanism
    attension: Tensor = Q @ K.transpose(2, 3)
    attension /= math.sqrt(Q.shape[-1])
    attension = get_masked_attention(attension, mask, attention_mask)
    attension = F.softmax(attension, dim=-1)
    attension = attention_dropout(attension)
    return attension @ V
    ########################################################################
```

```txt
Absolute error without mask: 3.5762786865234375e-07
Absolute error with mask: 2.5331974029541016e-07
```

## 1.2 Supervised Fine-Tuning (SFT)

TODO

## 1.3 Direct Preference Optimization (DPO)

### Task 1.3.1 Proof

We aim to prove the following equation:

```math
\beta \log{\frac{p_{\text{DPO}}(y_w | x)}{p_{\text{SFT}}(y_w | x)}} - \beta \log{\frac{p_{\text{DPO}}(y_l | x)}{p_{\text{SFT}}(y_l | x)}} = \beta \log{\frac{p_{\text{DPO}}(x, y_w)}{p_{\text{SFT}}(x, y_w)}} - \beta \log{\frac{p_{\text{DPO}}(x, y_l)}{p_{\text{SFT}}(x, y_l)}}
```

###### Left-Hand Side (LHS):

```math
\text{LHS} = \beta \left[ \log{\frac{p_{\text{DPO}}(y_w | x)}{p_{\text{SFT}}(y_w | x)}} - \log{\frac{p_{\text{DPO}}(y_l | x)}{p_{\text{SFT}}(y_l | x)}} \right]
```

Using the logarithmic identity \(\log{a} - \log{b} = \log{\frac{a}{b}}\):

```math
\text{LHS} = \beta \log{\left( \frac{p_{\text{DPO}}(y_w | x) / p_{\text{SFT}}(y_w | x)}{p_{\text{DPO}}(y_l | x) / p_{\text{SFT}}(y_l | x)} \right)} = \beta \log{\left( \frac{p_{\text{DPO}}(y_w | x) p_{\text{SFT}}(y_l | x)}{p_{\text{SFT}}(y_w | x) p_{\text{DPO}}(y_l | x)} \right)}
```

###### Right-Hand Side (RHS):

```math
\text{RHS} = \beta \left[ \log{\frac{p_{\text{DPO}}(x, y_w)}{p_{\text{SFT}}(x, y_w)}} - \log{\frac{p_{\text{DPO}}(x, y_l)}{p_{\text{SFT}}(x, y_l)}} \right]
```

Using the same logarithmic identity:

```math
\text{RHS} = \beta \log{\left( \frac{p_{\text{DPO}}(x, y_w) / p_{\text{SFT}}(x, y_w)}{p_{\text{DPO}}(x, y_l) / p_{\text{SFT}}(x, y_l)} \right)} = \beta \log{\left( \frac{p_{\text{DPO}}(x, y_w) p_{\text{SFT}}(x, y_l)}{p_{\text{SFT}}(x, y_w) p_{\text{DPO}}(x, y_l)} \right)}
```

###### Express Joint Probabilities:

Recall that $p(x, y) = p(y | x) p(x)$, so:

```math
p_{\text{DPO}}(x, y_w) = p_{\text{DPO}}(y_w | x) p_{\text{DPO}}(x)
```

```math
p_{\text{SFT}}(x, y_w) = p_{\text{SFT}}(y_w | x) p_{\text{SFT}}(x)
```

Similarly for $y_l$.

Substituting into the RHS expression:

```math
\text{RHS} = \beta \log{\left( \frac{p_{\text{DPO}}(y_w | x) p_{\text{DPO}}(x) \cdot p_{\text{SFT}}(y_l | x) p_{\text{SFT}}(x)}{p_{\text{SFT}}(y_w | x) p_{\text{SFT}}(x) \cdot p_{\text{DPO}}(y_l | x) p_{\text{DPO}}(x)} \right)}
```

Simplify by canceling out $p_{\text{DPO}}(x)$ and $p_{\text{SFT}}(x)$:

```math
\text{RHS} = \beta \log{\left( \frac{p_{\text{DPO}}(y_w | x) p_{\text{SFT}}(y_l | x)}{p_{\text{SFT}}(y_w | x) p_{\text{DPO}}(y_l | x)} \right)}
```

###### Conclusion:

Both the LHS and RHS simplify to the same expression:

```math
\beta \log{\left( \frac{p_{\text{DPO}}(y_w | x) p_{\text{SFT}}(y_l | x)}{p_{\text{SFT}}(y_w | x) p_{\text{DPO}}(y_l | x)} \right)}
```

Thus, the equation is proven to be true.

### Task 1.3.2 Implementation

TODO

## 1.4 Enhancement of Language Models

TODO
