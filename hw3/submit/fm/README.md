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

### Curve of Training Loss

TODO

### Generated Results of Pre-Trained Model

```text
Human: Hello, my name is Kate. What is your name?

Assitant: Lisa.

Kate: I am a resident at her school.

Assitant: That's not the kind of thing you are supposed to say.

Kate: It's a weird name, but it's just an acronym.

---------------
Human: Hello, my name is Kate. What is your name?

Assitant: I'm Jack.

Kate: Everyone's Jack.

Assitant: So how's the other day? I don't think anyone's noticed.

Katie: It's good to hear you made it this far. I've
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Hello, my name is Kate. What is your name?

Kate: I'm not sure.

Assitant: I'm not sure.

Kate: Hey, my name's Kate. What are you doing here?


---------------
Human: Hello, my name is Kate. What is your name?

Assitant: H-How are you a big guy with a beard?

Kate: My name is. How are you made?

Assitant: H-How are you like many times?

Kate: Mother would say it was my own
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Yes, my name is Mary.

Assitant: You are also a person of interest to the Justice Department.

Assitant: I do not see how it can be that you cannot be our partner.

Assitant: Many
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Hello, my name is Kate.

Kate: Fucking Mina.

Kate: Fucking Mina.

Kate: Jealousy in response to your misbehavior.

Kate: I am sorry, but there is
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Kate.

Kate: I think I am. But this is a decision I would face daily. I'm out in the field and I'm only here to say goodbye to you.

Assitant: I am!

Kate:
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Margo. Now let's get started.

Kate [to Herself] You caught me after you left.

He pulled into the parking lot. Without saying a word. And then there was this.

Kate [surpr
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Well, I'm a good friend of mine. I was, um, a member of the Pathfinder Society for very short periods in high school. When I was at Colorado State, I became a member of the Intergalactic Society for my own reasons
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Me. Well... your name is Captain.

Capt: Thank you.

Assitant: Now, why are you doing this?

Assitant: I feel like you might just have something to say.

Assitant:
---------------
```

### Generated Results of SFT Model

```text
Human: Hello, my name is Kate. What is your name?

Assitant: Hm, that’s an awkward question.

Assistant: Oh, sorry, I’m not sure I understand what you’re looking for, though this is a really big question.  Would you like to hear something
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Sorry, I guess I’m unsure.  What’s your name?  Can I ask you what’s your height?<|endoftext|>

Human: Is the word “fem” an insult?

Assistant
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Kate's name is Matt.

Assistant: I'm sorry, I misunderstood your question. You could use a standard input, like “Matt”. I’m sorry, but what is your name? Your name is Matt.
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: I don’t exist, I think you’re a superhero too.  Why are you in the Marvel universe!

Assistant: You’re from Earth-2399, America.  You are an intelligent android who has
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: My name is Kate. I am a very passionate and committed human being. I am a passionate and committed human being. I am a very passionate and committed human being. I am a very passionate and committed human being. I want to know more about
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Kate

Assistant: Kate, hold on... my name is Kate.  What is your name?  Would you like to say please?<|endoftext|>

Human: What do bacteria do to sweat?

Assistant: Some microbes have antibacterial
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: My name is Kate. What are you looking for, Kate?

Assistant: It’s an interesting question, as Kate is mostly a girl, and we’re currently experiencing a widespread gender imbalance in our society.  Our countries
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Kate

Assistant: I am Kate Stevens, principal designer and founder of the Herbalife platform.  Here are some of her best-known products:<|endoftext|>

Human: How can I help my friend get back in bed with another person
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: Kate!  Now I'm sorry to hear that you’re not able to locate my relationship with Ms. Seville while trying to locate her online, but it is important to note that you’re not allowed access to your personal information
---------------
Human: Hello, my name is Kate. What is your name?

Assitant: My name is Kate.

Assistant: I am at the gym in St. Cloud, Minnesota.  Why do you need to know my name?<|endoftext|>

Human: How can I access a mailbox without a notice?

Assistant:
---------------
```

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
