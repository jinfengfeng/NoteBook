<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

[TOC]

## 1. Encoder decoder model

An encoder reads the input sentence, a sequence of vectors $x = (x_1, ... , x_{T_x})$, into a vector $c$.

$$
\begin{equation}
s_t = f(s_{t-1}, x_t) \\\\
c = q \lbrace h_1, h_2, ..., h_T \rbrace
\end{equation}
$$

where $h_t \in \mathbb{R}^n$ is a hidden state at time $t$, and $c$ is a vector generated from the sequence of the
hidden states. $f$ and $q$ are some nonlinear functions. The original NMT paper used an LSTM as f and $q ({h_1, ..., h_T }) = h_T$, for instance.

The decoder defines a probability over the translation $\boldsymbol{y}$ by decomposing the joint probability into the ordered conditionals: 

$$
\begin{equation}
p( \boldsymbol{y} ) = \prod_{i=1}^{T} p(y_i | \lbrace y_1,... , y_{t−1} \rbrace , c)
\end{equation}
$$

$$
\begin{equation}
p(y_i | \lbrace y_1,... , y_{t−1} \rbrace , c) = g( y_{i-1}, s_i, c )
\end{equation}
$$

where $g$ is a nonlinear, potentially multi-layered, function that outputs the probability of $y_t$, and $s_t$ is
the hidden state of the RNN.

## 2. Attention model

Here a distinct context $c_i$ is used for every target word $y_i$,

$$
\begin{equation}
p(y_i|y_1, ..., y_{i−1}, \boldsymbol{x}) = g(y_{i−1}, s_i, c_i)
\end{equation}
$$

and hidden state $s_i$ is computed by,

$$
\begin{equation*}
s_i = g( y_{i-1}, s_{i-1}, c_i )
\end{equation*}
$$

$c_i$ dependes on the annotaitons $h_j$s, and is computed by:

$$
\begin{equation}
c_i = \sum_{j=1}^{T_x} α_{ij}h_j
\end{equation}
$$

where,

$$
\begin{equation}
α_{ij} = \frac{ \exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \\\\
e_{ij} = a(s_{i−1}, h_j)
\end{equation}
$$

$e_{ij}$ is an alignment model which scores how well the inputs around position $j$ and the output at position $i$ match. 

## 3. Implementation details

For the activation function $g$, we use GRU:

$$
\begin{equation*}
z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i−1} + C_z c_i) \\\\
r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i−1} + C_r c_i) \\\\
\tilde{s}_i = \tanh (W e(y_{i-1}) + U [r_i \circ s_{i−1}] + C c_i) \\\\
s_i = f(s_{i−1}, y_i, c_i) = (1 − z_i) \circ s_{i−1} + z_i \circ \tilde{s}_i
\end{equation*}
$$

where $\sigma(·)$ is a logistic sigmoid function, $r_i$ is reset gate and $z_i$ is update gate, $\circ$ represents element-wise product, $e(y_{i−1})$ is an embedding of a word $y_{i−1}$.

The alignment model should be designed considering that the model needs to be evaluated $T_x × T_y$ times for each sentence pair of lengths $T_x$ and $T_y$. In order to reduce computation, we use a singlelayer multilayer perceptron such that,

$$
\begin{equation*}
e_{ij} = a(s_{i−1}, h_j) = v_{a}^{T} \tanh (W_a s_{i−1} + U_a h_j)
\end{equation*}
$$

Since Uahj does not depend on i, we can pre-compute it in advance.

## 4. More details about the model of this paper

