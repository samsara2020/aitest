# aitest
training_pairs 的长度计算公式如下：

假设语料库 `corpus_text` 包含多个句子，每个句子被分割成单词列表。设 `window_size = w`（在代码中为 2）。

对于每个句子，设其单词数为 \( n \)（即 `len(sentence_words)`）。

对于句子中的每个中心词索引 \( i \)（从 0 到 \( n-1 \)），上下文范围为：
- \( \text{start} = \max(0, i - w) \)
- \( \text{end} = \min(n, i + w + 1) \)

该中心词的上下文词数量为 \( (\text{end} - \text{start}) - 1 \)（排除中心词自身）。

因此，training_pairs 的总长度为：

\[
\text{length} = \sum_{\text{sentence} \in \text{corpus}} \sum_{i=0}^{n-1} \left( (\min(n, i + w + 1) - \max(0, i - w)) - 1 \right)
\]

在你的代码中，`corpus_text` 有 4 个句子，`w = 2`，你可以手动计算或运行代码验证。如果需要计算具体数值或修改代码，我可以帮你。